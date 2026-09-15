#!/usr/bin/env python3
"""Taste-biased snowball — collect users whose libraries overlap a target library.

Historical explicit-execution research tooling; not a supported workflow.
Read docs/COMPONENTS.md and docs/DATA_AND_RIGHTS.md before considering use.
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH, MAX_TOP_TRACKS_PAGES, TOP_TRACKS_PER_PAGE
from src.db import (
    get_connection,
    init_db,
    transaction,
    migrate_snowball_columns,
    upsert_tracks_batch,
    add_user_tracks_batch,
    upsert_seed_progress,
    get_seed_progress,
    upsert_candidate_progress,
    get_evaluated_candidates,
    get_cohort_passers,
    get_cohort_summary,
)
from src.lastfm_api import LastFMClient
from src.scraper import ListenerScraper
from harness.backend.matching import normalize_artist, normalize_title

LOG_DIR = Path(__file__).parent.parent / "data"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "05_taste_snowball.log"),
    ],
)
logger = logging.getLogger(__name__)


# Write-time activity filter. A user that clears the overlap threshold but has
# a near-empty library (1 track in their top-tracks ⇒ trivially saturates the
# artist-overlap fraction) provides no usable training signal. Drop them at
# collection time so they don't accumulate in the DB and have to be cleaned up
# later.
MIN_LIBRARY_TRACKS = 50
MIN_LIBRARY_PLAYS = 500


# ---------- Strategy configs --------------------------------------------------

@dataclass(frozen=True)
class Strategy:
    """Per-cohort tuning. See docs/EXPERIMENTS.md for rationale per cohort."""

    name: str
    track_threshold: float
    artist_threshold: float
    max_pages_per_seed: int
    candidates_per_seed: int          # stratified eval cap
    seed_kind: str                    # 'artist' or 'user' (friend-expand only)
    cohort_tag: str                   # what to write into users.cohort


STRATEGIES: Dict[str, Strategy] = {
    "core-breadth": Strategy(
        name="core-breadth",
        track_threshold=0.50,
        artist_threshold=0.65,
        max_pages_per_seed=30,
        # 250 = the empirical Last.fm cap on unique listeners per artist page;
        # evaluating fewer leaves discovered candidates on the floor.
        candidates_per_seed=250,
        seed_kind="artist",
        cohort_tag="core-breadth",
    ),
    "region-fill": Strategy(
        name="region-fill",
        track_threshold=0.20,
        artist_threshold=0.30,
        # 30 pages is past Last.fm's unique-listener cap; more is wasted scraping.
        max_pages_per_seed=30,
        candidates_per_seed=250,
        seed_kind="artist",
        cohort_tag="region-fill",
    ),
    "niche-deep": Strategy(
        name="niche-deep",
        track_threshold=0.20,
        artist_threshold=0.30,
        max_pages_per_seed=30,
        candidates_per_seed=250,
        seed_kind="artist",
        cohort_tag="niche-deep",
    ),
    "friend-expand": Strategy(
        # threshold + cohort_tag get overridden at runtime from --source-cohort
        name="friend-expand",
        track_threshold=0.0,
        artist_threshold=0.0,
        max_pages_per_seed=1,
        candidates_per_seed=150,
        seed_kind="user",
        cohort_tag="friend-expand",
    ),
}

# Region-fill seeds must be supplied explicitly with --region-seeds.
# Personal-library-derived defaults are excluded from this snapshot.
DEFAULT_REGION_FILL_SEEDS = []


@dataclass
class SeedSpec:
    """One seed to process — either an artist (scrape listeners) or a user (fetch friends)."""

    kind: str             # 'artist' or 'user'
    name: str             # artist name or username
    listeners: int = 0    # last.fm listener count (artist seeds only; for reporting)


# ---------- Library loading ---------------------------------------------------

@dataclass
class TargetLibrary:
    library_id: int
    tracks: Set[Tuple[str, str]]                  # normalized (artist, title)
    artists: Set[str]                             # normalized artists
    artist_stats: Dict[str, Tuple[str, int, int]] # norm -> (display, plays, tracks)


def load_target_library(library_id: int) -> TargetLibrary:
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT artist, title, play_count
               FROM harness_library_tracks
               WHERE library_id = ? AND artist IS NOT NULL AND title IS NOT NULL""",
            (library_id,),
        ).fetchall()
    if not rows:
        raise SystemExit(f"library_id {library_id} has no tracks in harness_library_tracks")

    tracks: Set[Tuple[str, str]] = set()
    artists: Set[str] = set()
    plays: Dict[str, int] = {}
    track_counts: Dict[str, int] = {}
    display: Dict[str, Counter] = {}

    for r in rows:
        a = normalize_artist(r["artist"] or "")
        t = normalize_title(r["title"] or "")
        if not a or not t:
            continue
        tracks.add((a, t))
        artists.add(a)
        plays[a] = plays.get(a, 0) + int(r["play_count"] or 0)
        track_counts[a] = track_counts.get(a, 0) + 1
        display.setdefault(a, Counter())[r["artist"]] += 1

    artist_stats = {
        a: (display[a].most_common(1)[0][0], plays[a], track_counts[a])
        for a in artists
    }
    logger.info(
        "Target library %d: %d tracks, %d artists",
        library_id, len(tracks), len(artists),
    )
    return TargetLibrary(library_id, tracks, artists, artist_stats)


# ---------- Seed selection ----------------------------------------------------

def select_top_artists_by_plays(
    library: TargetLibrary, client: LastFMClient,
    rank_start: int, rank_end: int,
    min_listeners: int = 0, max_listeners: Optional[int] = None,
    n: Optional[int] = None,
) -> List[SeedSpec]:
    """Rank library artists by total play count, slice [rank_start:rank_end],
    optionally filter by listener band, return up to n SeedSpecs."""
    ranked = sorted(library.artist_stats.items(), key=lambda kv: -kv[1][1])
    sliced = ranked[rank_start:rank_end]
    out: List[SeedSpec] = []
    for a_norm, (a_display, plays, tracks) in sliced:
        if n is not None and len(out) >= n:
            break
        info = client.get_artist_info(a_display)
        if info is None:
            logger.debug("  no artist info for %s", a_display)
            continue
        listeners = info["listeners"]
        if listeners < min_listeners:
            continue
        if max_listeners is not None and listeners > max_listeners:
            continue
        out.append(SeedSpec(kind="artist", name=a_display, listeners=listeners))
        logger.info(
            "  seed: %-35s listeners=%-9d plays=%-5d tracks=%d",
            a_display[:35], listeners, plays, tracks,
        )
    return out


def get_seeds_for_strategy(
    strategy: Strategy, library: TargetLibrary, client: LastFMClient,
    args: argparse.Namespace,
) -> List[SeedSpec]:
    if strategy.name == "core-breadth":
        # Top N artists by plays, no listener cap (the breadth filter is the threshold).
        logger.info("Picking top %d core-breadth seeds", args.n_seeds)
        return select_top_artists_by_plays(
            library, client,
            rank_start=0, rank_end=args.seed_pool,
            n=args.n_seeds,
        )

    if strategy.name == "region-fill":
        # Hand-picked seeds from CLI or defaults.
        names = args.region_seeds or DEFAULT_REGION_FILL_SEEDS
        logger.info("Probing %d region-fill seeds for listener counts", len(names))
        out: List[SeedSpec] = []
        for name in names:
            info = client.get_artist_info(name)
            listeners = info["listeners"] if info else 0
            out.append(SeedSpec(kind="artist", name=name, listeners=listeners))
            logger.info("  seed: %-35s listeners=%d", name[:35], listeners)
        return out

    if strategy.name == "niche-deep":
        # Library artists ranked 50-150 with ≥20K listeners.
        logger.info(
            "Picking niche-deep seeds (library rank %d-%d, ≥%d listeners)",
            args.niche_rank_start, args.niche_rank_end, args.niche_min_listeners,
        )
        return select_top_artists_by_plays(
            library, client,
            rank_start=args.niche_rank_start,
            rank_end=args.niche_rank_end,
            min_listeners=args.niche_min_listeners,
            n=args.n_seeds,
        )

    if strategy.name == "friend-expand":
        # Seeds are usernames — passers from --source-cohort.
        with get_connection() as conn:
            passers = get_cohort_passers(conn, args.source_cohort)
        if not passers:
            raise SystemExit(
                f"no passers found for source-cohort '{args.source_cohort}' — "
                f"build that cohort first",
            )
        logger.info(
            "Friend-expanding from %d passers in cohort '%s'",
            len(passers), args.source_cohort,
        )
        return [SeedSpec(kind="user", name=u) for u in passers]

    raise ValueError(f"unknown strategy: {strategy.name}")


# ---------- Overlap evaluation ------------------------------------------------

def compute_overlap(
    candidate_top: List[dict], target: TargetLibrary,
) -> Tuple[float, float]:
    """Return (track_overlap_frac, artist_overlap_frac) for a candidate's top tracks."""
    if not candidate_top:
        return 0.0, 0.0
    track_hits = 0
    cand_artists: Set[str] = set()
    n = 0
    for t in candidate_top:
        a = normalize_artist(t.get("artist") or "")
        ti = normalize_title(t.get("name") or "")
        if not a or not ti:
            continue
        n += 1
        cand_artists.add(a)
        if (a, ti) in target.tracks:
            track_hits += 1
    track_frac = track_hits / n if n else 0.0
    artist_frac = (
        len(cand_artists & target.artists) / len(cand_artists) if cand_artists else 0.0
    )
    return track_frac, artist_frac


def fetch_full_top_tracks(
    api: LastFMClient, username: str,
) -> Optional[List[Tuple[str, str, str, int]]]:
    """Fetch a user's full top tracks across pages, matching 02_collect_top_tracks.

    Returns list of (lastfm_id, name, artist, playcount) or None if unfetchable.
    """
    out: List[Tuple[str, str, str, int]] = []
    seen: Set[str] = set()
    for page in range(1, MAX_TOP_TRACKS_PAGES + 1):
        tracks = api.get_user_top_tracks(
            username, period="overall", page=page, limit=TOP_TRACKS_PER_PAGE,
        )
        if tracks is None:
            return None if page == 1 else out
        if not tracks:
            break
        for t in tracks:
            lastfm_id = f"{t['artist']}::{t['name']}"
            if lastfm_id in seen:
                continue
            seen.add(lastfm_id)
            out.append((lastfm_id, t["name"], t["artist"], t["playcount"]))
        if len(tracks) < TOP_TRACKS_PER_PAGE:
            break
    return out


# ---------- Per-seed processing -----------------------------------------------

@dataclass
class SeedResult:
    seed: SeedSpec
    pages_scraped: int = 0
    discovered: int = 0
    evaluated: int = 0
    passed: int = 0
    skipped_existing: int = 0
    skipped_already_evaluated: int = 0


def discover_candidates(
    seed: SeedSpec, scraper: ListenerScraper, api: LastFMClient,
    strategy: Strategy,
) -> List[str]:
    """Discover candidate usernames for a seed.

    Artist seeds: scrape up to `max_pages_per_seed` listener pages. The scraper
    has its own early-stop when a page returns no usernames or fails to load.
    User seeds (friend-expand): fetch the user's friend list once.
    """
    if seed.kind == "user":
        friends = api.get_user_friends(seed.name, limit=200) or []
        seed._pages_scraped_actual = 1
        return list(dict.fromkeys(friends))

    usernames = scraper.scrape_artist_listeners(
        seed.name, max_pages=strategy.max_pages_per_seed,
    )
    seed._pages_scraped_actual = scraper.last_pages_scraped
    return list(usernames)


def process_seed(
    seed: SeedSpec, strategy: Strategy, library: TargetLibrary,
    scraper: ListenerScraper, api: LastFMClient,
    existing_users: Set[str],
) -> SeedResult:
    """Run discovery + evaluation + collection for one seed."""
    res = SeedResult(seed=seed)
    conn = get_connection()
    try:
        prev = get_seed_progress(conn, strategy.cohort_tag, seed.name)
        if prev is not None and prev["completed_at"]:
            logger.info("  %s already completed, skipping", seed.name)
            res.pages_scraped = prev["pages_scraped"]
            res.discovered = prev["candidates_discovered"]
            res.evaluated = prev["candidates_evaluated"]
            res.passed = prev["candidates_passed"]
            return res

        candidates = discover_candidates(seed, scraper, api, strategy)
        res.pages_scraped = getattr(seed, "_pages_scraped_actual", 1)
        res.discovered = len(candidates)
        logger.info(
            "  %s: discovered %d candidates from %d pages",
            seed.name, len(candidates), res.pages_scraped,
        )

        # Filter: drop candidates already in users table OR already evaluated this cohort.
        evaluated_already = get_evaluated_candidates(conn, strategy.cohort_tag)
        to_evaluate: List[str] = []
        for u in candidates:
            ul = u.lower()
            if ul in existing_users:
                res.skipped_existing += 1
                continue
            if u in evaluated_already:
                res.skipped_already_evaluated += 1
                continue
            to_evaluate.append(u)

        # Stratified cap per seed.
        random.shuffle(to_evaluate)
        to_evaluate = to_evaluate[:strategy.candidates_per_seed]
        logger.info(
            "  %s: %d new for evaluation (%d dupes, %d already-evaluated)",
            seed.name, len(to_evaluate),
            res.skipped_existing, res.skipped_already_evaluated,
        )

        # Evaluate each candidate.
        for i, username in enumerate(to_evaluate, 1):
            top = api.get_user_top_tracks(username, period="overall", page=1, limit=200)
            if not top:
                with transaction(conn):
                    upsert_candidate_progress(
                        conn, strategy.cohort_tag, username, "failed",
                        seed_artist=seed.name,
                    )
                continue

            track_frac, artist_frac = compute_overlap(top, library)
            passed = (
                track_frac >= strategy.track_threshold
                or artist_frac >= strategy.artist_threshold
            )
            res.evaluated += 1

            if not passed:
                with transaction(conn):
                    upsert_candidate_progress(
                        conn, strategy.cohort_tag, username, "evaluated",
                        seed_artist=seed.name,
                        overlap_track_pct=track_frac,
                        overlap_artist_pct=artist_frac,
                    )
                continue

            # Passer: collect full top 500, write user + tracks, mark passed.
            full = fetch_full_top_tracks(api, username)
            if not full:
                with transaction(conn):
                    upsert_candidate_progress(
                        conn, strategy.cohort_tag, username, "failed",
                        seed_artist=seed.name,
                        overlap_track_pct=track_frac,
                        overlap_artist_pct=artist_frac,
                    )
                continue

            # Activity filter: a user can pass the overlap threshold with a
            # near-empty library (e.g., 1 track in their top-tracks ⇒ 100%
            # artist overlap, trivially). Such users teach the model nothing.
            # Drop them at write time and mark with a distinct status so the
            # cohort dedup mechanism still skips them on resume.
            total_plays = sum(t[3] for t in full)
            if len(full) < MIN_LIBRARY_TRACKS or total_plays < MIN_LIBRARY_PLAYS:
                logger.info(
                    "  %s: low_activity %s (tracks=%d plays=%d) — skipping",
                    seed.name, username, len(full), total_plays,
                )
                with transaction(conn):
                    upsert_candidate_progress(
                        conn, strategy.cohort_tag, username, "low_activity",
                        seed_artist=seed.name,
                        overlap_track_pct=track_frac,
                        overlap_artist_pct=artist_frac,
                    )
                continue

            with transaction(conn):
                # Insert the user with cohort tagging. Use INSERT OR IGNORE on
                # username — if some race added them, we still want our cohort
                # tag, so follow up with UPDATE.
                conn.execute(
                    """INSERT OR IGNORE INTO users
                       (username, source, status, cohort, seed_artist,
                        overlap_track_pct, overlap_artist_pct, overlap_library_id)
                       VALUES (?, ?, 'collected', ?, ?, ?, ?, ?)""",
                    (
                        username, f"snowball:{strategy.cohort_tag}:{seed.name}",
                        strategy.cohort_tag, seed.name,
                        track_frac, artist_frac, library.library_id,
                    ),
                )
                # If they already existed, update the cohort metadata (don't
                # overwrite an existing cohort that isn't 'original' — that
                # would let later passes clobber earlier tagging).
                conn.execute(
                    """UPDATE users SET
                           cohort = CASE WHEN cohort IN ('original', NULL)
                                         THEN ? ELSE cohort END,
                           seed_artist = COALESCE(seed_artist, ?),
                           overlap_track_pct = COALESCE(overlap_track_pct, ?),
                           overlap_artist_pct = COALESCE(overlap_artist_pct, ?),
                           overlap_library_id = COALESCE(overlap_library_id, ?)
                       WHERE username = ?""",
                    (
                        strategy.cohort_tag, seed.name,
                        track_frac, artist_frac, library.library_id,
                        username,
                    ),
                )
                user_row = conn.execute(
                    "SELECT id FROM users WHERE username = ?", (username,),
                ).fetchone()
                user_id = user_row["id"]

                track_rows = [(t[0], t[1], t[2]) for t in full]
                upsert_tracks_batch(conn, track_rows)
                ut_rows = [(user_id, t[0], t[3]) for t in full]
                add_user_tracks_batch(conn, ut_rows)

                upsert_candidate_progress(
                    conn, strategy.cohort_tag, username, "passed",
                    seed_artist=seed.name,
                    overlap_track_pct=track_frac,
                    overlap_artist_pct=artist_frac,
                )

            res.passed += 1
            existing_users.add(username.lower())

            if i % 20 == 0:
                logger.info(
                    "  %s: [%d/%d] eval=%d passed=%d",
                    seed.name, i, len(to_evaluate), res.evaluated, res.passed,
                )

        # Mark seed completed.
        with transaction(conn):
            upsert_seed_progress(
                conn, strategy.cohort_tag, seed.name,
                pages_scraped=res.pages_scraped,
                candidates_discovered=res.discovered,
                candidates_evaluated=res.evaluated,
                candidates_passed=res.passed,
                completed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
        return res
    finally:
        conn.close()


# ---------- Reporting ---------------------------------------------------------

def report(
    strategy: Strategy, library: TargetLibrary,
    seed_results: List[SeedResult], elapsed: float,
    api_calls: int,
) -> None:
    print()
    print("=" * 78)
    print(f" SNOWBALL REPORT — cohort '{strategy.cohort_tag}'")
    print("=" * 78)

    total_pages = sum(r.pages_scraped for r in seed_results)
    total_disc = sum(r.discovered for r in seed_results)
    total_eval = sum(r.evaluated for r in seed_results)
    total_pass = sum(r.passed for r in seed_results)

    print(f"\nseeds processed:         {len(seed_results)}")
    print(f"pages scraped:           {total_pages}")
    print(f"candidates discovered:   {total_disc}")
    print(f"candidates evaluated:    {total_eval}")
    print(f"candidates passed:       {total_pass}"
          + (f" ({total_pass / total_eval:.1%})" if total_eval else ""))
    print(f"thresholds:              track≥{strategy.track_threshold:.0%}"
          f" OR artist≥{strategy.artist_threshold:.0%}")
    print(f"wall time:               {elapsed / 60:.1f} min")
    print(f"last.fm api calls:       {api_calls}")

    print("\n[per-seed breakdown]")
    print(f"  {'seed':<35} {'pages':>5} {'disc':>5} {'eval':>5} {'pass':>5}"
          f" {'rate':>6}")
    for r in sorted(seed_results, key=lambda x: -x.passed):
        rate = (r.passed / r.evaluated) if r.evaluated else 0.0
        print(f"  {r.seed.name[:35]:<35} {r.pages_scraped:>5d}"
              f" {r.discovered:>5d} {r.evaluated:>5d} {r.passed:>5d}"
              f" {rate:>5.1%}")

    # Cumulative cohort stats from DB (includes any prior runs of this cohort).
    with get_connection() as conn:
        summary = get_cohort_summary(conn, strategy.cohort_tag)
        total_in_users = conn.execute(
            "SELECT COUNT(*) AS c FROM users WHERE cohort = ?",
            (strategy.cohort_tag,),
        ).fetchone()["c"]

    print("\n[cumulative cohort state]")
    print(f"  taste_snowball_progress: {summary}")
    print(f"  users tagged '{strategy.cohort_tag}': {total_in_users}")
    print("=" * 78)


# ---------- Friend-expand override --------------------------------------------

def resolve_strategy(args: argparse.Namespace) -> Strategy:
    """For friend-expand, inherit threshold + cohort tag from --source-cohort.

    Supports chained expansion: --source-cohort can be either an artist
    strategy ('niche-deep') or an existing friend-expand cohort
    ('friend-expand-niche-deep', 'friend-expand-2-niche-deep', ...). Each
    chain step bumps the depth counter in the resulting cohort_tag.

    For other strategies, return the static Strategy unchanged.
    """
    base = STRATEGIES[args.strategy]
    if args.strategy != "friend-expand":
        return base

    src = args.source_cohort

    # Parse source to find underlying artist strategy and chain depth.
    #   "niche-deep"                 -> artist=niche-deep, next_depth=1
    #   "friend-expand-niche-deep"   -> artist=niche-deep, next_depth=2
    #   "friend-expand-2-niche-deep" -> artist=niche-deep, next_depth=3
    m = re.match(r"^friend-expand-(\d+)-(.+)$", src)
    if m:
        next_depth = int(m.group(1)) + 1
        artist_src = m.group(2)
    elif src.startswith("friend-expand-"):
        next_depth = 2
        artist_src = src[len("friend-expand-"):]
    else:
        next_depth = 1
        artist_src = src

    if artist_src not in STRATEGIES or STRATEGIES[artist_src].seed_kind != "artist":
        raise SystemExit(
            f"--source-cohort must resolve to an artist strategy "
            f"(core-breadth, region-fill, niche-deep). "
            f"Got source={src!r}, resolved artist={artist_src!r}",
        )

    src_strategy = STRATEGIES[artist_src]
    cohort_tag = (
        f"friend-expand-{artist_src}" if next_depth == 1
        else f"friend-expand-{next_depth}-{artist_src}"
    )
    return Strategy(
        name="friend-expand",
        track_threshold=src_strategy.track_threshold,
        artist_threshold=src_strategy.artist_threshold,
        max_pages_per_seed=base.max_pages_per_seed,
        candidates_per_seed=base.candidates_per_seed,
        seed_kind="user",
        cohort_tag=cohort_tag,
    )


# ---------- Existing-user fingerprint ----------------------------------------

def load_existing_usernames() -> Set[str]:
    with get_connection() as conn:
        rows = conn.execute("SELECT username FROM users").fetchall()
    return {r["username"].lower() for r in rows}


# ---------- Main --------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--strategy", required=True,
        choices=sorted(STRATEGIES.keys()),
        help="which cohort to build",
    )
    ap.add_argument("--library-id", type=int, default=1,
                    help="harness_libraries.id — the target library")
    ap.add_argument("--n-seeds", type=int, default=30,
                    help="number of seeds to use (artist strategies only)")
    ap.add_argument("--seed-pool", type=int, default=60,
                    help="how deep into the artist ranking to probe for seeds")
    ap.add_argument(
        "--region-seeds", nargs="+", default=None,
        help="region-fill: explicit list of seed artists (overrides defaults)",
    )
    ap.add_argument("--niche-rank-start", type=int, default=50,
                    help="niche-deep: library artist rank to start from")
    ap.add_argument("--niche-rank-end", type=int, default=150,
                    help="niche-deep: library artist rank to end at")
    ap.add_argument("--niche-min-listeners", type=int, default=20_000,
                    help="niche-deep: minimum last.fm listeners per seed")
    ap.add_argument(
        "--source-cohort", default=None,
        help="friend-expand: which cohort's passers to use as seeds",
    )
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--dry-run", action="store_true",
                    help="just resolve seeds and exit (no scraping/eval)")
    args = ap.parse_args()

    if args.strategy == "friend-expand" and not args.source_cohort:
        ap.error("--strategy friend-expand requires --source-cohort")

    random.seed(args.seed)

    # DB setup.
    init_db()
    with get_connection() as conn:
        migrate_snowball_columns(conn)

    strategy = resolve_strategy(args)
    library = load_target_library(args.library_id)
    client = LastFMClient()
    scraper = ListenerScraper()

    logger.info(
        "Strategy=%s cohort_tag=%s thresholds=track≥%.0f%%/artist≥%.0f%%",
        strategy.name, strategy.cohort_tag,
        strategy.track_threshold * 100, strategy.artist_threshold * 100,
    )

    seeds = get_seeds_for_strategy(strategy, library, client, args)
    if not seeds:
        logger.error("no seeds resolved — nothing to do")
        return 1

    logger.info("Resolved %d seeds", len(seeds))
    if args.dry_run:
        for s in seeds[:50]:
            logger.info("  dry-run seed: %s (kind=%s)", s.name, s.kind)
        if len(seeds) > 50:
            logger.info("  ... and %d more", len(seeds) - 50)
        return 0

    existing = load_existing_usernames()
    logger.info("Existing users in DB: %d", len(existing))

    t0 = time.time()
    seed_results: List[SeedResult] = []
    for i, seed in enumerate(seeds, 1):
        logger.info("[%d/%d] processing seed: %s", i, len(seeds), seed.name)
        try:
            res = process_seed(
                seed, strategy, library, scraper, client, existing,
            )
            seed_results.append(res)
        except KeyboardInterrupt:
            logger.warning("Interrupted — progress is checkpointed, safe to resume")
            break
        except Exception as e:
            logger.exception("error processing seed %s: %s", seed.name, e)
            continue

    elapsed = time.time() - t0
    report(strategy, library, seed_results, elapsed, client.request_count)
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
