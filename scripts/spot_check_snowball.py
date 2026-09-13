#!/usr/bin/env python3
"""Spot-check: is the taste-biased snowball tractable?

Historical explicit-execution research tooling; not a supported workflow.
Read docs/COMPONENTS.md and docs/DATA_AND_RIGHTS.md before considering use.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH
from src.db import get_connection
from src.lastfm_api import LastFMClient
from src.scraper import ListenerScraper
from harness.backend.matching import normalize_artist, normalize_title

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class SeedArtist:
    artist: str
    lib_play_count: int
    lib_track_count: int
    listeners: int


@dataclass
class CandidateOverlap:
    username: str
    seed: str
    candidate_top_n: int
    track_overlap_abs: int
    artist_overlap_abs: int
    candidate_distinct_artists: int
    track_overlap_frac: float
    artist_overlap_frac: float
    passed: bool


def load_library(library_id: int) -> Tuple[Set[Tuple[str, str]], Set[str], Dict[str, Tuple[int, int]]]:
    """Return (track_set, artist_set, per_artist_stats).

    - track_set: normalized (artist, title) pairs.
    - artist_set: normalized artists.
    - per_artist_stats: normalized_artist -> (summed_play_count, distinct_tracks)
      for picking seeds.
    """
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT artist, title, play_count
            FROM harness_library_tracks
            WHERE library_id = ? AND artist IS NOT NULL AND title IS NOT NULL
            """,
            (library_id,),
        ).fetchall()
    if not rows:
        raise SystemExit(f"library {library_id} has no tracks")

    tracks: Set[Tuple[str, str]] = set()
    artists: Set[str] = set()
    stats: Dict[str, List[int]] = {}  # norm_artist -> [plays, tracks, display_name_count]
    display_for: Dict[str, Counter] = {}

    for r in rows:
        a_raw, t_raw = r["artist"], r["title"]
        a = normalize_artist(a_raw)
        t = normalize_title(t_raw)
        if not a or not t:
            continue
        tracks.add((a, t))
        artists.add(a)
        s = stats.setdefault(a, [0, 0])
        s[0] += int(r["play_count"] or 0)
        s[1] += 1
        display_for.setdefault(a, Counter())[a_raw] += 1

    logger.info(
        "Library %d: %d tracks, %d distinct artists",
        library_id, len(tracks), len(artists),
    )

    # attach display name (most common raw form) to stats
    packed: Dict[str, Tuple[str, int, int]] = {
        a: (display_for[a].most_common(1)[0][0], s[0], s[1])
        for a, s in stats.items()
    }
    return tracks, artists, packed


def pick_seeds(
    stats: Dict[str, Tuple[str, int, int]],
    client: LastFMClient,
    n_seeds: int,
    min_listeners: int,
    max_listeners: int,
    candidate_pool: int,
) -> List[SeedArtist]:
    """Pick seeds from the user's most-played artists, filtered by listener band.

    The listener-count filter is the taste-discrimination knob: mega-mainstream
    artists have listeners from every taste cluster (useless for snowballing),
    very niche artists may not produce enough listener pages to scrape. The
    default band [50k, 1M] aims at "known enough to have listener pages, niche
    enough that listeners cluster by taste."
    """
    # sort by summed play count, take the top pool
    ranked = sorted(stats.items(), key=lambda kv: -kv[1][1])[:candidate_pool]
    logger.info("Probing top %d artists for listener counts…", len(ranked))

    seeds: List[SeedArtist] = []
    for a_norm, (a_display, plays, tracks) in ranked:
        if len(seeds) >= n_seeds:
            break
        info = client.get_artist_info(a_display)
        if not info:
            logger.debug("  %s: no artist info", a_display)
            continue
        listeners = info["listeners"]
        band_ok = min_listeners <= listeners <= max_listeners
        logger.info(
            "  %-35s listeners=%-9d plays=%-5d tracks=%-3d %s",
            a_display[:35], listeners, plays, tracks,
            "SEED" if band_ok else "skip",
        )
        if band_ok:
            seeds.append(SeedArtist(
                artist=a_display, lib_play_count=plays,
                lib_track_count=tracks, listeners=listeners,
            ))

    if len(seeds) < n_seeds:
        logger.warning(
            "Only found %d seeds in listener band [%d, %d] — consider widening",
            len(seeds), min_listeners, max_listeners,
        )
    return seeds


def existing_usernames() -> Set[str]:
    with get_connection() as conn:
        rows = conn.execute("SELECT username FROM users").fetchall()
    return {r["username"].lower() for r in rows}


def scrape_seed_listeners(
    seeds: List[SeedArtist], scraper: ListenerScraper, pages: int,
) -> Dict[str, Set[str]]:
    """Return {seed_artist -> set of usernames discovered}."""
    out: Dict[str, Set[str]] = {}
    for seed in seeds:
        logger.info("Scraping %d pages of listeners for %s…", pages, seed.artist)
        users = scraper.scrape_artist_listeners(seed.artist, max_pages=pages)
        out[seed.artist] = users
    return out


def compute_overlap(
    candidate_top: List[dict],
    user_tracks: Set[Tuple[str, str]],
    user_artists: Set[str],
) -> Tuple[int, int, int]:
    """Return (track_overlap_abs, artist_overlap_abs, distinct_artists_in_top)."""
    track_hits = 0
    candidate_artists: Set[str] = set()
    for t in candidate_top:
        a = normalize_artist(t.get("artist") or "")
        ti = normalize_title(t.get("name") or "")
        if not a or not ti:
            continue
        candidate_artists.add(a)
        if (a, ti) in user_tracks:
            track_hits += 1
    artist_hits = len(candidate_artists & user_artists)
    return track_hits, artist_hits, len(candidate_artists)


def evaluate_candidates(
    seed_to_users: Dict[str, Set[str]],
    existing: Set[str],
    user_tracks: Set[Tuple[str, str]],
    user_artists: Set[str],
    client: LastFMClient,
    n_candidates: int,
    track_thresh: float,
    artist_thresh: float,
    top_tracks_limit: int,
) -> Tuple[List[CandidateOverlap], int, int, int]:
    """Sample candidates across seeds, fetch top tracks, score overlap.

    Returns (results, total_discovered, total_new, total_sampled).
    """
    # flatten with provenance, dedupe, exclude already-known users
    seen: Set[str] = set()
    pool: List[Tuple[str, str]] = []  # (username, seed)
    total_discovered = 0
    for seed, users in seed_to_users.items():
        for u in users:
            total_discovered += 1
            ul = u.lower()
            if ul in seen:
                continue
            seen.add(ul)
            if ul in existing:
                continue
            pool.append((u, seed))

    total_new = len(pool)
    logger.info(
        "Discovered %d usernames across seeds; %d unique-new (not in users table)",
        total_discovered, total_new,
    )
    if total_new == 0:
        return [], total_discovered, 0, 0

    random.shuffle(pool)
    sample = pool[:n_candidates]
    logger.info("Sampling %d for overlap evaluation…", len(sample))

    results: List[CandidateOverlap] = []
    for i, (username, seed) in enumerate(sample, 1):
        top = client.get_user_top_tracks(username, period="overall",
                                         page=1, limit=top_tracks_limit)
        if not top:
            logger.debug("  [%d/%d] %s: no top tracks", i, len(sample), username)
            continue
        n = len(top)
        track_hits, artist_hits, cand_artists = compute_overlap(
            top, user_tracks, user_artists,
        )
        t_frac = track_hits / n if n else 0.0
        a_frac = artist_hits / cand_artists if cand_artists else 0.0
        passed = (t_frac >= track_thresh) or (a_frac >= artist_thresh)
        results.append(CandidateOverlap(
            username=username, seed=seed, candidate_top_n=n,
            track_overlap_abs=track_hits, artist_overlap_abs=artist_hits,
            candidate_distinct_artists=cand_artists,
            track_overlap_frac=t_frac, artist_overlap_frac=a_frac,
            passed=passed,
        ))
        if i % 10 == 0:
            logger.info("  evaluated %d/%d", i, len(sample))
    return results, total_discovered, total_new, len(sample)


def report(
    seeds: List[SeedArtist],
    seed_to_users: Dict[str, Set[str]],
    existing: Set[str],
    results: List[CandidateOverlap],
    total_discovered: int,
    total_new: int,
    total_sampled: int,
    track_thresh: float,
    artist_thresh: float,
) -> None:
    print()
    print("=" * 78)
    print(" SPOT-CHECK REPORT")
    print("=" * 78)

    print("\n[seeds]")
    for s in seeds:
        users = seed_to_users.get(s.artist, set())
        new = sum(1 for u in users if u.lower() not in existing)
        print(f"  {s.artist:<35} listeners={s.listeners:<9d} scraped={len(users):<4d} new={new}")

    print("\n[yield]")
    print(f"  discovered (with dupes): {total_discovered}")
    print(f"  unique new usernames:    {total_new}")
    print(f"  sampled for evaluation:  {total_sampled}")
    print(f"  evaluated (had tracks):  {len(results)}")

    if not results:
        print("\nno candidates had fetchable top tracks — nothing to score.")
        return

    passed = [r for r in results if r.passed]
    t_fracs = sorted((r.track_overlap_frac for r in results), reverse=True)
    a_fracs = sorted((r.artist_overlap_frac for r in results), reverse=True)

    def pct(xs, q):
        if not xs:
            return 0.0
        i = min(len(xs) - 1, int(q * len(xs)))
        return xs[i]

    print("\n[overlap distribution]")
    print(f"  track overlap  — max={t_fracs[0]:.1%}  p10={pct(t_fracs, 0.10):.1%}"
          f"  p25={pct(t_fracs, 0.25):.1%}  median={pct(t_fracs, 0.50):.1%}")
    print(f"  artist overlap — max={a_fracs[0]:.1%}  p10={pct(a_fracs, 0.10):.1%}"
          f"  p25={pct(a_fracs, 0.25):.1%}  median={pct(a_fracs, 0.50):.1%}")

    print(f"\n[threshold: track>={track_thresh:.0%} OR artist>={artist_thresh:.0%}]")
    print(f"  passed: {len(passed)}/{len(results)}  ({len(passed)/len(results):.1%})")

    # Per-seed pass-rate breakdown — tells us which slices of the user's taste
    # actually attract overlapping listeners vs. which are dead ends.
    by_seed: Dict[str, List[CandidateOverlap]] = {}
    for r in results:
        by_seed.setdefault(r.seed, []).append(r)

    def _median(xs: List[float]) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        return s[len(s) // 2]

    seed_rows = []
    for s in seeds:
        rs = by_seed.get(s.artist, [])
        n_eval = len(rs)
        n_pass = sum(1 for r in rs if r.passed)
        rate = (n_pass / n_eval) if n_eval else 0.0
        med_t = _median([r.track_overlap_frac for r in rs])
        med_a = _median([r.artist_overlap_frac for r in rs])
        seed_rows.append((s.artist, s.listeners, n_eval, n_pass, rate, med_t, med_a))
    seed_rows.sort(key=lambda x: (-x[4], -x[3]))  # by pass rate, then absolute passes

    print("\n[per-seed evaluation]")
    print(f"  {'seed':<32} {'lstnrs':>9} {'eval':>5} {'pass':>5} {'rate':>7}"
          f" {'med-t%':>7} {'med-a%':>7}")
    for artist, lstn, n_eval, n_pass, rate, med_t, med_a in seed_rows:
        print(f"  {artist[:32]:<32} {lstn:>9d} {n_eval:>5d} {n_pass:>5d}"
              f" {rate:>6.1%} {med_t:>6.1%} {med_a:>6.1%}")

    if passed:
        print("\n[top passing candidates]")
        passed_sorted = sorted(
            passed, key=lambda r: -(r.track_overlap_frac + r.artist_overlap_frac),
        )[:15]
        print(f"  {'username':<25} {'seed':<30} {'top':<5} {'t%':>6} {'a%':>6}")
        for r in passed_sorted:
            print(f"  {r.username[:25]:<25} {r.seed[:30]:<30} "
                  f"{r.candidate_top_n:<5d} "
                  f"{r.track_overlap_frac:>5.1%} {r.artist_overlap_frac:>5.1%}")

    print("\n[near-miss candidates]")  # context for whether thresholds are tuned
    near = [r for r in results if not r.passed]
    near.sort(key=lambda r: -(r.track_overlap_frac + r.artist_overlap_frac))
    for r in near[:5]:
        print(f"  {r.username[:25]:<25} {r.seed[:30]:<30} "
              f"{r.candidate_top_n:<5d} "
              f"{r.track_overlap_frac:>5.1%} {r.artist_overlap_frac:>5.1%}")

    # --- verdict -----------------------------------------------------------
    pass_rate = len(passed) / len(results) if results else 0
    yield_per_seed = total_new / max(len(seeds), 1)
    print("\n[verdict]")
    print(f"  new users per seed:       {yield_per_seed:.1f}")
    print(f"  pass rate among sampled:  {pass_rate:.1%}")

    # crude heuristic — meant to be a conversation starter, not an oracle
    if pass_rate >= 0.15 and yield_per_seed >= 20:
        print("  → looks tractable. Scaling this up should yield a taste-biased")
        print("    cohort. Full Phase 1 is worth building.")
    elif pass_rate >= 0.05:
        print("  → marginal. Consider widening scrape (more pages, more seeds)")
        print("    or loosening thresholds before scaling up.")
    else:
        print("  → weak signal. Taste-biased snowball via listener pages may not")
        print("    concentrate enough. Revisit seed selection or strategy.")
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--library-id", type=int, default=1)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed-pool", type=int, default=40,
                    help="top-N most-played artists to probe for listener counts")
    ap.add_argument("--min-listeners", type=int, default=50_000)
    ap.add_argument("--max-listeners", type=int, default=1_000_000)
    ap.add_argument("--scrape-pages", type=int, default=3)
    ap.add_argument("--n-candidates", type=int, default=50)
    ap.add_argument("--top-tracks-limit", type=int, default=200)
    ap.add_argument("--track-overlap-threshold", type=float, default=0.15)
    ap.add_argument("--artist-overlap-threshold", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    t0 = time.time()

    user_tracks, user_artists, stats = load_library(args.library_id)
    client = LastFMClient()
    scraper = ListenerScraper()

    seeds = pick_seeds(
        stats, client,
        n_seeds=args.n_seeds,
        min_listeners=args.min_listeners,
        max_listeners=args.max_listeners,
        candidate_pool=args.seed_pool,
    )
    if not seeds:
        logger.error("no seeds selected — widen --min/--max-listeners")
        return 1

    existing = existing_usernames()
    logger.info("Existing users table: %d usernames", len(existing))

    seed_to_users = scrape_seed_listeners(seeds, scraper, pages=args.scrape_pages)

    results, total_discovered, total_new, total_sampled = evaluate_candidates(
        seed_to_users, existing, user_tracks, user_artists, client,
        n_candidates=args.n_candidates,
        track_thresh=args.track_overlap_threshold,
        artist_thresh=args.artist_overlap_threshold,
        top_tracks_limit=args.top_tracks_limit,
    )

    report(
        seeds, seed_to_users, existing, results,
        total_discovered, total_new, total_sampled,
        args.track_overlap_threshold, args.artist_overlap_threshold,
    )

    logger.info("Done in %.1fs (API calls: %d)", time.time() - t0, client.request_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
