"""Fuzzy-match Apple-Music-exported tracks against the MERT embeddings catalog.

The embeddings HDF5 stores track_ids of the form `"artist::title"` (produced by
scripts/02_collect_top_tracks.py). We:
  1. Normalize both sides (fold unicode, lower, strip variant parentheticals).
  2. Bucket embedding track_ids by normalized artist (+ fuzzy artist fallback).
  3. For each imported track, score-rank titles within the matching artist bucket
     using rapidfuzz WRatio. Accept above a threshold.

The normalized artist index is cached to disk keyed on embeddings.h5 mtime so
subsequent imports reuse it.
"""

from __future__ import annotations

import logging
import pickle
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
from rapidfuzz import fuzz, process, utils as rf_utils

from config import DATA_DIR, EMBEDDINGS_PATH

from .library_import import ImportedTrack

logger = logging.getLogger(__name__)

MATCH_THRESHOLD = 85         # rapidfuzz WRatio score 0..100
ARTIST_FALLBACK_THRESHOLD = 90
INDEX_CACHE_PATH = Path(DATA_DIR) / "harness_match_index.pkl"

# Regex patterns to strip in "aggressive" policy. Order matters:
# feat/ft first (so "(feat. X) - Remastered" works), then bracketed noise.
_FEAT_RE = re.compile(
    r"""
    \s*
    [\(\[\-]?\s*                     # opening ( [ -
    (?:feat(?:uring)?|ft|with)\.?\s+
    [^)\]\-]*?                        # feat body
    [\)\]]?                           # optional close
    \s*$                              # allow trailing if not followed by more
    """,
    re.IGNORECASE | re.VERBOSE,
)

_VARIANT_RE = re.compile(
    r"""
    \s*
    [\(\[]                            # open paren/bracket
    [^)\]]*?                           # anything inside
    (?:remaster(?:ed)?|deluxe|edition|expanded|anniversary|
       live|acoustic|unplugged|
       radio\s*edit|single\s*version|album\s*version|
       mono|stereo|
       extended(?:\s*mix)?|club\s*mix|original\s*mix|
       bonus\s*track|demo|remix|version)
    [^)\]]*
    [\)\]]
    """,
    re.IGNORECASE | re.VERBOSE,
)

_DASH_VARIANT_RE = re.compile(
    r"""
    \s+-\s+
    (?:remaster(?:ed)?.*|deluxe.*|live.*|acoustic.*|unplugged.*|
       radio\s*edit.*|single\s*version.*|album\s*version.*|
       mono.*|stereo.*|bonus.*|demo.*|remix.*|version.*|
       from\s+.+|
       \d{4}\s*remaster.*)
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)

_MULTISPACE_RE = re.compile(r"\s+")


def normalize_title(s: str) -> str:
    """Strip variant noise and fold to a canonical form for matching."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.strip()
    # strip variant noise — apply until stable
    prev = None
    while prev != s:
        prev = s
        s = _VARIANT_RE.sub("", s)
        s = _DASH_VARIANT_RE.sub("", s)
        s = _FEAT_RE.sub("", s)
    s = s.lower()
    s = _MULTISPACE_RE.sub(" ", s).strip()
    return s


def normalize_artist(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.strip().lower()
    # Collapse "and"/ampersand for consistent bucketing across "X and Y" vs "X & Y"
    s = re.sub(r"\s*&\s*", " and ", s)
    s = _MULTISPACE_RE.sub(" ", s).strip()
    return s


# ----------------------------------------------------------------------------
# Index over the embedding catalog
# ----------------------------------------------------------------------------

@dataclass
class _IndexEntry:
    idx: int
    track_id: str          # original "artist::title"
    title_norm: str


def _build_index() -> Tuple[Dict[str, List[_IndexEntry]], List[str]]:
    """Read embeddings.h5 track_ids once, bucket by normalized artist."""
    with h5py.File(str(EMBEDDINGS_PATH), "r") as f:
        track_ids = f["track_ids"].asstr()[:]
    buckets: Dict[str, List[_IndexEntry]] = {}
    for i, tid in enumerate(track_ids):
        if not tid or "::" not in tid:
            continue
        artist_raw, title_raw = tid.split("::", 1)
        a_norm = normalize_artist(artist_raw)
        t_norm = normalize_title(title_raw)
        buckets.setdefault(a_norm, []).append(_IndexEntry(i, tid, t_norm))
    artists = list(buckets.keys())
    logger.info("Built match index: %d tracks across %d artists", len(track_ids), len(artists))
    return buckets, artists


def _load_or_build_index() -> Tuple[Dict[str, List[_IndexEntry]], List[str]]:
    emb_path = Path(EMBEDDINGS_PATH)
    if not emb_path.exists():
        raise FileNotFoundError(f"embeddings.h5 not found at {emb_path}")
    emb_mtime = emb_path.stat().st_mtime_ns
    if INDEX_CACHE_PATH.exists():
        try:
            with open(INDEX_CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if cached.get("emb_mtime") == emb_mtime:
                logger.info("Loaded match index from cache")
                return cached["buckets"], cached["artists"]
        except Exception as exc:
            logger.warning("cache unreadable, rebuilding: %s", exc)
    t0 = time.time()
    buckets, artists = _build_index()
    logger.info("Match index built in %.1fs", time.time() - t0)
    try:
        INDEX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INDEX_CACHE_PATH, "wb") as f:
            pickle.dump({"emb_mtime": emb_mtime, "buckets": buckets, "artists": artists}, f)
    except Exception as exc:
        logger.warning("could not write match cache: %s", exc)
    return buckets, artists


_INDEX: Optional[Tuple[Dict[str, List[_IndexEntry]], List[str]]] = None


def get_index() -> Tuple[Dict[str, List[_IndexEntry]], List[str]]:
    global _INDEX
    if _INDEX is None:
        _INDEX = _load_or_build_index()
    return _INDEX


# ----------------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------------

@dataclass
class MatchResult:
    matched: bool
    embedding_index: Optional[int] = None
    embedding_key: Optional[str] = None
    match_score: Optional[float] = None
    matched_artist: Optional[str] = None
    matched_title: Optional[str] = None


def _candidate_buckets(artist_norm: str, artists: List[str], buckets) -> Iterable[_IndexEntry]:
    """Yield entries for likely-matching artists: exact first, then one fuzzy fallback."""
    exact = buckets.get(artist_norm)
    if exact:
        for e in exact:
            yield e
        return
    # fuzzy artist fallback — useful for "Sigur Rós" vs "Sigur Ros" after NFKD
    best = process.extractOne(
        artist_norm, artists,
        scorer=fuzz.WRatio, score_cutoff=ARTIST_FALLBACK_THRESHOLD,
    )
    if best:
        for e in buckets.get(best[0], []):
            yield e


def match_track(track: ImportedTrack) -> MatchResult:
    buckets, artists = get_index()
    a_norm = normalize_artist(track.artist)
    t_norm = normalize_title(track.title)
    if not a_norm or not t_norm:
        return MatchResult(matched=False)

    candidates = list(_candidate_buckets(a_norm, artists, buckets))
    if not candidates:
        return MatchResult(matched=False)

    choices = [c.title_norm for c in candidates]
    best = process.extractOne(t_norm, choices, scorer=fuzz.WRatio, score_cutoff=MATCH_THRESHOLD)
    if best is None:
        return MatchResult(matched=False)

    _, score, idx = best
    entry = candidates[idx]
    artist_part, _, title_part = entry.track_id.partition("::")
    return MatchResult(
        matched=True,
        embedding_index=entry.idx,
        embedding_key=entry.track_id,
        match_score=float(score),
        matched_artist=artist_part,
        matched_title=title_part,
    )


def match_many(tracks: List[ImportedTrack]) -> List[Tuple[ImportedTrack, MatchResult]]:
    get_index()  # warm up before timing starts mattering
    t0 = time.time()
    out = [(t, match_track(t)) for t in tracks]
    hits = sum(1 for _, r in out if r.matched)
    logger.info("Matched %d/%d tracks in %.1fs", hits, len(tracks), time.time() - t0)
    return out
