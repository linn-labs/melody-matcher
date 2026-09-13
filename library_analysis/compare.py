#!/usr/bin/env python3
"""
Apple Music Library vs. Dataset Coverage Analysis

Parses library.xml and checks what percentage of the library
is present in our collected Last.fm dataset, and of those,
how many have Deezer matches (and thus embeddings).

Match strategy (applied in order, first hit wins):
  1. Exact case-insensitive: lowercase(artist) + lowercase(track)
  2. Normalized: strip unicode accents, parentheticals, punctuation
  3. Artist-only normalized: catches cases where track names differ
     slightly but artist is unambiguous (reported separately)

Run from the project root:
  python library_analysis/compare.py
"""

import plistlib
import sqlite3
import unicodedata
import re
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "melody_matcher.db"
XML_PATH = Path(__file__).parent / "library.xml"


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(text):
    """Aggressive normalization for fuzzy matching.

    Lowercases, strips accents, removes parentheticals and feat. suffixes,
    strips non-alphanumeric chars, collapses whitespace.
    """
    if not text:
        return ""
    text = text.lower()
    # Normalize unicode accents: é → e
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    # Strip parenthetical / bracket suffixes
    text = re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", text)
    # Strip feat. / ft. / featuring
    text = re.sub(r"\s+(feat\.?|ft\.?|featuring)\s+.*", "", text, flags=re.IGNORECASE)
    # Strip everything that isn't a letter, digit, or space
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # Collapse whitespace
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Load dataset tracks from SQLite
# ---------------------------------------------------------------------------

def load_dataset(db_path):
    """Return two lookup dicts keyed by (artist, track) tuples.

    exact_lookup:  {(lower_artist, lower_track): deezer_match_status}
    fuzzy_lookup:  {(norm_artist,  norm_track):  deezer_match_status}
    artist_lookup: {norm_artist: list of (norm_track, status)}
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT artist_name, track_name, deezer_match_status FROM tracks"
    ).fetchall()
    conn.close()

    exact_lookup = {}
    fuzzy_lookup = {}
    artist_lookup = defaultdict(list)

    for r in rows:
        artist = r["artist_name"] or ""
        track = r["track_name"] or ""
        status = r["deezer_match_status"] or "unprocessed"

        exact_key = (artist.lower(), track.lower())
        fuzzy_key = (normalize(artist), normalize(track))

        # Keep the "best" status if there are duplicates
        # (matched > unprocessed > not_found)
        for lookup, key in [(exact_lookup, exact_key), (fuzzy_lookup, fuzzy_key)]:
            existing = lookup.get(key)
            if existing is None or status == "matched":
                lookup[key] = status

        artist_lookup[normalize(artist)].append((normalize(track), status))

    return exact_lookup, fuzzy_lookup, artist_lookup


# ---------------------------------------------------------------------------
# Parse Apple Music XML
# ---------------------------------------------------------------------------

def parse_apple_library(xml_path):
    """Return list of dicts with keys: name, artist, genre, play_count."""
    with open(xml_path, "rb") as f:
        plist = plistlib.load(f)

    tracks = []
    for track_id, info in plist.get("Tracks", {}).items():
        # Skip non-audio entries (videos, podcasts, etc.)
        kind = info.get("Kind", "")
        if "video" in kind.lower() or "podcast" in kind.lower():
            continue
        track_type = info.get("Track Type", "")
        if track_type not in ("Remote", "File", ""):
            continue

        name = info.get("Name", "").strip()
        artist = info.get("Artist", "").strip()
        genre = info.get("Genre", "Unknown").strip()
        play_count = info.get("Play Count", 0)

        if not name or not artist:
            continue

        tracks.append({
            "name": name,
            "artist": artist,
            "genre": genre,
            "play_count": play_count,
        })

    return tracks


# ---------------------------------------------------------------------------
# Match one library track against the dataset
# ---------------------------------------------------------------------------

RESULT_MATCHED_EMBEDDED   = "in_dataset_embedded"
RESULT_MATCHED_NO_DEEZER  = "in_dataset_no_deezer"
RESULT_MATCHED_UNPROCESSED = "in_dataset_unprocessed"
RESULT_NOT_IN_DATASET     = "not_in_dataset"


def split_artists(artist_str):
    """Split a multi-artist string into individual artist candidates.

    Apple Music often uses formats like:
      "Elvis Presley & The Jordanaires"
      "Dreamville, J. Cole, JID, Cozz & EARTHGANG"
      "Gryffin & ILLENIUM"
      "Chance the Rapper & Jeremih"
    Last.fm usually stores just the primary artist or uses different formatting.
    We try every individual artist extracted from the string.
    """
    candidates = [artist_str]
    # Split on common multi-artist separators
    parts = re.split(r"\s*[,&]\s*|\s+and\s+", artist_str, flags=re.IGNORECASE)
    candidates.extend(p.strip() for p in parts if p.strip())
    # Also try stripping everything after " feat." / " ft." / " with "
    primary = re.split(r"\s+(feat\.?|ft\.?|featuring|with)\s+", artist_str, flags=re.IGNORECASE)[0].strip()
    if primary != artist_str:
        candidates.append(primary)
    return candidates


def match_track(track, exact_lookup, fuzzy_lookup):
    """Return (result_code, match_method)."""
    name = track["name"]
    artist_candidates = split_artists(track["artist"])

    for artist in artist_candidates:
        # Pass 1: exact case-insensitive
        key = (artist.lower(), name.lower())
        if key in exact_lookup:
            return _status_to_result(exact_lookup[key]), "exact"

    for artist in artist_candidates:
        # Pass 2: normalized fuzzy
        key = (normalize(artist), normalize(name))
        if key in fuzzy_lookup:
            return _status_to_result(fuzzy_lookup[key]), "fuzzy"

    return RESULT_NOT_IN_DATASET, None


def _status_to_result(status):
    if status == "matched":
        return RESULT_MATCHED_EMBEDDED
    if status in ("not_found", "no_preview"):
        return RESULT_MATCHED_NO_DEEZER
    return RESULT_MATCHED_UNPROCESSED


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(library_tracks, results):
    total = len(library_tracks)
    by_result = defaultdict(list)
    for track, (result, method) in zip(library_tracks, results):
        by_result[result].append((track, method))

    embedded   = by_result[RESULT_MATCHED_EMBEDDED]
    no_deezer  = by_result[RESULT_MATCHED_NO_DEEZER]
    unprocessed = by_result[RESULT_MATCHED_UNPROCESSED]
    missing    = by_result[RESULT_NOT_IN_DATASET]

    in_dataset = len(embedded) + len(no_deezer) + len(unprocessed)

    print("\n" + "=" * 62)
    print("  Apple Music Library vs. Dataset Coverage")
    print("=" * 62)
    print(f"\n  Your library:          {total:>6,} tracks")
    print(f"\n  In dataset:            {in_dataset:>6,}  ({100*in_dataset/total:.1f}%)")
    print(f"    → will get embedding: {len(embedded):>6,}  ({100*len(embedded)/total:.1f}%)")
    print(f"    → in dataset, no Deezer match: {len(no_deezer):>4,}  ({100*len(no_deezer)/total:.1f}%)")
    print(f"    → in dataset, Deezer unprocessed: {len(unprocessed):>2,}  ({100*len(unprocessed)/total:.1f}%)")
    print(f"\n  NOT in dataset at all: {len(missing):>6,}  ({100*len(missing)/total:.1f}%)")

    # Genre breakdown for missing tracks
    if missing:
        print("\n--- Missing tracks by genre ---")
        genre_counts = defaultdict(int)
        for track, _ in missing:
            genre_counts[track["genre"]] += 1
        for genre, cnt in sorted(genre_counts.items(), key=lambda x: -x[1]):
            pct = 100 * cnt / len(missing)
            print(f"  {cnt:>5,}  {genre}  ({pct:.0f}% of missing)")

    # Genre breakdown for no-Deezer tracks
    if no_deezer:
        print("\n--- In dataset but no Deezer preview (will be skipped by NN) ---")
        genre_counts = defaultdict(int)
        for track, _ in no_deezer:
            genre_counts[track["genre"]] += 1
        for genre, cnt in sorted(genre_counts.items(), key=lambda x: -x[1]):
            print(f"  {cnt:>5,}  {genre}")

    # Play count breakdown: are high-play-count tracks covered?
    print("\n--- Coverage by how much YOU play these tracks ---")
    buckets = [
        ("50+ plays (your favorites)",   lambda t: t["play_count"] >= 50),
        ("20-49 plays",                  lambda t: 20 <= t["play_count"] < 50),
        ("5-19 plays",                   lambda t: 5 <= t["play_count"] < 20),
        ("1-4 plays",                    lambda t: 1 <= t["play_count"] < 5),
        ("0 plays (never played)",       lambda t: t["play_count"] == 0),
    ]
    for label, pred in buckets:
        bucket_tracks = [(t, r, m) for (t, (r, m)) in zip(library_tracks, results) if pred(t)]
        if not bucket_tracks:
            continue
        b_total = len(bucket_tracks)
        b_embedded = sum(1 for _, r, _ in bucket_tracks if r == RESULT_MATCHED_EMBEDDED)
        b_missing  = sum(1 for _, r, _ in bucket_tracks if r == RESULT_NOT_IN_DATASET)
        print(f"  {label}: {b_total:,} tracks | "
              f"embeddable={b_embedded} ({100*b_embedded/b_total:.0f}%) | "
              f"missing={b_missing} ({100*b_missing/b_total:.0f}%)")

    # Print the actual missing high-play-count tracks (most actionable)
    high_play_missing = [
        (t, m) for (t, (r, m)) in zip(library_tracks, results)
        if r == RESULT_NOT_IN_DATASET and t["play_count"] >= 20
    ]
    if high_play_missing:
        print(f"\n--- Your most-played tracks NOT in dataset ({len(high_play_missing)} tracks) ---")
        high_play_missing.sort(key=lambda x: -x[0]["play_count"])
        for track, _ in high_play_missing[:40]:
            print(f"  [{track['play_count']:>4}]  {track['artist']} — {track['name']}  [{track['genre']}]")
        if len(high_play_missing) > 40:
            print(f"  ... and {len(high_play_missing) - 40} more")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not DB_PATH.exists():
        print(f"DB not found at {DB_PATH}")
        sys.exit(1)
    if not XML_PATH.exists():
        print(f"XML not found at {XML_PATH}")
        sys.exit(1)

    print("Loading dataset tracks from DB...")
    exact_lookup, fuzzy_lookup, artist_lookup = load_dataset(DB_PATH)
    print(f"  Loaded {len(exact_lookup):,} unique tracks")

    print("Parsing Apple Music library...")
    library_tracks = parse_apple_library(XML_PATH)
    print(f"  Found {len(library_tracks):,} tracks in library")

    print("Matching...")
    results = [match_track(t, exact_lookup, fuzzy_lookup) for t in library_tracks]

    print_report(library_tracks, results)


if __name__ == "__main__":
    main()
