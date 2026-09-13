#!/usr/bin/env python3
"""Progress Checker

Historical explicit-execution research tooling; not a supported workflow.
Read docs/COMPONENTS.md and docs/DATA_AND_RIGHTS.md before considering use.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DB_PATH, MIN_SUPER_GENRES, MAX_TOP5_CONCENTRATION
from src.db import get_connection, get_stats, get_scrobble_stats


def format_number(n):
    """Format a number with commas."""
    return f"{n:,}"


def main():
    if not DB_PATH.exists():
        print("No database found. Run 01_discover_users.py first.")
        return

    conn = get_connection()
    stats = get_stats(conn)

    print("\n" + "=" * 55)
    print("  Melody Matcher — Collection Progress")
    print("=" * 55)

    # Users by status
    print("\nUsers by status:")
    status_order = [
        "candidate", "validated", "collecting", "collected", "rejected", "error"
    ]
    by_status = stats["users_by_status"]
    for status in status_order:
        count = by_status.get(status, 0)
        if count > 0:
            bar = "#" * min(count // 100, 40)
            print(f"  {status:12s}  {format_number(count):>8s}  {bar}")

    print(f"\n  {'Total':12s}  {format_number(stats['total_users']):>8s}")

    # Tracks
    print(f"\nUnique tracks:      {format_number(stats['total_tracks'])}")
    print(f"User-track records: {format_number(stats['total_user_tracks'])}")
    print(f"Artists scraped:    {format_number(stats['artists_scraped'])}")
    print(f"Artist tags cached: {format_number(stats['artist_tags_cached'])}")

    # Average tracks per collected user
    collected = by_status.get("collected", 0)
    if collected > 0 and stats["total_user_tracks"] > 0:
        avg_tracks = stats["total_user_tracks"] / collected
        print(f"Avg tracks/user:    {avg_tracks:.0f}")

    # Diversity stats
    if stats["users_scored"] > 0:
        print(f"\n--- Diversity Metrics ({format_number(stats['users_scored'])} scored) ---")
        print(f"  Avg diversity score:    {stats['avg_diversity_score']:.3f}")
        print(f"  Avg genre count:        {stats['avg_genre_count']:.1f}")
        print(f"  Avg play entropy:       {stats['avg_play_entropy']:.2f}")
        print(f"  Avg top-5 concentration: {stats['avg_top5_concentration']:.2f}")

        # Count users passing filter
        row = conn.execute(
            """
            SELECT COUNT(*) as cnt FROM users
            WHERE diversity_score IS NOT NULL
              AND genre_count >= ?
              AND top5_concentration < ?
            """,
            (MIN_SUPER_GENRES, MAX_TOP5_CONCENTRATION),
        ).fetchone()
        passing = row["cnt"]

        # Get median entropy for the full filter
        entropies = conn.execute(
            "SELECT play_entropy FROM users WHERE play_entropy IS NOT NULL ORDER BY play_entropy"
        ).fetchall()
        if entropies:
            median_entropy = entropies[len(entropies) // 2]["play_entropy"]
            row = conn.execute(
                """
                SELECT COUNT(*) as cnt FROM users
                WHERE diversity_score IS NOT NULL
                  AND genre_count >= ?
                  AND top5_concentration < ?
                  AND play_entropy >= ?
                """,
                (MIN_SUPER_GENRES, MAX_TOP5_CONCENTRATION, median_entropy),
            ).fetchone()
            full_passing = row["cnt"]
            print(f"\n  Passing full filter:    {format_number(full_passing)} / {format_number(stats['users_scored'])}")
        else:
            print(f"\n  Passing genre+conc:     {format_number(passing)} / {format_number(stats['users_scored'])}")

    # Phase estimation
    candidates = by_status.get("candidate", 0)
    validated = by_status.get("validated", 0)

    if candidates > 0:
        est_min = candidates / (4.5 * 60)  # 4.5 req/sec
        print(f"\nEstimated validation time: ~{est_min:.0f} min ({candidates} candidates)")

    if validated > 0:
        est_min = (validated * 5) / (4.5 * 60)  # ~5 pages per user
        print(f"Estimated collection time: ~{est_min:.0f} min ({validated} users)")

    # Uncached artists
    unique_artists = conn.execute(
        "SELECT COUNT(DISTINCT artist_name) as cnt FROM tracks"
    ).fetchone()["cnt"]
    if unique_artists > 0:
        cached = stats["artist_tags_cached"]
        uncached = unique_artists - cached
        if uncached > 0:
            est_min = (uncached * 2) / (4.5 * 60)  # 2 calls per artist
            print(f"Estimated tag fetch time: ~{est_min:.0f} min ({uncached} uncached artists)")

    # Deezer matching stats
    deezer_stats = stats.get("deezer_by_status", {})
    if deezer_stats:
        print("\n--- Deezer Matching ---")
        deezer_order = ["matched", "not_found", "no_preview", "error", "unmatched"]
        for status in deezer_order:
            count = deezer_stats.get(status, 0)
            if count > 0:
                print(f"  {status:12s}  {format_number(count):>10s}")
        total_deezer = sum(deezer_stats.values())
        matched = deezer_stats.get("matched", 0)
        attempted = total_deezer - deezer_stats.get("unmatched", 0)
        if attempted > 0:
            print(f"\n  Match rate: {100 * matched / attempted:.1f}% (of {format_number(attempted)} attempted)")

    # Scrobble collection stats
    scrobble_stats = get_scrobble_stats(conn)
    scrobble_by_status = scrobble_stats.get("scrobble_users_by_status", {})
    if scrobble_by_status:
        print("\n--- Scrobble Collection ---")
        scrobble_order = ["completed", "in_progress", "pending", "error"]
        for status in scrobble_order:
            count = scrobble_by_status.get(status, 0)
            if count > 0:
                print(f"  {status:12s}  {format_number(count):>10s}")
        total_scrobbles = scrobble_stats.get("total_scrobbles", 0)
        print(f"  Scrobble records: {format_number(total_scrobbles)}")

    # Embedding stats
    emb_count = stats.get("embeddings_computed", 0)
    emb_size = stats.get("embeddings_file_size_mb", 0)
    matched_count = deezer_stats.get("matched", 0)
    if emb_count > 0 or matched_count > 0:
        print("\n--- Audio Embeddings ---")
        if matched_count > 0:
            print(f"  Computed: {format_number(emb_count)} / {format_number(matched_count)}"
                  f" ({100 * emb_count / matched_count:.1f}%)" if matched_count > 0 else "")
        else:
            print(f"  Computed: {format_number(emb_count)}")
        if emb_size > 0:
            print(f"  File size: {emb_size:.1f} MB")

    print()
    conn.close()


if __name__ == "__main__":
    main()
