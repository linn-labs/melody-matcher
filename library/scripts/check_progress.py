#!/usr/bin/env python3
"""Library Method — Training Data Readiness

Historical explicit-execution research tooling; not a supported workflow.
Read docs/COMPONENTS.md and docs/DATA_AND_RIGHTS.md before considering use.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    DB_PATH,
    EMBEDDINGS_PATH,
    MIN_SUPER_GENRES,
    MAX_TOP5_CONCENTRATION,
)
from src.db import get_connection


def format_number(n):
    """Format a number with commas."""
    return f"{n:,}"


def main():
    if not DB_PATH.exists():
        print("No database found. Run the shared pipeline first.")
        return

    conn = get_connection()

    print("\n" + "=" * 55)
    print("  Library Method — Training Data Readiness")
    print("=" * 55)

    # Users passing diversity filter
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM users WHERE diversity_score IS NOT NULL"
    ).fetchone()
    total_scored = row["cnt"]

    row = conn.execute(
        """SELECT COUNT(*) as cnt FROM users
           WHERE diversity_score IS NOT NULL
             AND genre_count >= ?
             AND top5_concentration < ?""",
        (MIN_SUPER_GENRES, MAX_TOP5_CONCENTRATION),
    ).fetchone()
    passing_basic = row["cnt"]

    # Median entropy filter
    entropies = conn.execute(
        "SELECT play_entropy FROM users WHERE play_entropy IS NOT NULL ORDER BY play_entropy"
    ).fetchall()
    median_entropy = None
    eligible_users = passing_basic
    if entropies:
        median_entropy = entropies[len(entropies) // 2]["play_entropy"]
        row = conn.execute(
            """SELECT COUNT(*) as cnt FROM users
               WHERE diversity_score IS NOT NULL
                 AND genre_count >= ?
                 AND top5_concentration < ?
                 AND play_entropy >= ?""",
            (MIN_SUPER_GENRES, MAX_TOP5_CONCENTRATION, median_entropy),
        ).fetchone()
        eligible_users = row["cnt"]

    print(f"\nUsers scored:       {format_number(total_scored)}")
    print(f"Passing filter:     {format_number(eligible_users)}")

    # Track embedding coverage
    total_tracks = conn.execute("SELECT COUNT(*) as cnt FROM tracks").fetchone()["cnt"]

    # Check Deezer columns exist
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(tracks)").fetchall()
    }
    matched_tracks = 0
    if "deezer_match_status" in existing_cols:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM tracks WHERE deezer_match_status = 'matched'"
        ).fetchone()
        matched_tracks = row["cnt"]

    embedded_count = 0
    if EMBEDDINGS_PATH.exists():
        try:
            import h5py
            with h5py.File(str(EMBEDDINGS_PATH), "r") as f:
                if "track_ids" in f:
                    embedded_count = len(f["track_ids"])
        except Exception:
            pass

    print(f"\nTotal tracks:       {format_number(total_tracks)}")
    print(f"Deezer matched:     {format_number(matched_tracks)}")
    print(f"Embeddings:         {format_number(embedded_count)}")
    if matched_tracks > 0:
        pct = 100 * embedded_count / matched_tracks
        print(f"Embedding coverage: {pct:.1f}%")

    # Training data dimensions: eligible user-track pairs with embeddings
    if eligible_users > 0 and embedded_count > 0:
        # Count user-track pairs where user is eligible and track has embedding
        # This is approximate — we check Deezer-matched tracks as proxy for embedded
        if median_entropy is not None:
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM user_tracks ut
                   JOIN users u ON ut.user_id = u.id
                   JOIN tracks t ON ut.lastfm_id = t.lastfm_id
                   WHERE u.diversity_score IS NOT NULL
                     AND u.genre_count >= ?
                     AND u.top5_concentration < ?
                     AND u.play_entropy >= ?
                     AND t.deezer_match_status = 'matched'""",
                (MIN_SUPER_GENRES, MAX_TOP5_CONCENTRATION, median_entropy),
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT COUNT(*) as cnt FROM user_tracks ut
                   JOIN users u ON ut.user_id = u.id
                   JOIN tracks t ON ut.lastfm_id = t.lastfm_id
                   WHERE u.diversity_score IS NOT NULL
                     AND u.genre_count >= ?
                     AND u.top5_concentration < ?
                     AND t.deezer_match_status = 'matched'""",
                (MIN_SUPER_GENRES, MAX_TOP5_CONCENTRATION),
            ).fetchone()
        training_pairs = row["cnt"]
        print(f"\nTraining pairs:     {format_number(training_pairs)}")
        print(f"  ({format_number(eligible_users)} users × matched+embedded tracks)")

    # Readiness status
    print("\n--- Readiness ---")
    blockers = []
    if total_scored == 0:
        blockers.append("Run 03_score_diversity.py (no users scored)")
    if eligible_users == 0 and total_scored > 0:
        blockers.append("No users pass diversity filter")
    if matched_tracks == 0 or embedded_count == 0:
        blockers.append("Run 04_match_and_embed.py (no embeddings)")

    if blockers:
        print("  BLOCKED:")
        for b in blockers:
            print(f"    - {b}")
    else:
        print("  READY for training")

    print()
    conn.close()


if __name__ == "__main__":
    main()
