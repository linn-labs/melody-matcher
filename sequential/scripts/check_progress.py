#!/usr/bin/env python3
"""Sequential Method — Scrobble Collection Progress

Historical explicit-execution research tooling; not a supported workflow.
Read docs/COMPONENTS.md and docs/DATA_AND_RIGHTS.md before considering use.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import DB_PATH
from src.db import get_connection, get_scrobble_stats


def format_number(n):
    """Format a number with commas."""
    return f"{n:,}"


def main():
    if not DB_PATH.exists():
        print("No database found. Run the shared pipeline first.")
        return

    conn = get_connection()

    # Check if scrobble tables exist
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}

    if "scrobble_progress" not in tables:
        print("Scrobble tables not yet created. Run 01_collect_scrobbles.py first.")
        conn.close()
        return

    print("\n" + "=" * 55)
    print("  Sequential Method — Scrobble Collection Progress")
    print("=" * 55)

    stats = get_scrobble_stats(conn)

    # Users by scrobble status
    by_status = stats.get("scrobble_users_by_status", {})
    if by_status:
        print("\nUsers by scrobble status:")
        status_order = ["pending", "in_progress", "completed", "error"]
        for status in status_order:
            count = by_status.get(status, 0)
            if count > 0:
                print(f"  {status:12s}  {format_number(count):>8s}")
        total_tracked = sum(by_status.values())
        print(f"  {'Total':12s}  {format_number(total_tracked):>8s}")

    # Total scrobbles
    total_scrobbles = stats.get("total_scrobbles", 0)
    print(f"\nScrobble records:   {format_number(total_scrobbles)}")

    # Avg scrobbles per completed user
    completed = by_status.get("completed", 0)
    if completed > 0 and total_scrobbles > 0:
        avg = total_scrobbles / completed
        print(f"Avg per user:       {format_number(int(avg))}")

    # New tracks discovered (tracks in scrobbles but not in user_tracks)
    if "scrobbles" in tables:
        row = conn.execute("""
            SELECT COUNT(DISTINCT s.lastfm_id) as cnt
            FROM scrobbles s
            LEFT JOIN user_tracks ut ON s.lastfm_id = ut.lastfm_id
            WHERE ut.lastfm_id IS NULL
        """).fetchone()
        new_tracks = row["cnt"]
        if new_tracks > 0:
            print(f"New tracks found:   {format_number(new_tracks)} (not in top-tracks)")

    # Time estimate for remaining
    pending = by_status.get("pending", 0) + by_status.get("in_progress", 0)
    if pending > 0 and completed > 0:
        # Estimate based on completed users' page counts
        row = conn.execute("""
            SELECT AVG(pages_fetched) as avg_pages
            FROM scrobble_progress WHERE status = 'completed'
        """).fetchone()
        avg_pages = row["avg_pages"] or 100
        est_calls = pending * avg_pages
        est_hours = est_calls / (4.5 * 3600)
        print(f"\nRemaining:          {format_number(pending)} users")
        print(f"Est. API calls:     {format_number(int(est_calls))}")
        print(f"Est. time:          ~{est_hours:.1f} hours")
    elif pending > 0:
        # No completed users yet — rough estimate
        est_calls = pending * 100  # assume ~100 pages avg
        est_hours = est_calls / (4.5 * 3600)
        print(f"\nRemaining:          {format_number(pending)} users")
        print(f"Est. time:          ~{est_hours:.1f} hours (rough)")

    print()
    conn.close()


if __name__ == "__main__":
    main()
