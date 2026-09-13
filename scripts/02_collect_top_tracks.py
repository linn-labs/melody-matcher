#!/usr/bin/env python3
"""Top Tracks Collection Script

Historical explicit-execution research tooling; not a supported workflow.
Read docs/COMPONENTS.md and docs/DATA_AND_RIGHTS.md before considering use.
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MIN_UNIQUE_TRACKS,
    MAX_TOP_TRACKS_PAGES,
    TOP_TRACKS_PER_PAGE,
)
from src.db import (
    init_db,
    get_connection,
    transaction,
    get_users_by_status,
    get_user_count_by_status,
    set_user_status,
    upsert_tracks_batch,
    add_user_tracks_batch,
)
from src.lastfm_api import LastFMClient

# Logging
LOG_DIR = Path(__file__).parent.parent / "data"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "02_collect_top_tracks.log"),
    ],
)
logger = logging.getLogger(__name__)


def collect_user_tracks(api, username):
    """Fetch all top tracks for a user across paginated results.

    Returns:
        list of (lastfm_id, track_name, artist_name, play_count) or None on failure.
    """
    all_tracks = []
    seen_ids = set()

    for page in range(1, MAX_TOP_TRACKS_PAGES + 1):
        tracks = api.get_user_top_tracks(
            username, period="overall", page=page, limit=TOP_TRACKS_PER_PAGE
        )

        if tracks is None:
            if page == 1:
                return None  # First page failed — user inaccessible
            break  # Later pages failing = we got what we can

        if not tracks:
            break  # Empty page = no more tracks

        for t in tracks:
            artist = t["artist"]
            name = t["name"]
            lastfm_id = f"{artist}::{name}"

            if lastfm_id not in seen_ids:
                seen_ids.add(lastfm_id)
                all_tracks.append((lastfm_id, name, artist, t["playcount"]))

        # If we got fewer than the limit, no more pages
        if len(tracks) < TOP_TRACKS_PER_PAGE:
            break

    return all_tracks


def main():
    logger.info("Starting listening history collection")

    init_db()
    conn = get_connection()
    api = LastFMClient()

    try:
        # Get users to process: validated + errored (for retry)
        validated = get_users_by_status(conn, "validated")
        errored = get_users_by_status(conn, "error")
        users = list(validated) + list(errored)

        if not users:
            logger.info("No users to collect. Run 01_discover_users.py first.")
            return

        total = len(users)
        logger.info("Collecting histories for %d users", total)

        collected = 0
        rejected = 0
        errors = 0
        start_time = time.time()

        for i, user in enumerate(users, 1):
            username = user["username"]
            user_id = user["id"]

            # Mark as collecting
            with transaction(conn):
                set_user_status(conn, username, "collecting")

            try:
                tracks = collect_user_tracks(api, username)

                if tracks is None:
                    with transaction(conn):
                        set_user_status(
                            conn, username, "error",
                            error_message="Failed to fetch top tracks",
                        )
                    errors += 1
                    continue

                unique_count = len(tracks)

                if unique_count < MIN_UNIQUE_TRACKS:
                    with transaction(conn):
                        set_user_status(conn, username, "rejected")
                    rejected += 1
                    continue

                # Insert all tracks and user_tracks in a single transaction
                with transaction(conn):
                    track_rows = [(t[0], t[1], t[2]) for t in tracks]
                    upsert_tracks_batch(conn, track_rows)

                    user_track_rows = [(user_id, t[0], t[3]) for t in tracks]
                    add_user_tracks_batch(conn, user_track_rows)

                    set_user_status(conn, username, "collected")

                collected += 1

            except Exception as e:
                logger.error("Error collecting %s: %s", username, e)
                with transaction(conn):
                    set_user_status(
                        conn, username, "error",
                        error_message=str(e)[:200],
                    )
                errors += 1

            # Progress logging
            if i % 100 == 0 or i == total:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total - i) / rate if rate > 0 else 0
                logger.info(
                    "[%d/%d] collected=%d rejected=%d errors=%d "
                    "| %.1f users/min | ETA: %.0f min",
                    i, total, collected, rejected, errors,
                    rate * 60, remaining / 60,
                )

        # Final stats
        elapsed = time.time() - start_time
        logger.info("=== Collection complete ===")
        logger.info(
            "Processed %d users in %.1f min: collected=%d rejected=%d errors=%d",
            total, elapsed / 60, collected, rejected, errors,
        )
        logger.info("Total API requests: %d", api.request_count)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
