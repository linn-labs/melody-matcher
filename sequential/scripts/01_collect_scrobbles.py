#!/usr/bin/env python3
"""Scrobble Collection Script

Historical explicit-execution research tooling; not a supported workflow.
Read docs/COMPONENTS.md and docs/DATA_AND_RIGHTS.md before considering use.
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import SCROBBLES_PER_PAGE, MAX_SCROBBLE_PAGES, SCROBBLE_BATCH_COMMIT
from src.db import (
    init_db,
    get_connection,
    transaction,
    get_users_by_status,
    migrate_scrobble_tables,
    upsert_scrobble_progress,
    get_users_needing_scrobbles,
    add_scrobbles_batch,
    upsert_tracks_batch,
)
from src.lastfm_api import LastFMClient

# Logging
LOG_DIR = Path(__file__).parent.parent.parent / "data"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "01_collect_scrobbles.log"),
    ],
)
logger = logging.getLogger(__name__)


def init_scrobble_progress(conn):
    """Create scrobble_progress entries for all collected users not yet tracked."""
    existing = {r[0] for r in conn.execute(
        "SELECT user_id FROM scrobble_progress"
    ).fetchall()}

    collected_users = get_users_by_status(conn, "collected")
    new_count = 0

    for user in collected_users:
        if user["id"] not in existing:
            upsert_scrobble_progress(conn, user["id"], status="pending")
            new_count += 1

    if new_count > 0:
        conn.commit()
        logger.info("Initialized scrobble_progress for %d new users", new_count)

    return new_count


def collect_user_scrobbles(api, conn, user_id, username, resume_page, total_pages_known):
    """Fetch scrobble history for a user, page by page.

    Returns (pages_fetched, scrobbles_stored, new_tracks) or raises on fatal error.
    """
    pages_fetched = resume_page
    scrobbles_stored = 0
    new_tracks_found = 0
    scrobble_buffer = []
    track_buffer = []
    track_ids_seen = set()

    # Determine starting page
    start_page = resume_page + 1

    # If we don't know total pages yet, fetch page 1 to find out
    if total_pages_known is None or total_pages_known == 0:
        tracks, total_pages = api.get_user_recent_tracks(
            username, page=1, limit=SCROBBLES_PER_PAGE
        )
        if tracks is None:
            return None, None, None  # User inaccessible

        total_pages_known = total_pages

        # Update progress with total info
        with transaction(conn):
            upsert_scrobble_progress(
                conn, user_id,
                total_pages=total_pages_known,
            )

        # Process page 1 results
        for t in tracks:
            lastfm_id = f"{t['artist']}::{t['name']}"
            scrobble_buffer.append((user_id, lastfm_id, t["listened_at"]))
            if lastfm_id not in track_ids_seen:
                track_ids_seen.add(lastfm_id)
                track_buffer.append((lastfm_id, t["name"], t["artist"]))

        pages_fetched = 1
        start_page = 2

    # Apply page limit if configured
    max_page = total_pages_known
    if MAX_SCROBBLE_PAGES > 0:
        max_page = min(total_pages_known, MAX_SCROBBLE_PAGES)

    for page in range(start_page, max_page + 1):
        tracks, _ = api.get_user_recent_tracks(
            username, page=page, limit=SCROBBLES_PER_PAGE
        )

        if tracks is None:
            # API error on this page — commit what we have and stop
            logger.warning("Failed to fetch page %d/%d for %s", page, max_page, username)
            break

        if not tracks:
            break  # Empty page = no more data

        for t in tracks:
            lastfm_id = f"{t['artist']}::{t['name']}"
            scrobble_buffer.append((user_id, lastfm_id, t["listened_at"]))
            if lastfm_id not in track_ids_seen:
                track_ids_seen.add(lastfm_id)
                track_buffer.append((lastfm_id, t["name"], t["artist"]))

        pages_fetched = page

        # Batch commit when buffer is large enough
        if len(scrobble_buffer) >= SCROBBLE_BATCH_COMMIT:
            with transaction(conn):
                if track_buffer:
                    upsert_tracks_batch(conn, track_buffer)
                    new_tracks_found += len(track_buffer)
                    track_buffer = []
                add_scrobbles_batch(conn, scrobble_buffer)
                scrobbles_stored += len(scrobble_buffer)
                scrobble_buffer = []
                upsert_scrobble_progress(
                    conn, user_id,
                    pages_fetched=pages_fetched,
                    scrobbles_stored=scrobbles_stored,
                )

    # Final commit for remaining buffer
    if scrobble_buffer or track_buffer:
        with transaction(conn):
            if track_buffer:
                upsert_tracks_batch(conn, track_buffer)
                new_tracks_found += len(track_buffer)
            add_scrobbles_batch(conn, scrobble_buffer)
            scrobbles_stored += len(scrobble_buffer)
            upsert_scrobble_progress(
                conn, user_id,
                pages_fetched=pages_fetched,
                scrobbles_stored=scrobbles_stored,
            )

    return pages_fetched, scrobbles_stored, new_tracks_found


def main():
    logger.info("Starting scrobble collection")

    init_db()
    conn = get_connection()
    migrate_scrobble_tables(conn)
    api = LastFMClient()

    try:
        # Initialize progress for any new collected users
        init_scrobble_progress(conn)

        # Get users needing scrobble collection
        users = get_users_needing_scrobbles(conn)
        if not users:
            logger.info("No users need scrobble collection.")
            return

        total = len(users)
        logger.info("Collecting scrobbles for %d users", total)

        completed = 0
        errors = 0
        total_scrobbles = 0
        total_pages_done = 0
        start_time = time.time()

        for i, user in enumerate(users, 1):
            user_id = user["id"]
            username = user["username"]
            resume_page = user["pages_fetched"] or 0
            known_pages = user["total_pages"]

            # Mark as in_progress
            with transaction(conn):
                upsert_scrobble_progress(conn, user_id, status="in_progress")

            try:
                pages, scrobbles, new_tracks = collect_user_scrobbles(
                    api, conn, user_id, username, resume_page, known_pages
                )

                if pages is None:
                    # User inaccessible
                    with transaction(conn):
                        upsert_scrobble_progress(
                            conn, user_id,
                            status="error",
                            error_message="Failed to fetch recent tracks",
                        )
                    errors += 1
                    continue

                # Mark completed
                with transaction(conn):
                    upsert_scrobble_progress(
                        conn, user_id,
                        status="completed",
                        pages_fetched=pages,
                        scrobbles_stored=scrobbles,
                    )

                completed += 1
                total_scrobbles += scrobbles
                total_pages_done += pages
                if new_tracks:
                    logger.debug("%s: %d new tracks discovered", username, new_tracks)

            except Exception as e:
                logger.error("Error collecting scrobbles for %s: %s", username, e)
                with transaction(conn):
                    upsert_scrobble_progress(
                        conn, user_id,
                        status="error",
                        error_message=str(e)[:200],
                    )
                errors += 1

            # Progress logging every 5 users
            if i % 5 == 0 or i == total:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (total - i) / rate if rate > 0 else 0
                logger.info(
                    "[%d/%d] completed=%d errors=%d | scrobbles=%s pages=%s "
                    "| %.2f users/min | ETA: %.0f min",
                    i, total, completed, errors,
                    f"{total_scrobbles:,}", f"{total_pages_done:,}",
                    rate * 60, remaining / 60,
                )

        # Final stats
        elapsed = time.time() - start_time
        logger.info("=== Scrobble collection complete ===")
        logger.info(
            "Processed %d users in %.1f min: completed=%d errors=%d",
            total, elapsed / 60, completed, errors,
        )
        logger.info("Total scrobbles stored: %s", f"{total_scrobbles:,}")
        logger.info("Total API requests: %d", api.request_count)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
