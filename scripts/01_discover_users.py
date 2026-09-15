#!/usr/bin/env python3
"""User Discovery Script

Historical explicit-execution research tooling; not a supported workflow.
Read docs/COMPONENTS.md and docs/DATA_AND_RIGHTS.md before considering use.
"""

import sys
import os
import random
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SEED_ARTISTS,
    SCRAPE_PAGES_PER_ARTIST,
    FRIENDS_SAMPLE_SIZE,
    FRIENDS_PER_USER,
    MIN_SCROBBLES,
)
from src.db import (
    init_db,
    get_connection,
    transaction,
    add_users_batch,
    is_artist_scraped,
    mark_artist_scraped,
    get_users_by_status,
    get_user_count_by_status,
    set_user_status,
    user_exists,
    get_stats,
)
from src.lastfm_api import LastFMClient
from src.scraper import ListenerScraper

# Set up logging
LOG_DIR = Path(__file__).parent.parent / "data"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "01_discover.log"),
    ],
)
logger = logging.getLogger(__name__)


def phase_a_scrape(conn, scraper):
    """Phase A: Scrape artist listener pages for candidate usernames."""
    logger.info("=== Phase A: Scraping artist listener pages ===")

    artists_to_scrape = [a for a in SEED_ARTISTS if not is_artist_scraped(conn, a)]
    if not artists_to_scrape:
        logger.info("All %d seed artists already scraped, skipping Phase A", len(SEED_ARTISTS))
        return

    logger.info(
        "Scraping %d artists (%d already done)",
        len(artists_to_scrape),
        len(SEED_ARTISTS) - len(artists_to_scrape),
    )

    total_new = 0
    for i, artist in enumerate(artists_to_scrape, 1):
        logger.info(
            "[%d/%d] Scraping listeners of: %s",
            i, len(artists_to_scrape), artist,
        )

        usernames = scraper.scrape_artist_listeners(
            artist, max_pages=SCRAPE_PAGES_PER_ARTIST
        )

        if usernames:
            with transaction(conn):
                new_count = add_users_batch(
                    conn, usernames, source=f"scrape:{artist}"
                )
                mark_artist_scraped(conn, artist, SCRAPE_PAGES_PER_ARTIST, len(usernames))
                total_new += new_count
                logger.info(
                    "  Found %d users, %d new candidates (total new: %d)",
                    len(usernames), new_count, total_new,
                )
        else:
            with transaction(conn):
                mark_artist_scraped(conn, artist, 0, 0)
            logger.warning("  No users found for %s", artist)

    logger.info("Phase A complete: %d new candidates added", total_new)


def phase_b_friends(conn, api):
    """Phase B: Expand candidate pool via friends of discovered users."""
    logger.info("=== Phase B: Friends expansion ===")

    candidates = get_users_by_status(conn, "candidate")
    if not candidates:
        logger.warning("No candidates available for friends expansion")
        return

    sample_size = min(FRIENDS_SAMPLE_SIZE, len(candidates))
    sample = random.sample(list(candidates), sample_size)
    logger.info("Sampling friends from %d users", sample_size)

    total_new = 0
    for i, user in enumerate(sample, 1):
        username = user["username"]
        friends = api.get_user_friends(username, limit=FRIENDS_PER_USER)

        if friends:
            with transaction(conn):
                new_count = add_users_batch(
                    conn, friends, source=f"friends:{username}"
                )
                total_new += new_count

        if i % 50 == 0:
            logger.info(
                "  [%d/%d] Processed friends, %d new candidates so far",
                i, sample_size, total_new,
            )

    logger.info("Phase B complete: %d new candidates from friends", total_new)


def phase_c_validate(conn, api):
    """Phase C: Validate candidates via user.getInfo, reject low-scrobble users."""
    logger.info("=== Phase C: Pre-validation ===")

    candidates = get_users_by_status(conn, "candidate")
    total = len(candidates)
    if total == 0:
        logger.info("No candidates to validate")
        return

    logger.info("Validating %d candidates (min scrobbles: %d)", total, MIN_SCROBBLES)

    validated = 0
    rejected = 0
    errors = 0

    for i, user in enumerate(candidates, 1):
        username = user["username"]
        info = api.get_user_info(username)

        if info is None:
            with transaction(conn):
                set_user_status(conn, username, "rejected", error_message="API lookup failed")
            errors += 1
        elif info["playcount"] < MIN_SCROBBLES:
            with transaction(conn):
                set_user_status(
                    conn, username, "rejected",
                    total_scrobbles=info["playcount"],
                )
            rejected += 1
        else:
            with transaction(conn):
                set_user_status(
                    conn, username, "validated",
                    total_scrobbles=info["playcount"],
                    registered_date=info.get("registered"),
                )
            validated += 1

        if i % 100 == 0:
            elapsed_pct = i / total * 100
            logger.info(
                "  [%d/%d %.0f%%] validated=%d rejected=%d errors=%d",
                i, total, elapsed_pct, validated, rejected, errors,
            )

    logger.info(
        "Phase C complete: %d validated, %d rejected, %d errors (of %d)",
        validated, rejected, errors, total,
    )


def main():
    logger.info("Starting user discovery pipeline")

    init_db()
    conn = get_connection()
    api = LastFMClient()
    scraper = ListenerScraper()

    try:
        # Phase A: Scrape artist listener pages
        phase_a_scrape(conn, scraper)

        stats = get_stats(conn)
        logger.info(
            "After Phase A: %s",
            {k: v for k, v in stats["users_by_status"].items()},
        )

        # Phase B: Friends expansion
        phase_b_friends(conn, api)

        stats = get_stats(conn)
        logger.info(
            "After Phase B: %s",
            {k: v for k, v in stats["users_by_status"].items()},
        )

        # Phase C: Validate candidates
        phase_c_validate(conn, api)

        # Final stats
        stats = get_stats(conn)
        logger.info("=== Discovery complete ===")
        logger.info("Users by status: %s", stats["users_by_status"])
        logger.info("Total API requests: %d", api.request_count)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
