#!/usr/bin/env python3
"""Diversity Scoring Script

Historical explicit-execution research tooling; not a supported workflow.
Read docs/COMPONENTS.md and docs/DATA_AND_RIGHTS.md before considering use.
"""

import sys
import json
import math
import time
import logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    SUPER_GENRE_MAP,
    MIN_SUPER_GENRES,
    MAX_TOP5_CONCENTRATION,
)
from src.db import (
    init_db,
    get_connection,
    transaction,
    get_users_by_status,
    get_user_track_artists,
    get_all_unique_artists,
    get_uncached_artists,
    get_cached_artist_tags,
    cache_artist_tags,
    set_user_status,
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
        logging.FileHandler(LOG_DIR / "03_diversity.log"),
    ],
)
logger = logging.getLogger(__name__)


def compute_entropy(play_counts):
    """Compute Shannon entropy of a play count distribution."""
    total = sum(play_counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in play_counts:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def compute_top5_concentration(artist_plays):
    """Compute fraction of total plays in top 5 artists."""
    if not artist_plays:
        return 1.0
    sorted_plays = sorted(artist_plays, reverse=True)
    total = sum(sorted_plays)
    if total == 0:
        return 1.0
    top5 = sum(sorted_plays[:5])
    return top5 / total


def classify_tag(tag_name):
    """Map a Last.fm tag to a super-genre, or None if no match."""
    tag_lower = tag_name.lower().strip()
    for genre, keywords in SUPER_GENRE_MAP.items():
        if tag_lower in keywords:
            return genre
    # Partial matching for compound tags
    for genre, keywords in SUPER_GENRE_MAP.items():
        for keyword in keywords:
            if keyword in tag_lower or tag_lower in keyword:
                return genre
    return None


def get_artist_genres(conn, artist_name):
    """Get super-genres for an artist from cached tags."""
    cached = get_cached_artist_tags(conn, artist_name)
    if cached is None or cached["tags"] is None:
        return set()

    try:
        tags = json.loads(cached["tags"])
    except (json.JSONDecodeError, TypeError):
        return set()

    genres = set()
    for tag_name, count in tags:
        if count < 10:  # Skip very low-count tags
            continue
        genre = classify_tag(tag_name)
        if genre:
            genres.add(genre)
    return genres


def fetch_and_cache_artist_data(conn, api, artists):
    """Fetch and cache tags + info for a list of artists."""
    total = len(artists)
    if total == 0:
        return

    logger.info("Fetching data for %d uncached artists", total)
    start_time = time.time()

    for i, artist in enumerate(artists, 1):
        # Fetch tags
        tags = api.get_artist_top_tags(artist)
        tags_json = json.dumps(tags) if tags else None

        # Fetch listener count
        info = api.get_artist_info(artist)
        listener_count = info["listeners"] if info else None

        with transaction(conn):
            cache_artist_tags(conn, artist, tags_json, listener_count)

        if i % 500 == 0 or i == total:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (total - i) / rate if rate > 0 else 0
            logger.info(
                "  [%d/%d] Cached artist data | %.1f artists/min | ETA: %.0f min",
                i, total, rate * 60, remaining / 60,
            )


def score_user(conn, user):
    """Compute all diversity metrics for a single user.

    Returns dict with: unique_artists, play_entropy, top5_concentration,
    genre_count, genres, median_popularity, diversity_score
    """
    artist_rows = get_user_track_artists(conn, user["id"])
    if not artist_rows:
        return None

    artist_plays = {r["artist_name"]: r["total_plays"] for r in artist_rows}
    play_values = list(artist_plays.values())

    # Tier 1: Free metrics
    unique_artists = len(artist_plays)
    play_entropy = compute_entropy(play_values)
    top5_conc = compute_top5_concentration(play_values)

    # Tier 2: Genre diversity
    total_plays = sum(play_values)
    genre_plays = defaultdict(int)

    for artist_name, plays in artist_plays.items():
        genres = get_artist_genres(conn, artist_name)
        for genre in genres:
            genre_plays[genre] += plays

    # Count genres with meaningful representation (>5% of total plays)
    significant_genres = set()
    for genre, plays in genre_plays.items():
        if total_plays > 0 and plays / total_plays > 0.05:
            significant_genres.add(genre)
    genre_count = len(significant_genres)

    # Tier 2 bonus: Popularity mix (median listener count of user's artists)
    popularities = []
    for artist_name in artist_plays:
        cached = get_cached_artist_tags(conn, artist_name)
        if cached and cached["listener_count"] is not None:
            popularities.append(cached["listener_count"])

    median_popularity = None
    if popularities:
        popularities.sort()
        mid = len(popularities) // 2
        median_popularity = popularities[mid]

    # Composite diversity score (simple weighted sum, can be refined later)
    # Normalize components to ~0-1 range
    entropy_norm = min(play_entropy / 8.0, 1.0)  # max entropy for 256 artists ≈ 8
    genre_norm = min(genre_count / 8.0, 1.0)  # 8+ genres is excellent
    concentration_norm = 1.0 - top5_conc  # lower concentration = better

    diversity_score = (
        0.3 * entropy_norm
        + 0.4 * genre_norm
        + 0.3 * concentration_norm
    )

    return {
        "unique_artists": unique_artists,
        "play_entropy": play_entropy,
        "top5_concentration": top5_conc,
        "genre_count": genre_count,
        "genres": significant_genres,
        "median_popularity": median_popularity,
        "diversity_score": diversity_score,
    }


def main():
    logger.info("Starting diversity scoring")

    init_db()
    conn = get_connection()
    api = LastFMClient()

    try:
        # Step 1: Fetch and cache artist data for all unique artists
        all_artists = get_all_unique_artists(conn)
        logger.info("Total unique artists in dataset: %d", len(all_artists))

        uncached = get_uncached_artists(conn, all_artists)
        if uncached:
            fetch_and_cache_artist_data(conn, api, uncached)
        else:
            logger.info("All artist data already cached")

        # Step 2: Score each collected user
        users = get_users_by_status(conn, "collected")
        total = len(users)
        logger.info("Scoring %d collected users", total)

        # Track stats for summary
        all_scores = []
        all_genre_counts = []
        all_entropies = []
        all_concentrations = []
        passing = 0
        start_time = time.time()

        # Pre-compute median entropy after first pass for threshold
        entropy_values = []

        for i, user in enumerate(users, 1):
            metrics = score_user(conn, user)

            if metrics is None:
                continue

            all_scores.append(metrics["diversity_score"])
            all_genre_counts.append(metrics["genre_count"])
            all_entropies.append(metrics["play_entropy"])
            all_concentrations.append(metrics["top5_concentration"])
            entropy_values.append(metrics["play_entropy"])

            # Store metrics in DB
            with transaction(conn):
                set_user_status(
                    conn,
                    user["username"],
                    "collected",  # Keep status as collected for now
                    diversity_score=metrics["diversity_score"],
                    genre_count=metrics["genre_count"],
                    top5_concentration=metrics["top5_concentration"],
                    play_entropy=metrics["play_entropy"],
                )

            if i % 200 == 0 or i == total:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                logger.info(
                    "  [%d/%d] Scored users | %.1f users/sec",
                    i, total, rate,
                )

        # Step 3: Apply diversity filter
        if not entropy_values:
            logger.warning("No users scored, nothing to filter")
            return

        median_entropy = sorted(entropy_values)[len(entropy_values) // 2]
        logger.info("Median play entropy: %.2f", median_entropy)

        # Re-read users with scores and apply filter
        scored_users = conn.execute(
            """
            SELECT * FROM users
            WHERE status = 'collected' AND diversity_score IS NOT NULL
            """
        ).fetchall()

        passing = 0
        failing = 0
        for user in scored_users:
            passes = (
                user["genre_count"] >= MIN_SUPER_GENRES
                and user["top5_concentration"] < MAX_TOP5_CONCENTRATION
                and user["play_entropy"] >= median_entropy
            )
            if not passes:
                failing += 1

            if passes:
                passing += 1

        # Print summary
        logger.info("=== Diversity Scoring Complete ===")
        logger.info("Total users scored: %d", len(all_scores))

        if all_scores:
            logger.info(
                "Diversity score: min=%.2f avg=%.2f max=%.2f",
                min(all_scores), sum(all_scores) / len(all_scores), max(all_scores),
            )
        if all_genre_counts:
            logger.info(
                "Genre count: min=%d avg=%.1f max=%d",
                min(all_genre_counts),
                sum(all_genre_counts) / len(all_genre_counts),
                max(all_genre_counts),
            )
        if all_entropies:
            logger.info(
                "Play entropy: min=%.2f avg=%.2f max=%.2f",
                min(all_entropies),
                sum(all_entropies) / len(all_entropies),
                max(all_entropies),
            )
        if all_concentrations:
            logger.info(
                "Top-5 concentration: min=%.2f avg=%.2f max=%.2f",
                min(all_concentrations),
                sum(all_concentrations) / len(all_concentrations),
                max(all_concentrations),
            )

        logger.info(
            "Passing filter (%d+ genres, <%.0f%% top-5, entropy >= median): %d / %d (%.1f%%)",
            MIN_SUPER_GENRES,
            MAX_TOP5_CONCENTRATION * 100,
            passing,
            len(scored_users),
            passing / len(scored_users) * 100 if scored_users else 0,
        )
        logger.info("Total API requests: %d", api.request_count)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
