"""SQLite database module with schema, connection management, and query helpers."""

import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path

from config import DB_PATH, EMBEDDINGS_PATH

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    source TEXT,
    status TEXT NOT NULL DEFAULT 'candidate',
    total_scrobbles INTEGER,
    registered_date TEXT,
    error_message TEXT,
    diversity_score REAL,
    genre_count INTEGER,
    top5_concentration REAL,
    play_entropy REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tracks (
    lastfm_id TEXT PRIMARY KEY,
    track_name TEXT NOT NULL,
    artist_name TEXT NOT NULL,
    spotify_id TEXT
);

CREATE TABLE IF NOT EXISTS user_tracks (
    user_id INTEGER NOT NULL,
    lastfm_id TEXT NOT NULL,
    play_count INTEGER NOT NULL,
    PRIMARY KEY (user_id, lastfm_id),
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (lastfm_id) REFERENCES tracks(lastfm_id)
);

CREATE TABLE IF NOT EXISTS scrape_progress (
    artist_name TEXT PRIMARY KEY,
    pages_scraped INTEGER NOT NULL DEFAULT 0,
    users_found INTEGER NOT NULL DEFAULT 0,
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS artist_tags (
    artist_name TEXT PRIMARY KEY,
    tags TEXT,
    listener_count INTEGER,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_user_tracks_user ON user_tracks(user_id);
CREATE INDEX IF NOT EXISTS idx_tracks_spotify ON tracks(spotify_id);
CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist_name);
"""


def get_connection(db_path=None):
    """Create a new database connection with WAL mode."""
    path = db_path or DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    """Initialize database schema."""
    conn = get_connection(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()
    logger.info("Database initialized at %s", db_path or DB_PATH)


@contextmanager
def transaction(conn):
    """Context manager for atomic transactions."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def add_user(conn, username, source=None, status="candidate"):
    """Insert a user if they don't already exist. Returns True if inserted."""
    try:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO users (username, source, status) VALUES (?, ?, ?)",
            (username, source, status),
        )
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        logger.error("Failed to add user %s: %s", username, e)
        return False


def add_users_batch(conn, users, source=None, status="candidate"):
    """Insert multiple users efficiently. Returns count of new users inserted."""
    cursor = conn.executemany(
        "INSERT OR IGNORE INTO users (username, source, status) VALUES (?, ?, ?)",
        [(u, source, status) for u in users],
    )
    return cursor.rowcount


def user_exists(conn, username):
    """Check if a user already exists in the database."""
    row = conn.execute(
        "SELECT 1 FROM users WHERE username = ?", (username,)
    ).fetchone()
    return row is not None


def get_user_by_username(conn, username):
    """Get a user row by username."""
    return conn.execute(
        "SELECT * FROM users WHERE username = ?", (username,)
    ).fetchone()


def set_user_status(conn, username, status, error_message=None, **kwargs):
    """Update a user's status and optional fields."""
    fields = ["status = ?", "updated_at = CURRENT_TIMESTAMP"]
    values = [status]

    if error_message is not None:
        fields.append("error_message = ?")
        values.append(error_message)

    for key, val in kwargs.items():
        fields.append(f"{key} = ?")
        values.append(val)

    values.append(username)
    conn.execute(
        f"UPDATE users SET {', '.join(fields)} WHERE username = ?",
        values,
    )


def get_users_by_status(conn, status, limit=None):
    """Get users with a given status."""
    query = "SELECT * FROM users WHERE status = ?"
    params = [status]
    if limit:
        query += " LIMIT ?"
        params.append(limit)
    return conn.execute(query, params).fetchall()


def get_user_count_by_status(conn, status):
    """Get count of users with a given status."""
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM users WHERE status = ?", (status,)
    ).fetchone()
    return row["cnt"]


def upsert_track(conn, lastfm_id, track_name, artist_name):
    """Insert or update a track."""
    conn.execute(
        "INSERT OR IGNORE INTO tracks (lastfm_id, track_name, artist_name) VALUES (?, ?, ?)",
        (lastfm_id, track_name, artist_name),
    )


def upsert_tracks_batch(conn, tracks):
    """Insert multiple tracks efficiently. tracks = list of (lastfm_id, track_name, artist_name)."""
    conn.executemany(
        "INSERT OR IGNORE INTO tracks (lastfm_id, track_name, artist_name) VALUES (?, ?, ?)",
        tracks,
    )


def add_user_track(conn, user_id, lastfm_id, play_count):
    """Add a user-track play count record."""
    conn.execute(
        "INSERT OR REPLACE INTO user_tracks (user_id, lastfm_id, play_count) VALUES (?, ?, ?)",
        (user_id, lastfm_id, play_count),
    )


def add_user_tracks_batch(conn, records):
    """Insert multiple user-track records. records = list of (user_id, lastfm_id, play_count)."""
    conn.executemany(
        "INSERT OR REPLACE INTO user_tracks (user_id, lastfm_id, play_count) VALUES (?, ?, ?)",
        records,
    )


def mark_artist_scraped(conn, artist_name, pages_scraped, users_found):
    """Record that an artist's listener pages have been scraped."""
    conn.execute(
        "INSERT OR REPLACE INTO scrape_progress (artist_name, pages_scraped, users_found) VALUES (?, ?, ?)",
        (artist_name, pages_scraped, users_found),
    )


def is_artist_scraped(conn, artist_name):
    """Check if an artist has already been scraped."""
    row = conn.execute(
        "SELECT 1 FROM scrape_progress WHERE artist_name = ?", (artist_name,)
    ).fetchone()
    return row is not None


def get_cached_artist_tags(conn, artist_name):
    """Get cached tags for an artist, or None if not cached."""
    return conn.execute(
        "SELECT * FROM artist_tags WHERE artist_name = ?", (artist_name,)
    ).fetchone()


def cache_artist_tags(conn, artist_name, tags_json, listener_count):
    """Cache artist tags and listener count."""
    conn.execute(
        "INSERT OR REPLACE INTO artist_tags (artist_name, tags, listener_count) VALUES (?, ?, ?)",
        (artist_name, tags_json, listener_count),
    )


def get_user_track_artists(conn, user_id):
    """Get all artist play data for a user (for diversity scoring)."""
    return conn.execute(
        """
        SELECT t.artist_name, SUM(ut.play_count) as total_plays
        FROM user_tracks ut
        JOIN tracks t ON ut.lastfm_id = t.lastfm_id
        WHERE ut.user_id = ?
        GROUP BY t.artist_name
        ORDER BY total_plays DESC
        """,
        (user_id,),
    ).fetchall()


def get_all_unique_artists(conn):
    """Get all unique artist names in the tracks table."""
    rows = conn.execute(
        "SELECT DISTINCT artist_name FROM tracks"
    ).fetchall()
    return [r["artist_name"] for r in rows]


def get_uncached_artists(conn, artists):
    """Filter a list of artists to only those not yet cached in artist_tags."""
    if not artists:
        return []
    cached = conn.execute("SELECT artist_name FROM artist_tags").fetchall()
    cached_set = {r["artist_name"] for r in cached}
    return [a for a in artists if a not in cached_set]


def migrate_deezer_columns(conn):
    """Add Deezer columns to tracks table if they don't exist (idempotent).

    Also clears any stored preview URLs — these contain short-lived CDN tokens
    and are fetched fresh at embedding time instead of being cached.
    """
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info(tracks)").fetchall()
    }
    for col_name, col_type in [("deezer_id", "TEXT"), ("deezer_match_status", "TEXT")]:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE tracks ADD COLUMN {col_name} {col_type}")
            logger.info("Added column tracks.%s", col_name)

    if "deezer_preview_url" in existing:
        conn.execute(
            "UPDATE tracks SET deezer_preview_url = NULL WHERE deezer_preview_url IS NOT NULL"
        )
        logger.info("Cleared expired preview URLs from tracks.deezer_preview_url")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_tracks_deezer ON tracks(deezer_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_tracks_deezer_status ON tracks(deezer_match_status)"
    )
    conn.commit()


def set_track_deezer(conn, lastfm_id, deezer_id, status):
    """Set Deezer match result for a track."""
    conn.execute(
        "UPDATE tracks SET deezer_id = ?, deezer_match_status = ? WHERE lastfm_id = ?",
        (deezer_id, status, lastfm_id),
    )


def set_track_deezer_status(conn, lastfm_id, status):
    """Update only the deezer_match_status, preserving the existing deezer_id."""
    conn.execute(
        "UPDATE tracks SET deezer_match_status = ? WHERE lastfm_id = ?",
        (status, lastfm_id),
    )


def get_tracks_for_processing(conn, embedded_ids):
    """Get tracks that need matching and/or embedding.

    Returns unmatched tracks and matched tracks not yet embedded.
    Excludes permanently unembeddable tracks (not_found, no_preview, embed_failed).
    """
    rows = conn.execute(
        """SELECT lastfm_id, track_name, artist_name, deezer_id, deezer_match_status
           FROM tracks
           WHERE deezer_match_status IS NULL
              OR deezer_match_status IN ('matched', 'error')"""
    ).fetchall()
    return [r for r in rows if r["lastfm_id"] not in embedded_ids]


def migrate_snowball_columns(conn):
    """Add taste-snowball columns to `users` and create progress tables.

    Idempotent. See docs/EXPERIMENTS.md for the design rationale.
    """
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()
    }
    additions = [
        ("cohort", "TEXT DEFAULT 'original'"),
        ("seed_artist", "TEXT"),
        ("overlap_track_pct", "REAL"),
        ("overlap_artist_pct", "REAL"),
        ("overlap_library_id", "INTEGER"),
    ]
    for col_name, col_def in additions:
        if col_name not in existing:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_def}")
            logger.info("Added column users.%s", col_name)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS taste_snowball_progress (
            cohort TEXT NOT NULL,
            candidate_username TEXT NOT NULL,
            seed_artist TEXT,
            status TEXT NOT NULL,
            overlap_track_pct REAL,
            overlap_artist_pct REAL,
            discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (cohort, candidate_username)
        );
        CREATE INDEX IF NOT EXISTS idx_snowball_progress_status
            ON taste_snowball_progress(cohort, status);

        CREATE TABLE IF NOT EXISTS taste_snowball_seed_progress (
            cohort TEXT NOT NULL,
            seed_artist TEXT NOT NULL,
            pages_scraped INTEGER NOT NULL DEFAULT 0,
            candidates_discovered INTEGER NOT NULL DEFAULT 0,
            candidates_evaluated INTEGER NOT NULL DEFAULT 0,
            candidates_passed INTEGER NOT NULL DEFAULT 0,
            completed_at TIMESTAMP,
            PRIMARY KEY (cohort, seed_artist)
        );

        CREATE INDEX IF NOT EXISTS idx_users_cohort ON users(cohort);
    """)
    conn.commit()
    logger.info("Snowball schema migrated")


def get_seed_progress(conn, cohort, seed_artist):
    """Return the seed-progress row for (cohort, seed_artist), or None."""
    return conn.execute(
        """SELECT * FROM taste_snowball_seed_progress
           WHERE cohort = ? AND seed_artist = ?""",
        (cohort, seed_artist),
    ).fetchone()


def upsert_seed_progress(conn, cohort, seed_artist, **fields):
    """Insert or update a taste_snowball_seed_progress row."""
    row = get_seed_progress(conn, cohort, seed_artist)
    if row is None:
        cols = {"cohort": cohort, "seed_artist": seed_artist}
        cols.update(fields)
        col_names = ", ".join(cols.keys())
        placeholders = ", ".join("?" for _ in cols)
        conn.execute(
            f"INSERT INTO taste_snowball_seed_progress ({col_names}) "
            f"VALUES ({placeholders})",
            list(cols.values()),
        )
    elif fields:
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        conn.execute(
            f"UPDATE taste_snowball_seed_progress SET {set_clause} "
            f"WHERE cohort = ? AND seed_artist = ?",
            list(fields.values()) + [cohort, seed_artist],
        )


def get_candidate_progress(conn, cohort, candidate_username):
    """Return the per-candidate progress row, or None."""
    return conn.execute(
        """SELECT * FROM taste_snowball_progress
           WHERE cohort = ? AND candidate_username = ?""",
        (cohort, candidate_username),
    ).fetchone()


def upsert_candidate_progress(
    conn, cohort, candidate_username, status,
    seed_artist=None, overlap_track_pct=None, overlap_artist_pct=None,
):
    """Insert or update a per-candidate progress row."""
    conn.execute(
        """INSERT INTO taste_snowball_progress
               (cohort, candidate_username, seed_artist, status,
                overlap_track_pct, overlap_artist_pct)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(cohort, candidate_username) DO UPDATE SET
               status = excluded.status,
               overlap_track_pct = COALESCE(excluded.overlap_track_pct,
                                            taste_snowball_progress.overlap_track_pct),
               overlap_artist_pct = COALESCE(excluded.overlap_artist_pct,
                                             taste_snowball_progress.overlap_artist_pct)""",
        (cohort, candidate_username, seed_artist, status,
         overlap_track_pct, overlap_artist_pct),
    )


def get_evaluated_candidates(conn, cohort):
    """Return set of candidate_usernames already evaluated (any terminal status).

    'low_activity' means the user passed the overlap threshold but their
    library was too small/sparse to provide training signal — treated as
    terminal so they aren't re-evaluated on resume.
    """
    rows = conn.execute(
        """SELECT candidate_username FROM taste_snowball_progress
           WHERE cohort = ? AND status IN
                 ('evaluated', 'passed', 'failed', 'low_activity')""",
        (cohort,),
    ).fetchall()
    return {r["candidate_username"] for r in rows}


def get_cohort_passers(conn, cohort):
    """Return list of usernames that passed in the given cohort. Used for friend-expand."""
    rows = conn.execute(
        """SELECT candidate_username FROM taste_snowball_progress
           WHERE cohort = ? AND status = 'passed'""",
        (cohort,),
    ).fetchall()
    return [r["candidate_username"] for r in rows]


def get_cohort_summary(conn, cohort):
    """Return aggregate counts for a cohort: discovered/evaluated/passed/failed."""
    rows = conn.execute(
        """SELECT status, COUNT(*) as cnt FROM taste_snowball_progress
           WHERE cohort = ? GROUP BY status""",
        (cohort,),
    ).fetchall()
    return {r["status"]: r["cnt"] for r in rows}


def migrate_scrobble_tables(conn):
    """Add scrobbles + scrobble_progress tables if they don't exist (idempotent)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS scrobbles (
            user_id INTEGER NOT NULL,
            lastfm_id TEXT NOT NULL,
            listened_at INTEGER NOT NULL,
            PRIMARY KEY (user_id, lastfm_id, listened_at),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE INDEX IF NOT EXISTS idx_scrobbles_user_time
            ON scrobbles(user_id, listened_at);

        CREATE TABLE IF NOT EXISTS scrobble_progress (
            user_id INTEGER PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'pending',
            total_pages INTEGER,
            pages_fetched INTEGER NOT NULL DEFAULT 0,
            total_scrobbles INTEGER,
            scrobbles_stored INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    logger.info("Scrobble tables migrated")


def add_scrobbles_batch(conn, records):
    """Insert scrobble records. records = list of (user_id, lastfm_id, listened_at).
    Uses INSERT OR IGNORE for deduplication on resume."""
    conn.executemany(
        "INSERT OR IGNORE INTO scrobbles (user_id, lastfm_id, listened_at) VALUES (?, ?, ?)",
        records,
    )


def upsert_scrobble_progress(conn, user_id, **kwargs):
    """Insert or update a scrobble_progress row. Pass fields as kwargs."""
    # Check if row exists
    row = conn.execute(
        "SELECT 1 FROM scrobble_progress WHERE user_id = ?", (user_id,)
    ).fetchone()

    if row is None:
        # Insert with defaults
        fields = {"user_id": user_id, "status": "pending", "pages_fetched": 0,
                  "scrobbles_stored": 0}
        fields.update(kwargs)
        cols = ", ".join(fields.keys())
        placeholders = ", ".join("?" for _ in fields)
        conn.execute(
            f"INSERT INTO scrobble_progress ({cols}) VALUES ({placeholders})",
            list(fields.values()),
        )
    else:
        # Update specified fields
        if kwargs:
            set_clause = ", ".join(f"{k} = ?" for k in kwargs)
            conn.execute(
                f"UPDATE scrobble_progress SET {set_clause} WHERE user_id = ?",
                list(kwargs.values()) + [user_id],
            )


def get_users_needing_scrobbles(conn):
    """Get users where scrobble_progress status is 'pending' or 'in_progress'."""
    return conn.execute(
        """SELECT u.id, u.username, sp.status, sp.pages_fetched, sp.total_pages
           FROM scrobble_progress sp
           JOIN users u ON sp.user_id = u.id
           WHERE sp.status IN ('pending', 'in_progress')
           ORDER BY u.id""",
    ).fetchall()


def get_scrobble_stats(conn):
    """Get scrobble collection statistics."""
    stats = {}

    # Check if scrobble tables exist
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}

    if "scrobble_progress" not in tables:
        return stats

    # Progress by status
    rows = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM scrobble_progress GROUP BY status"
    ).fetchall()
    stats["scrobble_users_by_status"] = {r["status"]: r["cnt"] for r in rows}

    # Total scrobble records
    if "scrobbles" in tables:
        row = conn.execute("SELECT COUNT(*) as cnt FROM scrobbles").fetchone()
        stats["total_scrobbles"] = row["cnt"]
    else:
        stats["total_scrobbles"] = 0

    return stats


def get_stats(conn):
    """Get collection statistics."""
    stats = {}

    # User counts by status
    rows = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM users GROUP BY status"
    ).fetchall()
    stats["users_by_status"] = {r["status"]: r["cnt"] for r in rows}
    stats["total_users"] = sum(stats["users_by_status"].values())

    # Track counts
    row = conn.execute("SELECT COUNT(*) as cnt FROM tracks").fetchone()
    stats["total_tracks"] = row["cnt"]

    # User-track records
    row = conn.execute("SELECT COUNT(*) as cnt FROM user_tracks").fetchone()
    stats["total_user_tracks"] = row["cnt"]

    # Artists scraped
    row = conn.execute("SELECT COUNT(*) as cnt FROM scrape_progress").fetchone()
    stats["artists_scraped"] = row["cnt"]

    # Artist tags cached
    row = conn.execute("SELECT COUNT(*) as cnt FROM artist_tags").fetchone()
    stats["artist_tags_cached"] = row["cnt"]

    # Diversity stats (if computed)
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM users WHERE diversity_score IS NOT NULL"
    ).fetchone()
    stats["users_scored"] = row["cnt"]

    if stats["users_scored"] > 0:
        row = conn.execute(
            """
            SELECT AVG(diversity_score) as avg_score,
                   AVG(genre_count) as avg_genres,
                   AVG(play_entropy) as avg_entropy,
                   AVG(top5_concentration) as avg_top5
            FROM users WHERE diversity_score IS NOT NULL
            """
        ).fetchone()
        stats["avg_diversity_score"] = row["avg_score"]
        stats["avg_genre_count"] = row["avg_genres"]
        stats["avg_play_entropy"] = row["avg_entropy"]
        stats["avg_top5_concentration"] = row["avg_top5"]

    # Deezer matching stats
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(tracks)").fetchall()
    }
    if "deezer_match_status" in existing_cols:
        rows = conn.execute(
            """SELECT deezer_match_status, COUNT(*) as cnt
               FROM tracks GROUP BY deezer_match_status"""
        ).fetchall()
        stats["deezer_by_status"] = {
            (r["deezer_match_status"] or "unmatched"): r["cnt"] for r in rows
        }
    else:
        stats["deezer_by_status"] = {}

    # Embedding stats
    stats["embeddings_computed"] = 0
    stats["embeddings_file_size_mb"] = 0.0
    if EMBEDDINGS_PATH.exists():
        try:
            import h5py
            with h5py.File(str(EMBEDDINGS_PATH), "r") as f:
                if "track_ids" in f:
                    stats["embeddings_computed"] = len(f["track_ids"])
            stats["embeddings_file_size_mb"] = EMBEDDINGS_PATH.stat().st_size / (1024 * 1024)
        except Exception:
            pass

    return stats
