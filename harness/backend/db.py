"""Harness-local SQLite tables. Shares the file with the training DB but
all tables are prefixed `harness_` so the training schema is untouched."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from config import DB_PATH

HARNESS_SCHEMA = """
CREATE TABLE IF NOT EXISTS harness_libraries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    source          TEXT    NOT NULL,
    match_policy    TEXT    NOT NULL,
    total_count     INTEGER NOT NULL,
    matched_count   INTEGER NOT NULL,
    raw_xml_path    TEXT,
    display_name    TEXT
);

CREATE TABLE IF NOT EXISTS harness_library_tracks (
    library_id       INTEGER NOT NULL,
    apple_track_id   INTEGER NOT NULL,
    artist           TEXT,
    title            TEXT,
    album            TEXT,
    year             INTEGER,
    genre            TEXT,
    duration_ms      INTEGER,
    play_count       INTEGER,
    embedding_index  INTEGER,
    embedding_key    TEXT,
    match_score      REAL,
    PRIMARY KEY (library_id, apple_track_id),
    FOREIGN KEY (library_id) REFERENCES harness_libraries(id)
);

CREATE INDEX IF NOT EXISTS idx_harness_library_tracks_lib
    ON harness_library_tracks(library_id);

CREATE TABLE IF NOT EXISTS harness_runs (
    id            TEXT PRIMARY KEY,
    model_id      TEXT NOT NULL,
    model_name    TEXT NOT NULL,
    model_path    TEXT NOT NULL,
    library_id    INTEGER NOT NULL,
    created_at    TEXT NOT NULL,
    latency_ms    INTEGER,
    results_json  TEXT NOT NULL,
    ranking_mode  TEXT,
    threshold     REAL,
    FOREIGN KEY (library_id) REFERENCES harness_libraries(id)
);

CREATE TABLE IF NOT EXISTS harness_feedback (
    run_id      TEXT NOT NULL,
    track_key   TEXT NOT NULL,
    vote        TEXT NOT NULL CHECK (vote IN ('up','down')),
    created_at  TEXT NOT NULL,
    PRIMARY KEY (run_id, track_key),
    FOREIGN KEY (run_id) REFERENCES harness_runs(id)
);
"""


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(HARNESS_SCHEMA)
        # Idempotent additive migrations for installs that pre-date the
        # ranking_mode / threshold columns. ALTER TABLE ADD COLUMN is a no-op
        # on a fresh DB (where the columns already exist via CREATE TABLE)
        # and a one-time add on existing DBs.
        for ddl in (
            "ALTER TABLE harness_runs ADD COLUMN ranking_mode TEXT",
            "ALTER TABLE harness_runs ADD COLUMN threshold REAL",
        ):
            try:
                conn.execute(ddl)
            except sqlite3.OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise
        conn.commit()


@contextmanager
def get_connection():
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()


def db_path() -> Path:
    return Path(DB_PATH)
