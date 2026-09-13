#!/usr/bin/env python3
"""Match & Embed Script

Historical explicit-execution research tooling; not a supported workflow.
Read docs/COMPONENTS.md and docs/DATA_AND_RIGHTS.md before considering use.
"""

import sys
import time
import logging
import threading
import queue
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import EMBEDDINGS_PATH, MERT_BATCH_SIZE, MERT_EMBEDDING_DIM
from src.db import (
    init_db,
    get_connection,
    migrate_deezer_columns,
    get_tracks_for_processing,
    set_track_deezer,
    set_track_deezer_status,
)
from src.deezer_api import DeezerClient, DeezerRetryableError
from src.embedder import MERTEmbedder

LOG_DIR = Path(__file__).parent.parent / "data"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "04_match_and_embed.log"),
    ],
)
logger = logging.getLogger(__name__)

DOWNLOAD_QUEUE_SIZE = 128  # MP3s buffered ahead of GPU/CPU
NUM_DOWNLOAD_THREADS = 16  # parallel Deezer API + download threads
DB_COMMIT_INTERVAL = 1    # commit after every write — avoids SQLite lock contention across 16 threads
FLUSH_INTERVAL = 10        # HDF5 flushes per N batches
PROGRESS_LOG_INTERVAL = 100
SENTINEL = None


def load_existing_ids(h5_path):
    """Return set of lastfm_ids already written to the HDF5 file."""
    if not h5_path.exists():
        return set()
    try:
        with h5py.File(str(h5_path), "r") as f:
            if "track_ids" in f:
                return set(f["track_ids"].asstr()[:])
        return set()
    except Exception as e:
        logger.warning("Could not read existing HDF5: %s", e)
        return set()


def downloader_thread(tracks, deezer, q):
    """Producer thread: match → fresh URL → download → enqueue.

    Opens its own DB connection for writing match results.
    """
    conn = get_connection()
    pending_commits = 0

    def maybe_commit():
        nonlocal pending_commits
        pending_commits += 1
        if pending_commits >= DB_COMMIT_INTERVAL:
            conn.commit()
            pending_commits = 0

    for track in tracks:
        lastfm_id = track["lastfm_id"]
        deezer_id = track["deezer_id"]

        # Step 1: match if not yet done
        if deezer_id is None:
            try:
                result = deezer.search_track(track["artist_name"], track["track_name"])
            except DeezerRetryableError:
                set_track_deezer(conn, lastfm_id, None, "error")
                maybe_commit()
                continue

            if result is None:
                set_track_deezer(conn, lastfm_id, None, "not_found")
                maybe_commit()
                continue

            deezer_id = result["deezer_id"]
            set_track_deezer(conn, lastfm_id, deezer_id, "matched")
            maybe_commit()

        # Step 2: fresh preview URL (stored URLs have expired tokens)
        try:
            preview_url = deezer.get_fresh_preview_url(deezer_id)
        except DeezerRetryableError:
            logger.debug("Rate-limited fetching URL for %s", lastfm_id)
            continue

        if not preview_url:
            set_track_deezer(conn, lastfm_id, deezer_id, "no_preview")
            maybe_commit()
            continue

        # Step 3: download
        mp3_bytes = deezer.download_preview(preview_url)
        if mp3_bytes is not None:
            q.put((lastfm_id, mp3_bytes))
        else:
            logger.debug("Download failed for %s", lastfm_id)

    if pending_commits > 0:
        conn.commit()
    conn.close()


def main():
    init_db()
    conn = get_connection()
    migrate_deezer_columns(conn)

    embedded_ids = load_existing_ids(EMBEDDINGS_PATH)
    logger.info("Existing embeddings: %d", len(embedded_ids))

    tracks = get_tracks_for_processing(conn, embedded_ids)
    conn.close()

    total = len(tracks)
    if total == 0:
        logger.info("Nothing to process — all tracks already embedded.")
        return

    logger.info("Tracks to process: %d", total)

    embedder = MERTEmbedder()
    deezer = DeezerClient()

    # Open HDF5 for append
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    h5 = h5py.File(str(EMBEDDINGS_PATH), "a")
    if "track_ids" not in h5:
        h5.create_dataset(
            "track_ids", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype()
        )
        h5.create_dataset(
            "embeddings", shape=(0, MERT_EMBEDDING_DIM),
            maxshape=(None, MERT_EMBEDDING_DIM),
            dtype=np.float32, chunks=(1000, MERT_EMBEDDING_DIM),
        )

    # Start parallel producer threads
    # Interleave tracks across threads so each gets a mix of matched/unmatched
    dl_queue = queue.Queue(maxsize=DOWNLOAD_QUEUE_SIZE)
    chunks = [tracks[i::NUM_DOWNLOAD_THREADS] for i in range(NUM_DOWNLOAD_THREADS)]
    dl_threads = [
        threading.Thread(
            target=downloader_thread, args=(chunk, deezer, dl_queue), daemon=True
        )
        for chunk in chunks
    ]
    for t in dl_threads:
        t.start()

    # Sentinel coordinator: send SENTINEL only after all downloaders finish
    def _sentinel_coordinator():
        for t in dl_threads:
            t.join()
        dl_queue.put(SENTINEL)

    threading.Thread(target=_sentinel_coordinator, daemon=True).start()

    # Consumer: batch embed and write to HDF5
    # Own DB connection for recording embed failures on the main thread
    embed_conn = get_connection()
    batch_ids = []
    batch_audio = []
    embedded_count = 0
    failed_count = 0
    batches_since_flush = 0
    start_time = time.time()

    def process_batch():
        nonlocal embedded_count, failed_count, batches_since_flush

        embeddings = embedder.embed_batch(batch_audio)
        valid_ids, valid_embeddings = [], []
        failed_ids = []
        for track_id, emb in zip(batch_ids, embeddings):
            if emb is not None:
                valid_ids.append(track_id)
                valid_embeddings.append(emb)
            else:
                failed_ids.append(track_id)
                failed_count += 1

        # Mark decode/inference failures so they aren't retried forever
        for fid in failed_ids:
            set_track_deezer_status(embed_conn, fid, "embed_failed")
        if failed_ids:
            embed_conn.commit()

        if valid_ids:
            n = h5["track_ids"].shape[0]
            k = len(valid_ids)
            h5["track_ids"].resize(n + k, axis=0)
            h5["embeddings"].resize(n + k, axis=0)
            h5["track_ids"][n:] = valid_ids
            h5["embeddings"][n:] = np.array(valid_embeddings)
            embedded_count += k

        batches_since_flush += 1
        if batches_since_flush >= FLUSH_INTERVAL:
            h5.flush()
            batches_since_flush = 0

        batch_ids.clear()
        batch_audio.clear()

        if embedded_count > 0 and embedded_count % PROGRESS_LOG_INTERVAL < MERT_BATCH_SIZE:
            elapsed = time.time() - start_time
            rate = embedded_count / elapsed if elapsed > 0 else 0
            remaining = (total - embedded_count) / rate if rate > 0 else 0
            logger.info(
                "Embedded: %d/%d (%.1f%%) | failed=%d | %.2f/sec | ETA: %.1f hrs",
                embedded_count, total, 100 * embedded_count / total,
                failed_count, rate, remaining / 3600,
            )

    while True:
        item = dl_queue.get()
        if item is SENTINEL:
            break
        lastfm_id, mp3_bytes = item
        batch_ids.append(lastfm_id)
        batch_audio.append(mp3_bytes)
        if len(batch_ids) >= MERT_BATCH_SIZE:
            process_batch()

    if batch_ids:
        process_batch()

    h5.flush()
    h5.close()
    embed_conn.close()

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(
        "Complete in %.1f hrs | embedded=%d | failed=%d | HDF5=%.1f MB",
        elapsed / 3600, embedded_count, failed_count,
        EMBEDDINGS_PATH.stat().st_size / (1024 * 1024),
    )


if __name__ == "__main__":
    main()
