#!/usr/bin/env python3
"""Prepare Training Data — default dataset.

Historical explicit-execution research tooling; not a supported workflow.
Read docs/COMPONENTS.md and docs/DATA_AND_RIGHTS.md before considering use.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    EMBEDDINGS_PATH,
    TRAINING_DATA_DIR,
    MIN_LIBRARY_SIZE,
    MIN_SUPER_GENRES,
    MAX_TOP5_CONCENTRATION,
)
from src.db import get_connection
from library.src.preprocessing import (
    DatasetVariant,
    build_embedding_index,
    save_embedding_index,
    prepare_dataset_variant,
)

LOG_DIR = Path(__file__).parent.parent.parent / "data"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "prepare_training_data.log"),
    ],
)
logger = logging.getLogger(__name__)


DEFAULT_VARIANT = DatasetVariant(
    name="baseline_all_diverse",
    description=(
        "Default training dataset: all collected users passing the project's "
        "diversity filter (min genres, max top-5 concentration, min library size). "
        "Log-normalized scores, no track-level filter."
    ),
    user_filter_sql="",
    user_filter_params=(),
    track_filter="all",
    score_method="log_norm",
    min_library_size_after_filter=MIN_LIBRARY_SIZE,
    min_genres=MIN_SUPER_GENRES,
    max_top5_conc=MAX_TOP5_CONCENTRATION,
    min_library_size=MIN_LIBRARY_SIZE,
)


def main():
    logger.info("=" * 60)
    logger.info("Preparing training data (default variant)")
    logger.info("=" * 60)

    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Embedding index — built once and shared across all variants
    embedding_index_path = TRAINING_DATA_DIR / "embedding_index.json"
    if embedding_index_path.exists():
        import json
        logger.info("Loading existing embedding index")
        with open(embedding_index_path) as f:
            embedding_index = json.load(f)
        logger.info("Loaded index with %d tracks", len(embedding_index))
    else:
        embedding_index = build_embedding_index(EMBEDDINGS_PATH)
        save_embedding_index(embedding_index, embedding_index_path)

    conn = get_connection()
    try:
        result = prepare_dataset_variant(
            variant=DEFAULT_VARIANT,
            training_dir=TRAINING_DATA_DIR,
            embedding_index=embedding_index,
            conn=conn,
        )
    finally:
        conn.close()

    logger.info("=" * 60)
    logger.info("Done! %s", result)
    logger.info("Output directory: %s", TRAINING_DATA_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
