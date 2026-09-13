#!/usr/bin/env python3
"""Convert chunked HDF5 embeddings to an NPY memmap for fast training reads.

Historical explicit-execution research tooling; not a supported workflow.
Read docs/COMPONENTS.md and docs/DATA_AND_RIGHTS.md before considering use.
"""

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import EMBEDDINGS_NPY_PATH, EMBEDDINGS_PATH


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def convert_embeddings(
    h5_path: Path,
    npy_path: Path,
    block_size: int,
    overwrite: bool,
) -> None:
    if npy_path.exists() and not overwrite:
        logger.info("Already exists: %s", npy_path)
        logger.info("Use --overwrite to rebuild it.")
        return

    h5_path = h5_path.expanduser().resolve()
    npy_path = npy_path.expanduser().resolve()
    npy_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(h5_path), "r") as h5:
        embeddings = h5["embeddings"]
        logger.info(
            "Converting %s -> %s | shape=%s dtype=%s chunks=%s compression=%s",
            h5_path,
            npy_path,
            embeddings.shape,
            embeddings.dtype,
            embeddings.chunks,
            embeddings.compression,
        )

        out = np.lib.format.open_memmap(
            npy_path,
            mode="w+",
            dtype=embeddings.dtype,
            shape=embeddings.shape,
        )

        total = embeddings.shape[0]
        for start in range(0, total, block_size):
            end = min(start + block_size, total)
            out[start:end] = embeddings[start:end]
            if start == 0 or end == total or end % (block_size * 10) == 0:
                logger.info("Copied %d / %d rows (%.1f%%)", end, total, 100 * end / total)

        out.flush()

    logger.info("Done. Training will automatically prefer %s when present.", npy_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert data/embeddings.h5 to data/embeddings.npy for faster training"
    )
    parser.add_argument("--h5", type=Path, default=EMBEDDINGS_PATH)
    parser.add_argument("--out", type=Path, default=EMBEDDINGS_NPY_PATH)
    parser.add_argument("--block-size", type=int, default=20000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    convert_embeddings(args.h5, args.out, args.block_size, args.overwrite)


if __name__ == "__main__":
    main()
