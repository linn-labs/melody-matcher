"""PyTorch Dataset and DataLoader for taste-scoring training."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """Efficient embedding lookup with a memmap fast path.

    If data/embeddings.npy exists next to embeddings.h5, training uses the NPY
    memmap. Random row access against the chunked HDF5 file is much slower.
    """

    def __init__(self, embeddings_path: Path):
        self.embeddings_path = Path(embeddings_path)
        self.npy_path = self._resolve_npy_path(self.embeddings_path)
        self.h5 = None
        self.embeddings = None
        self.backend = ""
        self.shape = (0, 0)
        self.dim = 0
        self._load_metadata()

    def _load_metadata(self):
        if self.npy_path.exists():
            meta = np.load(self.npy_path, mmap_mode="r")
            self.backend = "npy-memmap"
            self.shape = meta.shape
            self.dim = meta.shape[1]
            del meta
        else:
            with h5py.File(str(self.embeddings_path), "r") as h5:
                self.backend = "hdf5"
                self.shape = h5["embeddings"].shape
                self.dim = self.shape[1]

        logger.info(
            "Embedding store metadata (%s): %d tracks x %d dims",
            self.backend,
            self.shape[0],
            self.dim,
        )

    @staticmethod
    def _resolve_npy_path(path: Path) -> Path:
        if path.suffix == ".npy":
            return path
        return path.with_suffix(".npy")

    def _open(self):
        if self.embeddings is not None:
            return

        if self.npy_path.exists():
            self.embeddings = np.load(self.npy_path, mmap_mode="r")
            self.backend = "npy-memmap"
        else:
            self.h5 = h5py.File(str(self.embeddings_path), "r")
            self.embeddings = self.h5["embeddings"]
            self.backend = "hdf5"

        logger.info(
            "Opened embedding store (%s): %d tracks x %d dims",
            self.backend,
            self.shape[0],
            self.dim,
        )

    def __getitem__(self, indices: np.ndarray) -> np.ndarray:
        """Fetch embeddings by row indices.

        Args:
            indices: Array of integer indices into the embedding matrix

        Returns:
            (N, dim) array of embeddings in the same order as indices
        """
        self._open()
        indices = np.asarray(indices, dtype=np.int64)

        if len(indices) == 0:
            return np.empty((0, self.dim), dtype=np.float32)

        if self.backend == "npy-memmap":
            return np.asarray(self.embeddings[indices], dtype=np.float32)

        unique_indices, inverse = np.unique(indices, return_inverse=True)
        fetched = self.embeddings[unique_indices.tolist()]
        return fetched[inverse].astype(np.float32, copy=False)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["h5"] = None
        state["embeddings"] = None
        return state

    def close(self):
        """Close any open HDF5 handle."""
        if self.h5 is not None:
            self.h5.close()
        self.h5 = None
        self.embeddings = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class TasteDataset(Dataset):
    """PyTorch dataset for taste-scoring training.

    Each __getitem__ returns a single user's library split into
    context (known songs with scores) and targets (held-out for prediction).
    Optionally injects sampled out-of-library negatives into the target set
    with score=0; these carry is_positive=False so losses can treat them
    separately if needed.
    """

    def __init__(
        self,
        user_ids: List[int],
        user_libraries_path: Path,
        embedding_store: EmbeddingStore,
        context_ratio: float = 0.7,
        window_size: int = 300,
        num_negatives: int = 0,
        catalog_size: Optional[int] = None,
    ):
        """Initialize dataset.

        Args:
            user_ids: List of user IDs in this split
            user_libraries_path: Path to HDF5 file with user libraries
            embedding_store: EmbeddingStore for fetching embeddings
            context_ratio: Fraction of library used as context (rest are targets)
            window_size: Max songs to sample per training step
            num_negatives: Out-of-library negatives to inject per user per step
                (with score=0, is_positive=False). 0 keeps the previous behavior.
            catalog_size: Total embedding catalog size; required when
                num_negatives > 0. Defaults to embedding_store.shape[0].
        """
        self.user_ids = user_ids
        self.user_libraries_path = Path(user_libraries_path)
        self.user_h5 = None
        self.embeddings = embedding_store
        self.context_ratio = context_ratio
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.catalog_size = catalog_size if catalog_size is not None else embedding_store.shape[0]
        if num_negatives > 0 and self.catalog_size <= 0:
            raise ValueError("catalog_size must be > 0 when num_negatives > 0")

    def _open_user_libraries(self):
        if self.user_h5 is None:
            self.user_h5 = h5py.File(str(self.user_libraries_path), "r")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["user_h5"] = None
        return state

    def close(self):
        if self.user_h5 is not None:
            self.user_h5.close()
        self.user_h5 = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get one user's context/target split.

        Returns dict with:
            context_embeddings: (ctx_len, 1024)
            context_scores: (ctx_len,)
            target_embeddings: (tgt_len, 1024)
            target_scores: (tgt_len,)
            target_is_positive: (tgt_len,) bool — False for sampled negatives
            context_len: int
            target_len: int
        """
        self._open_user_libraries()
        user_id = self.user_ids[idx]
        group = self.user_h5[f"user_{user_id}"]

        track_indices = group["track_indices"][:]
        scores = group["scores"][:]
        # Full library set for negative-sampling rejection (not just the
        # windowed subset — a sampled negative that's elsewhere in the user's
        # library is a label-noise we can cheaply avoid).
        full_lib_set = set(int(x) for x in track_indices.tolist())

        N = len(track_indices)

        # Sample window if library is larger
        if N > self.window_size:
            sample_idx = np.random.choice(N, self.window_size, replace=False)
            track_indices = track_indices[sample_idx]
            scores = scores[sample_idx]
            N = self.window_size

        # Random permute and split into context/target
        perm = np.random.permutation(N)
        split = int(N * self.context_ratio)

        ctx_idx = perm[:split]
        tgt_idx = perm[split:]

        ctx_tracks = track_indices[ctx_idx]
        ctx_scores = scores[ctx_idx]
        tgt_tracks = track_indices[tgt_idx]
        tgt_scores = scores[tgt_idx]

        # Fetch embeddings
        ctx_emb = self.embeddings[ctx_tracks]
        tgt_emb = self.embeddings[tgt_tracks]

        n_positives = len(tgt_idx)

        # Sampled out-of-library negatives. Rejection sampling against the
        # user's full library (~hundreds out of 2.2M, so rejection rate is
        # negligible). Negatives carry score=0 and is_positive=False.
        if self.num_negatives > 0:
            neg_indices = np.empty(self.num_negatives, dtype=np.int64)
            filled = 0
            # Oversample to reduce loop overhead — rejection rate is tiny.
            while filled < self.num_negatives:
                draw = np.random.randint(
                    0, self.catalog_size, size=self.num_negatives - filled + 8
                )
                for v in draw:
                    if int(v) in full_lib_set:
                        continue
                    neg_indices[filled] = v
                    filled += 1
                    if filled == self.num_negatives:
                        break
            neg_emb = self.embeddings[neg_indices]
            neg_scores = np.zeros(self.num_negatives, dtype=np.float32)

            tgt_emb = np.concatenate([tgt_emb, neg_emb], axis=0)
            tgt_scores = np.concatenate([tgt_scores, neg_scores], axis=0)
            tgt_is_positive = np.concatenate([
                np.ones(n_positives, dtype=np.bool_),
                np.zeros(self.num_negatives, dtype=np.bool_),
            ], axis=0)
        else:
            tgt_is_positive = np.ones(n_positives, dtype=np.bool_)

        return {
            "context_embeddings": ctx_emb,
            "context_scores": ctx_scores,
            "target_embeddings": tgt_emb,
            "target_scores": tgt_scores,
            "target_is_positive": tgt_is_positive,
            "context_len": len(ctx_idx),
            "target_len": len(tgt_scores),
        }


def taste_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate variable-length samples into padded batch.

    Args:
        batch: List of dicts from TasteDataset.__getitem__

    Returns:
        Dict with padded tensors:
            context_embeddings: (B, max_ctx, 1024)
            context_scores: (B, max_ctx, 1)
            context_mask: (B, max_ctx) - True where padded
            target_embeddings: (B, max_tgt, 1024)
            target_scores: (B, max_tgt, 1)
            target_mask: (B, max_tgt) - True where padded
            target_is_positive: (B, max_tgt) - True for real held-out
                library tracks, False for sampled negatives or padding
    """
    max_ctx = max(b["context_len"] for b in batch)
    max_tgt = max(b["target_len"] for b in batch)
    B = len(batch)
    dim = batch[0]["context_embeddings"].shape[1]

    # Pre-allocate padded tensors
    ctx_emb = torch.zeros(B, max_ctx, dim)
    ctx_scores = torch.zeros(B, max_ctx, 1)
    ctx_mask = torch.ones(B, max_ctx, dtype=torch.bool)  # True = padding

    tgt_emb = torch.zeros(B, max_tgt, dim)
    tgt_scores = torch.zeros(B, max_tgt, 1)
    tgt_mask = torch.ones(B, max_tgt, dtype=torch.bool)
    tgt_is_positive = torch.zeros(B, max_tgt, dtype=torch.bool)

    for i, b in enumerate(batch):
        c_len = b["context_len"]
        t_len = b["target_len"]

        ctx_emb[i, :c_len] = torch.from_numpy(b["context_embeddings"])
        ctx_scores[i, :c_len, 0] = torch.from_numpy(b["context_scores"])
        ctx_mask[i, :c_len] = False  # Not padding

        tgt_emb[i, :t_len] = torch.from_numpy(b["target_embeddings"])
        tgt_scores[i, :t_len, 0] = torch.from_numpy(b["target_scores"])
        tgt_mask[i, :t_len] = False
        tgt_is_positive[i, :t_len] = torch.from_numpy(b["target_is_positive"])

    return {
        "context_embeddings": ctx_emb,
        "context_scores": ctx_scores,
        "context_mask": ctx_mask,
        "target_embeddings": tgt_emb,
        "target_scores": tgt_scores,
        "target_mask": tgt_mask,
        "target_is_positive": tgt_is_positive,
    }


def get_dataloaders(
    training_dir: Optional[Path] = None,
    embeddings_path: Optional[Path] = None,
    batch_size: int = 32,
    context_ratio: float = 0.7,
    window_size: int = 300,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    prefetch_factor: Optional[int] = None,
    num_negatives: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders.

    Args:
        training_dir: Path to training data directory (default: from config)
        embeddings_path: Path to embeddings.h5 (default: from config)
        batch_size: Samples per batch
        context_ratio: Fraction of library as context
        window_size: Max songs per sample
        num_workers: DataLoader workers (0 = main process)
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: Batches each worker preloads when num_workers > 0
        num_negatives: Out-of-library negatives to inject per user per step.
            Applied identically to train/val/test so cross-variant metrics
            are comparable.

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config import (
        TRAINING_DATA_DIR,
        EMBEDDINGS_PATH,
        TRAIN_NUM_WORKERS,
        TRAIN_PREFETCH_FACTOR,
    )

    if training_dir is None:
        training_dir = TRAINING_DATA_DIR
    if embeddings_path is None:
        embeddings_path = EMBEDDINGS_PATH
    if num_workers is None:
        num_workers = TRAIN_NUM_WORKERS
    if prefetch_factor is None:
        prefetch_factor = TRAIN_PREFETCH_FACTOR

    user_libraries_path = training_dir / "user_libraries.h5"

    # Load user splits
    with open(training_dir / "user_splits.json") as f:
        splits = json.load(f)

    # Open shared resources. HDF5 handles are opened lazily inside each worker.
    embedding_store = EmbeddingStore(embeddings_path)

    # Filter splits to users actually in the HDF5
    with h5py.File(str(user_libraries_path), "r") as user_libraries_h5:
        available_users = {
            int(k.split("_")[1]) for k in user_libraries_h5.keys() if k.startswith("user_")
        }

    def filter_available(user_list):
        return [u for u in user_list if u in available_users]

    train_ids = filter_available(splits["train"])
    val_ids = filter_available(splits["val"])
    test_ids = filter_available(splits["test"])

    logger.info(
        "DataLoader users: %d train, %d val, %d test",
        len(train_ids),
        len(val_ids),
        len(test_ids),
    )

    # Create datasets
    catalog_size = embedding_store.shape[0]
    train_dataset = TasteDataset(
        train_ids, user_libraries_path, embedding_store, context_ratio,
        window_size, num_negatives=num_negatives, catalog_size=catalog_size,
    )
    val_dataset = TasteDataset(
        val_ids, user_libraries_path, embedding_store, context_ratio,
        window_size, num_negatives=num_negatives, catalog_size=catalog_size,
    )
    test_dataset = TasteDataset(
        test_ids, user_libraries_path, embedding_store, context_ratio,
        window_size, num_negatives=num_negatives, catalog_size=catalog_size,
    )

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": taste_collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader
