"""Preprocessing utilities for training data preparation."""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import h5py

logger = logging.getLogger(__name__)


def compute_log_normalized_scores(play_counts: np.ndarray) -> np.ndarray:
    """Convert play counts to [0, 1] scores using log normalization.

    Formula: log(1 + play_count) / max(log(1 + plays))

    Args:
        play_counts: Array of integer play counts

    Returns:
        Array of float32 scores in [0, 1]
    """
    log_counts = np.log1p(play_counts.astype(np.float32))
    max_log = log_counts.max()
    if max_log > 0:
        return (log_counts / max_log).astype(np.float32)
    return np.zeros_like(play_counts, dtype=np.float32)


def _rank_percentile_scores(play_counts: np.ndarray) -> np.ndarray:
    """Per-user rank-percentile in [0, 1]. Highest play_count -> 1.0, lowest -> 0.0."""
    n = len(play_counts)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if n == 1:
        return np.ones(1, dtype=np.float32)
    order = np.argsort(play_counts.astype(np.float64))
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32)
    return (ranks / (n - 1)).astype(np.float32)


def _binary_top_quartile_scores(play_counts: np.ndarray) -> np.ndarray:
    """1.0 if track is in user's top 25% by play_count, else 0.0."""
    if len(play_counts) == 0:
        return np.zeros(0, dtype=np.float32)
    threshold = np.quantile(play_counts.astype(np.float64), 0.75)
    return (play_counts.astype(np.float64) >= threshold).astype(np.float32)


def _z_score_per_user_scores(play_counts: np.ndarray) -> np.ndarray:
    """z-score of log(1+play_count), clipped to [-3, 3], min-max to [0, 1]."""
    if len(play_counts) == 0:
        return np.zeros(0, dtype=np.float32)
    log_counts = np.log1p(play_counts.astype(np.float32))
    mean = log_counts.mean()
    std = log_counts.std()
    if std < 1e-6:
        return np.full_like(log_counts, 0.5, dtype=np.float32)
    z = (log_counts - mean) / std
    z_clipped = np.clip(z, -3.0, 3.0)
    return ((z_clipped + 3.0) / 6.0).astype(np.float32)


SCORE_METHODS = {
    "log_norm": compute_log_normalized_scores,
    "rank_percentile": _rank_percentile_scores,
    "binary_top_quartile": _binary_top_quartile_scores,
    "z_score": _z_score_per_user_scores,
}


def compute_scores(play_counts: np.ndarray, method: str = "log_norm") -> np.ndarray:
    """Convert play counts to [0, 1] scores using the chosen method.

    Args:
        play_counts: Array of integer play counts
        method: One of SCORE_METHODS keys

    Returns:
        Array of float32 scores
    """
    if method not in SCORE_METHODS:
        raise ValueError(f"Unknown score method: {method!r}. Valid: {list(SCORE_METHODS)}")
    return SCORE_METHODS[method](play_counts)


def apply_track_filter(
    track_indices: np.ndarray,
    play_counts: np.ndarray,
    method: str = "all",
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter a user's tracks by play_count rule.

    Args:
        track_indices: Array of embedding row indices
        play_counts: Parallel array of play counts
        method: One of "all" | "drop_one_play" | "min_play_3" | "top_100"

    Returns:
        (filtered_indices, filtered_play_counts)
    """
    if method == "all":
        return track_indices, play_counts
    if method == "drop_one_play":
        mask = play_counts > 1
        return track_indices[mask], play_counts[mask]
    if method == "min_play_3":
        mask = play_counts >= 3
        return track_indices[mask], play_counts[mask]
    if method == "top_100":
        if len(play_counts) <= 100:
            return track_indices, play_counts
        order = np.argsort(-play_counts.astype(np.int64), kind="stable")[:100]
        order_sorted = np.sort(order)
        return track_indices[order_sorted], play_counts[order_sorted]
    raise ValueError(f"Unknown track filter method: {method!r}")


def filter_diverse_users(
    conn,
    min_genres: int = 3,
    max_top5_conc: float = 0.60,
    min_library_size: int = 50,
) -> List[int]:
    """Return user_ids passing diversity filter with minimum library size.

    Args:
        conn: SQLite connection
        min_genres: Minimum genre_count required
        max_top5_conc: Maximum top5_concentration allowed
        min_library_size: Minimum tracks in user's library

    Returns:
        List of user_ids passing all filters
    """
    return filter_users_sql(
        conn,
        where_body="",
        params=(),
        min_genres=min_genres,
        max_top5_conc=max_top5_conc,
        min_library_size=min_library_size,
    )


def filter_users_sql(
    conn,
    where_body: str = "",
    params: Tuple = (),
    min_genres: int = 3,
    max_top5_conc: float = 0.60,
    min_library_size: int = 50,
) -> List[int]:
    """Generalized user filter. `where_body` is appended via AND to the diversity criteria.

    Diversity gate semantics: enforced where stats are available (original cohort, where
    `diversity_score IS NOT NULL`). Snowball users (cohort != 'original') were filtered
    for overlap at discovery time and lack diversity stats — they're admitted regardless
    of `min_genres` / `max_top5_conc`. Variant-specific `where_body` still applies on top.

    The `where_body` is interpolated directly into SQL — only pass code-defined strings,
    never user input. Use `params` for any value substitutions.

    Args:
        conn: SQLite connection
        where_body: Optional SQL WHERE body (no leading "AND"), e.g. "u.cohort = 'core-breadth'"
        params: Positional parameters referenced by ? placeholders in where_body
        min_genres: Minimum genre_count required (only enforced where diversity_score IS NOT NULL)
        max_top5_conc: Maximum top5_concentration allowed (only enforced where stats exist)
        min_library_size: Minimum raw library size (pre track-level filter)

    Returns:
        List of user_ids passing all filters
    """
    base_where = (
        "u.status = 'collected' "
        "AND ("
        "  (u.diversity_score IS NOT NULL "
        "       AND u.genre_count >= ? "
        "       AND u.top5_concentration < ?) "
        "  OR u.diversity_score IS NULL"
        ") "
        "AND (SELECT COUNT(*) FROM user_tracks ut WHERE ut.user_id = u.id) >= ?"
    )
    where = base_where
    extra_params: Tuple = ()
    if where_body and where_body.strip():
        where = base_where + " AND (" + where_body + ")"
        extra_params = tuple(params)

    query = f"SELECT u.id FROM users u WHERE {where}"
    rows = conn.execute(
        query, (min_genres, max_top5_conc, min_library_size) + extra_params
    ).fetchall()
    return [r[0] for r in rows]


def assign_split_deterministic(user_id: int) -> str:
    """Assign a user to train/val/test based on `user_id % 10` (8/1/1 ratio).

    Deterministic — any user that appears in two dataset variants always lands
    in the same split, so cross-variant Spearman comparisons stay aligned.
    """
    h = user_id % 10
    if h < 8:
        return "train"
    if h == 8:
        return "val"
    return "test"


def build_embedding_index(h5_path: Path) -> Dict[str, int]:
    """Build lastfm_id -> row index mapping from embeddings HDF5.

    Args:
        h5_path: Path to embeddings.h5

    Returns:
        Dict mapping lastfm_id strings to integer row indices
    """
    logger.info("Building embedding index from %s", h5_path)
    with h5py.File(str(h5_path), "r") as f:
        track_ids = f["track_ids"].asstr()[:]

    index = {tid: i for i, tid in enumerate(track_ids)}
    logger.info("Built index with %d tracks", len(index))
    return index


def get_user_library(conn, user_id: int) -> List[Tuple[str, int]]:
    """Get a user's library as (lastfm_id, play_count) pairs.

    Args:
        conn: SQLite connection
        user_id: User's database ID

    Returns:
        List of (lastfm_id, play_count) tuples
    """
    rows = conn.execute(
        """
        SELECT ut.lastfm_id, ut.play_count
        FROM user_tracks ut
        WHERE ut.user_id = ?
        """,
        (user_id,),
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def get_user_library_with_embeddings(
    conn,
    user_id: int,
    embedding_index: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Get a user's library filtered to tracks with embeddings.

    Args:
        conn: SQLite connection
        user_id: User's database ID
        embedding_index: lastfm_id -> row index mapping

    Returns:
        Tuple of (track_indices, play_counts) as numpy arrays
    """
    library = get_user_library(conn, user_id)

    # Filter to tracks that have embeddings
    indices = []
    play_counts = []
    for lastfm_id, play_count in library:
        if lastfm_id in embedding_index:
            indices.append(embedding_index[lastfm_id])
            play_counts.append(play_count)

    return (
        np.array(indices, dtype=np.int32),
        np.array(play_counts, dtype=np.int32),
    )


def save_embedding_index(index: Dict[str, int], path: Path) -> None:
    """Save embedding index to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(index, f)
    logger.info("Saved embedding index to %s", path)


def load_embedding_index(path: Path) -> Dict[str, int]:
    """Load embedding index from JSON file."""
    with open(path) as f:
        return json.load(f)


# ============================================================================
# DatasetVariant — a single dataset experiment definition
# ============================================================================


@dataclass
class DatasetVariant:
    """A single dataset configuration for an experiment.

    Each variant operationalizes one testable theory about what training data
    produces the best recommender. Hyperparameters are held fixed across variants
    so the dataset is the only varying axis.
    """

    name: str
    description: str
    user_filter_sql: str = ""
    user_filter_params: Tuple = field(default_factory=tuple)
    track_filter: str = "all"           # all | drop_one_play | min_play_3 | top_100
    score_method: str = "log_norm"      # log_norm | rank_percentile | binary_top_quartile | z_score
    min_library_size_after_filter: int = 50
    min_genres: int = 3                 # diversity-filter override
    max_top5_conc: float = 0.60         # diversity-filter override
    min_library_size: int = 50          # raw library size pre-track-filter
    # --- Loss / target axis (orthogonal to dataset axis above) -------------
    num_negatives: int = 0              # out-of-library negatives per user per step
    model_type: str = "scoring"         # scoring | hurdle
    hurdle_alpha: float = 1.0           # BCE + alpha * MSE weighting (hurdle only)

    def to_serializable(self) -> Dict:
        d = asdict(self)
        d["user_filter_params"] = list(self.user_filter_params)
        return d


def prepare_dataset_variant(
    variant: DatasetVariant,
    training_dir: Path,
    embedding_index: Dict[str, int],
    conn,
    progress_log_interval: int = 500,
) -> Dict:
    """Prepare per-variant training data. Idempotent — skips if user_libraries.h5 already exists.

    Writes:
        {training_dir}/user_libraries.h5
        {training_dir}/user_splits.json
        {training_dir}/dataset_config.json

    Args:
        variant: The DatasetVariant config
        training_dir: Output directory (created if needed)
        embedding_index: lastfm_id -> embedding row index
        conn: SQLite connection
        progress_log_interval: Log every N users

    Returns:
        Dict with status, user_count, and any skipped counts
    """
    training_dir.mkdir(parents=True, exist_ok=True)
    libraries_path = training_dir / "user_libraries.h5"
    splits_path = training_dir / "user_splits.json"
    config_path = training_dir / "dataset_config.json"

    # Always (re)snapshot the variant config for traceability
    with config_path.open("w") as f:
        json.dump(variant.to_serializable(), f, indent=2)

    # Resume: skip if h5 already complete
    if libraries_path.exists() and splits_path.exists():
        with h5py.File(str(libraries_path), "r") as h5:
            user_count = sum(1 for k in h5.keys() if k.startswith("user_"))
        logger.info(
            "Variant '%s': existing data found (%d users), skipping prep",
            variant.name, user_count,
        )
        return {"status": "skipped", "user_count": user_count}

    # Step 1 — user filter
    user_ids = filter_users_sql(
        conn,
        where_body=variant.user_filter_sql,
        params=variant.user_filter_params,
        min_genres=variant.min_genres,
        max_top5_conc=variant.max_top5_conc,
        min_library_size=variant.min_library_size,
    )
    logger.info(
        "Variant '%s': %d users passed user filter",
        variant.name, len(user_ids),
    )

    # Step 2 — deterministic split
    splits: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    for uid in user_ids:
        splits[assign_split_deterministic(uid)].append(uid)
    with splits_path.open("w") as f:
        json.dump(splits, f)
    logger.info(
        "Variant '%s': split %d/%d/%d (train/val/test)",
        variant.name, len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )

    # Step 3 — process each user's library
    h5 = h5py.File(str(libraries_path), "a")
    skipped_too_small = 0
    processed = 0

    try:
        for i, user_id in enumerate(user_ids):
            track_indices, play_counts = get_user_library_with_embeddings(
                conn, user_id, embedding_index
            )

            # Track-level filter (drop low-play tracks, keep top-N, etc.)
            track_indices, play_counts = apply_track_filter(
                track_indices, play_counts, method=variant.track_filter
            )

            if len(track_indices) < variant.min_library_size_after_filter:
                skipped_too_small += 1
                continue

            scores = compute_scores(play_counts, method=variant.score_method)

            group = h5.create_group(f"user_{user_id}")
            group.create_dataset("track_indices", data=track_indices)
            group.create_dataset("scores", data=scores)
            processed += 1

            if (i + 1) % progress_log_interval == 0:
                logger.info(
                    "Variant '%s': processed %d/%d users (%.1f%%)",
                    variant.name, i + 1, len(user_ids),
                    100.0 * (i + 1) / max(1, len(user_ids)),
                )
                h5.flush()
    finally:
        h5.close()

    logger.info(
        "Variant '%s' done: %d users in HDF5, %d skipped (post-filter library too small)",
        variant.name, processed, skipped_too_small,
    )
    return {
        "status": "complete",
        "user_count": processed,
        "skipped_too_small": skipped_too_small,
        "user_filter_count": len(user_ids),
    }
