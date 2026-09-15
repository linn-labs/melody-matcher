"""Load a checkpoint and score the full 2.2M-track embedding catalog against
a user's library. Caches loaded models + encoded-library tensors across calls
so subsequent runs of the same (library, model) are near-instant (only the
candidate pass must run, and even then results are already cached).
"""

from __future__ import annotations

import heapq
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

from config import EMBEDDINGS_PATH
from library.src.model import create_model
from library.src.preprocessing import compute_scores

from . import db
from .models_registry import ModelRecord, arch_kwargs, get_model, scan_models

logger = logging.getLogger(__name__)

CANDIDATE_BATCH = 4096

VALID_RANKING_MODES = ("p_played", "joint", "threshold")

_MODEL_CACHE: Dict[Tuple[str, int], torch.nn.Module] = {}
_ENCODED_LIB_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}
# Cache key includes ranking_mode + quantized threshold so swapping rankings
# on a hurdle model doesn't return stale results.
_RUN_CACHE: Dict[Tuple[int, str, str, int], Dict[str, Any]] = {}
_H5_HANDLE: Optional[h5py.File] = None


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _h5() -> h5py.File:
    global _H5_HANDLE
    if _H5_HANDLE is None:
        _H5_HANDLE = h5py.File(str(EMBEDDINGS_PATH), "r")
    return _H5_HANDLE


_ARCH_PASSTHROUGH = {
    "embed_dim", "d_model", "num_heads",
    "enc_layers", "dec_layers", "dropout",
    "dim_feedforward", "model_type",
}


def _load_model(record: ModelRecord) -> torch.nn.Module:
    p = Path(record.path)
    key = (str(p), p.stat().st_mtime_ns)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    blob = torch.load(p, map_location="cpu", weights_only=False)
    cfg = blob.get("model_config") or arch_kwargs(record)
    if not cfg:
        raise RuntimeError(f"no model_config for {record.id}; refusing to guess architecture")
    kwargs = {k: cfg[k] for k in cfg if k in _ARCH_PASSTHROUGH}
    # Old checkpoints (pre-hurdle refactor) lack model_type; fall back to the
    # record value (which defaults to "scoring").
    kwargs.setdefault("model_type", record.model_type)
    model = create_model(**kwargs)
    model.load_state_dict(blob["model_state_dict"])
    model = model.to(_device())
    model.eval()
    _MODEL_CACHE[key] = model
    return model


def _fetch_library_tensors(library_id: int) -> Tuple[np.ndarray, np.ndarray]:
    with db.get_connection() as conn:
        rows = conn.execute(
            """
            SELECT embedding_index, play_count FROM harness_library_tracks
            WHERE library_id = ? AND embedding_index IS NOT NULL
            """,
            (library_id,),
        ).fetchall()
    if not rows:
        raise ValueError(f"library {library_id} has no matched tracks")
    raw_idx = np.array([r["embedding_index"] for r in rows], dtype=np.int64)
    raw_plays = np.array([r["play_count"] or 0 for r in rows], dtype=np.int64)
    # Multiple library tracks may have fuzzy-matched to the same embedding row
    # (variants, remasters). h5py fancy indexing requires strictly increasing
    # indices, so dedupe — and sum play counts across dupes so the user's
    # total listening to that track is preserved.
    indices, inverse = np.unique(raw_idx, return_inverse=True)
    plays = np.zeros(indices.shape[0], dtype=np.int64)
    np.add.at(plays, inverse, raw_plays)
    embs = _h5()["embeddings"][indices]  # (N_unique, 1024)
    return embs.astype(np.float32), plays


def _encode_library(
    model: torch.nn.Module,
    library_id: int,
    model_id: str,
    score_method: str = "log_norm",
) -> Tuple[torch.Tensor, np.ndarray]:
    """Returns (encoded_library on device, numpy array of library embedding indices).

    `score_method` must match the score distribution the model was trained on
    (read from the trial's summary.json). Mismatching distributions silently
    produces nonsense recommendations — see registry.ModelRecord.score_method.
    """
    with db.get_connection() as conn:
        idx_rows = conn.execute(
            "SELECT embedding_index FROM harness_library_tracks "
            "WHERE library_id = ? AND embedding_index IS NOT NULL",
            (library_id,),
        ).fetchall()
    lib_indices = np.sort(np.array([r["embedding_index"] for r in idx_rows], dtype=np.int64))

    key = (library_id, model_id)
    enc = _ENCODED_LIB_CACHE.get(key)
    if enc is not None:
        return enc, lib_indices

    embs, plays = _fetch_library_tensors(library_id)
    scores_np = compute_scores(plays, method=score_method)
    lib_embs = torch.from_numpy(embs).unsqueeze(0).to(_device())         # (1, N, 1024)
    lib_scores = torch.from_numpy(scores_np).unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
    lib_scores = lib_scores.to(_device())
    with torch.no_grad():
        enc = model.encode_library(lib_embs, lib_scores, padding_mask=None)
    _ENCODED_LIB_CACHE[key] = enc
    return enc, lib_indices


def _score_candidates(
    model: torch.nn.Module,
    encoded_lib: torch.Tensor,
    exclude_set: set,
    top_k: int,
    ranking_mode: str = "p_played",
    threshold: float = 0.5,
) -> List[Tuple[float, int, float, float]]:
    """Iterate the full embedding catalog in batches, keep top-k by ranking_score.

    Returns a list of (ranking_score, idx, p_played, score_pred) tuples sorted
    by ranking_score desc. For non-hurdle (scoring) models, p_played and
    score_pred are NaN; for hurdle models, both are populated.

    ranking_mode (hurdle-only; ignored for scoring models):
        "p_played"  - rank by sigmoid(play_logits)
        "joint"     - rank by sigmoid(play_logits) * score_pred
        "threshold" - rank by score_pred among candidates with P(played) > threshold;
                      others get -inf so they never enter the top-k.
    """
    f = _h5()
    total = f["embeddings"].shape[0]
    device = _device()
    is_hurdle = hasattr(model, "play_head") and hasattr(model, "score_head") \
                and hasattr(model, "candidate_backbone")
    # min-heap on ranking_score; idx is the unique tiebreaker so the 3rd/4th
    # fields never get compared.
    heap: List[Tuple[float, int, float, float]] = []
    NEG_INF = float("-inf")
    NAN = float("nan")

    for start in range(0, total, CANDIDATE_BATCH):
        end = min(start + CANDIDATE_BATCH, total)
        batch_np = f["embeddings"][start:end].astype(np.float32)
        batch = torch.from_numpy(batch_np).unsqueeze(0).to(device)    # (1, B, 1024)
        with torch.no_grad():
            out = model.score_candidates(batch, encoded_lib)

        if is_hurdle:
            p_played = torch.sigmoid(out["play_logits"]).squeeze(0).squeeze(-1)
            score_pred = out["score_pred"].squeeze(0).squeeze(-1)
            if ranking_mode == "p_played":
                rank_t = p_played
            elif ranking_mode == "joint":
                rank_t = p_played * score_pred
            elif ranking_mode == "threshold":
                gated = torch.where(
                    p_played > threshold,
                    score_pred,
                    torch.full_like(score_pred, NEG_INF),
                )
                rank_t = gated
            else:
                raise ValueError(
                    f"unknown ranking_mode {ranking_mode!r}; "
                    f"expected one of {VALID_RANKING_MODES}"
                )
            rank_np = rank_t.float().cpu().numpy()
            p_np = p_played.float().cpu().numpy()
            sp_np = score_pred.float().cpu().numpy()
            for i in range(end - start):
                idx = start + i
                if idx in exclude_set:
                    continue
                s = float(rank_np[i])
                if s == NEG_INF:
                    continue
                entry = (s, idx, float(p_np[i]), float(sp_np[i]))
                if len(heap) < top_k:
                    heapq.heappush(heap, entry)
                elif s > heap[0][0]:
                    heapq.heapreplace(heap, entry)
        else:
            preds = out.squeeze(0).squeeze(-1).float().cpu().numpy()    # (B,)
            for i, s in enumerate(preds):
                idx = start + i
                if idx in exclude_set:
                    continue
                entry = (float(s), idx, NAN, NAN)
                if len(heap) < top_k:
                    heapq.heappush(heap, entry)
                elif s > heap[0][0]:
                    heapq.heapreplace(heap, entry)

    return sorted(heap, key=lambda x: -x[0])


def _resolve_metadata(
    top: List[Tuple[float, int, float, float]],
    library_id: int,
) -> List[Dict[str, Any]]:
    """Attach artist/title/album/year/genre/duration where available.

    `embedding_key` in the catalog is "artist::title". For metadata beyond
    that we look up the user's own library (in case a recommended track
    happens to be in their library — ideally it shouldn't, we excluded them —
    but also for consistent metadata shapes). We fall back to parsing the key.

    For hurdle results, also surface p_played and score_pred on each row so
    the UI can show both signals alongside the ranking score.
    """
    f = _h5()
    track_ids = f["track_ids"]
    idx_to_entry: Dict[int, Tuple[float, float, float]] = {
        idx: (score, p, sp) for (score, idx, p, sp) in top
    }
    sorted_idx = sorted(idx_to_entry)
    keys = track_ids.asstr()[sorted_idx]
    out: List[Dict[str, Any]] = []
    for i, idx in enumerate(sorted_idx):
        key = keys[i]
        artist, _, title = key.partition("::")
        score, p, sp = idx_to_entry[idx]
        row: Dict[str, Any] = {
            "embedding_index": int(idx),
            "embedding_key": key,
            "artist": artist,
            "title": title,
            "score": float(score),
        }
        # Surface hurdle outputs only when populated. NaN flows through here
        # for non-hurdle results and is filtered out so the row stays clean.
        if not np.isnan(p):
            row["p_played"] = float(p)
        if not np.isnan(sp):
            row["score_pred"] = float(sp)
        out.append(row)
    out.sort(key=lambda r: -r["score"])
    for rank, r in enumerate(out, start=1):
        r["rank"] = rank
    return out


def run_inference(
    model_id: str,
    library_id: int,
    top_k: int = 50,
    ranking_mode: str = "p_played",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Score the full catalog against a library; return top-K with metadata.

    For scoring (single-head) models, ranking_mode and threshold are ignored.
    For hurdle models:
        - "p_played" ranks by sigmoid(play_logits)
        - "joint"    ranks by sigmoid(play_logits) * score_pred
        - "threshold" ranks by score_pred among candidates with P(played) > t.
    """
    if ranking_mode not in VALID_RANKING_MODES:
        raise ValueError(
            f"unknown ranking_mode {ranking_mode!r}; expected one of {VALID_RANKING_MODES}"
        )

    record = get_model(model_id)
    if record is None:
        raise ValueError(f"unknown model_id: {model_id}")

    # Scoring models don't depend on ranking_mode/threshold; collapse the
    # cache key so all settings share one cached result.
    if record.model_type == "hurdle":
        # 0.01 resolution on threshold — enough for UI, prevents key blowup.
        thr_q = int(round(threshold * 100))
        cache_key = (library_id, model_id, ranking_mode, thr_q)
    else:
        cache_key = (library_id, model_id, "_", 0)
    cached = _RUN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    t0 = time.time()
    model = _load_model(record)
    encoded_lib, lib_indices = _encode_library(
        model, library_id, model_id, score_method=record.score_method
    )
    exclude = set(int(i) for i in lib_indices.tolist())
    top = _score_candidates(
        model, encoded_lib, exclude, top_k,
        ranking_mode=ranking_mode, threshold=threshold,
    )
    results = _resolve_metadata(top, library_id)
    latency_ms = int((time.time() - t0) * 1000)

    payload = {
        "results": results,
        "latency_ms": latency_ms,
        "ranking_mode": ranking_mode if record.model_type == "hurdle" else None,
        "threshold": threshold if record.model_type == "hurdle" else None,
        "model_type": record.model_type,
    }
    _RUN_CACHE[cache_key] = payload
    return payload


def validate_all_models() -> Dict[str, str]:
    """Cheap pre-flight: confirm every registered checkpoint has the keys
    needed to instantiate a model. Doesn't cache or move to device — we just
    want to know on startup if any of the 17 are broken before the user
    clicks them. Returns {model_id: error_msg} for failures."""
    errors: Dict[str, str] = {}
    arch_required = {"d_model", "num_heads", "enc_layers", "dec_layers",
                     "dropout", "dim_feedforward"}
    for record in scan_models():
        try:
            blob = torch.load(record.path, map_location="cpu", weights_only=False)
            if "model_state_dict" not in blob:
                raise RuntimeError("checkpoint missing 'model_state_dict'")
            cfg = blob.get("model_config") or arch_kwargs(record)
            missing = arch_required - set(cfg or {})
            if missing:
                raise RuntimeError(f"missing arch keys: {sorted(missing)}")
        except Exception as exc:
            errors[record.id] = f"{type(exc).__name__}: {exc}"
            logger.warning("model %s failed validation: %s", record.id, exc)
    if errors:
        logger.warning("%d model(s) failed validation: %s", len(errors), list(errors))
    else:
        logger.info("validated %d model checkpoints OK", len(scan_models()))
    return errors
