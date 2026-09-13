"""Scan the repo's data/ tree and return a unified list of trained models
the harness can run inference with. Each record carries a stable id, the
hyperparameters used (so the UI can render them), and the .pt path.

Sources:
- data/best_model.pt                     -> "production"
- data/sweeps/<sweep>/trial_*/best.pt    -> "experimental" (metadata from summary.json)

Training-checkpoint files at data/checkpoints/checkpoint_epoch_*.pt are
intentionally skipped — they're intermediate saves from a single training run
and would add noise without adding distinct behaviors to compare.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from config import DATA_DIR

logger = logging.getLogger(__name__)

ARCH_KEYS = ("d_model", "num_heads", "enc_layers", "dec_layers", "dropout", "dim_feedforward")
_STATEDICT_CACHE: Dict[tuple, Dict[str, Any]] = {}


@dataclass
class ModelRecord:
    id: str
    name: str
    family: str            # display family ("Sweep trial", etc.)
    family_key: str        # machine key for filtering ("dataset_experiment", "hyperparam_trial", "production", "other")
    trained: str           # YYYY-MM-DD
    status: str            # "production" | "experimental" | "candidate" | "baseline"
    path: str              # absolute path to .pt
    hyperparams: Dict[str, Any]
    description: str
    theory_tag: str        # short, human label ("Top 100 per user")
    score_method: str      # "log_norm" | "rank_percentile" | "binary_top_quartile" | "z_score"
    model_type: str = "scoring"          # "scoring" | "hurdle"
    num_negatives: int = 0               # K negatives per user per step (training-time)
    hurdle_alpha: Optional[float] = None # BCE + alpha*MSE weighting (hurdle only)
    sweep_name: Optional[str] = None
    val_spearman: Optional[float] = None
    test_spearman: Optional[float] = None
    val_auc: Optional[float] = None
    test_auc: Optional[float] = None
    test_ndcg50: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _theory_tag(name: str) -> str:
    """Turn 'drop_one_play_tracks' into 'Drop 1-play tracks'.
    A small lookup table covers the cases where stripping underscores reads
    awkwardly; everything else just title-cases the snake_case name."""
    overrides = {
        "baseline_all_diverse":          "Baseline · all diverse",
        "binary_top_quartile":           "Binary top quartile",
        "diverse_engaged_clean":         "Diverse + engaged",
        "diverse_listeners_strict":      "Diverse listeners (strict)",
        "drop_low_play_tracks":          "Drop low-play tracks",
        "drop_one_play_tracks":          "Drop 1-play tracks",
        "engaged_users":                 "Engaged users only",
        "high_overlap_broad":            "High overlap (broad)",
        "high_overlap_strict":           "High overlap (strict)",
        "me_centric_maximal":            "Me-centric (maximal)",
        "me_overlap_clean":              "Me-overlap (clean)",
        "niche_deep_only":               "Niche-deep only",
        "non_original_snowball_only":    "Snowball only",
        "original_17k_only":             "Original 17K only",
        "rank_percentile_scores":        "Rank-percentile scores",
        "top_100_per_user":              "Top 100 per user",
        "z_score_per_user":              "Z-score per user",
        "neg_0":                         "No negatives (baseline)",
        "neg_30":                        "30 negatives · 1:3",
        "neg_90":                        "90 negatives · 1:1",
        "neg_270":                       "270 negatives · 3:1",
        "neg_900":                       "900 negatives · 10:1",
        "hurdle_neg_270":                "Hurdle (270 neg)",
    }
    if name in overrides:
        return overrides[name]
    return name.replace("_", " ").strip().capitalize()


def _load_config_from_pt(path: Path) -> Dict[str, Any]:
    """Extract the `model_config` dict from a checkpoint without keeping state in memory."""
    key = (str(path), path.stat().st_mtime_ns)
    cached = _STATEDICT_CACHE.get(key)
    if cached is not None:
        return cached
    # weights_only=False because checkpoints contain plain python dicts;
    # we trust local files.
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(blob.get("model_config") or {})
    _STATEDICT_CACHE[key] = cfg
    return cfg


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")


def _scan_best_model() -> Optional[ModelRecord]:
    p = Path(DATA_DIR) / "best_model.pt"
    if not p.exists():
        return None
    try:
        cfg = _load_config_from_pt(p)
    except Exception as exc:
        logger.warning("best_model.pt unreadable: %s", exc)
        cfg = {}
    return ModelRecord(
        id="best",
        name="Best Model",
        family="Taste-scoring NN",
        family_key="production",
        trained=_mtime_iso(p),
        status="production",
        path=str(p),
        hyperparams=cfg,
        description="Lowest validation loss from the most recent training run.",
        theory_tag="Best model",
        score_method="log_norm",
        model_type=cfg.get("model_type", "scoring"),
    )


def _scan_sweep_trials() -> List[ModelRecord]:
    sweeps_root = Path(DATA_DIR) / "sweeps"
    if not sweeps_root.exists():
        return []
    out: List[ModelRecord] = []
    for sweep_dir in sorted(p for p in sweeps_root.iterdir() if p.is_dir()):
        for trial_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
            weights = trial_dir / "best.pt"
            summary = trial_dir / "summary.json"
            if not weights.exists():
                continue
            meta: Dict[str, Any] = {}
            if summary.exists():
                try:
                    meta = json.loads(summary.read_text())
                except Exception as exc:
                    logger.warning("unreadable summary at %s: %s", summary, exc)
            cfg_full = meta.get("config") or {}
            hparams = {k: cfg_full.get(k) for k in cfg_full}  # preserve order
            trial_name = meta.get("name") or trial_dir.name
            sweep_name = sweep_dir.name
            ds_cfg = cfg_full.get("dataset") or {}
            score_method = ds_cfg.get("score_method") or "log_norm"
            model_type = ds_cfg.get("model_type") or "scoring"
            num_negatives = int(ds_cfg.get("num_negatives") or 0)
            hurdle_alpha = _maybe_float(ds_cfg.get("hurdle_alpha"))

            is_dataset_exp = (
                sweep_name.startswith("dataset_experiments")
                and "partial" not in sweep_name
            )
            is_neg_exp = sweep_name.startswith("negative_experiments")
            is_smoke = "smoke" in sweep_name
            if is_dataset_exp:
                family = "Dataset experiment"
                family_key = "dataset_experiment"
                # Strip the sweep prefix; the trial name alone identifies the theory.
                display = trial_name
            elif is_neg_exp:
                family = "Negative-sampling sweep"
                family_key = "negative_experiment"
                display = trial_name
            elif is_smoke:
                family = "Smoke test"
                family_key = "other"
                display = f"{sweep_name} · {trial_name}"
            else:
                family = "Hyperparam trial"
                family_key = "hyperparam_trial"
                display = f"{sweep_name} · {trial_name}"

            model_id = f"sweep:{sweep_name}:{trial_dir.name}"
            description = (
                meta.get("description")
                or _describe_trial(trial_name, meta)
            )
            out.append(ModelRecord(
                id=model_id,
                name=display,
                family=family,
                family_key=family_key,
                trained=_mtime_iso(weights),
                status="experimental",
                path=str(weights),
                hyperparams=hparams,
                description=description,
                theory_tag=_theory_tag(trial_name),
                score_method=score_method,
                model_type=model_type,
                num_negatives=num_negatives,
                hurdle_alpha=hurdle_alpha,
                sweep_name=sweep_name,
                val_spearman=_maybe_float(meta.get("best_val_spearman")),
                test_spearman=_maybe_float(meta.get("test_spearman")),
                val_auc=_maybe_float(meta.get("best_val_auc")),
                test_auc=_maybe_float(meta.get("test_auc")),
                test_ndcg50=_maybe_float(meta.get("test_ndcg50")),
            ))
    return out


def _describe_trial(name: str, meta: Dict[str, Any]) -> str:
    bits: List[str] = []
    if "best_val_spearman" in meta:
        bits.append(f"val spearman {meta['best_val_spearman']:.3f}")
    if "test_spearman" in meta:
        bits.append(f"test spearman {meta['test_spearman']:.3f}")
    if "epochs_run" in meta:
        bits.append(f"{meta['epochs_run']} epochs")
    suffix = " · ".join(bits) if bits else "hyperparameter sweep trial"
    return f"{name} — {suffix}"


def _maybe_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def scan_models() -> List[ModelRecord]:
    out: List[ModelRecord] = []
    best = _scan_best_model()
    if best is not None:
        out.append(best)
    out.extend(_scan_sweep_trials())
    return out


def get_model(model_id: str) -> Optional[ModelRecord]:
    for m in scan_models():
        if m.id == model_id:
            return m
    return None


def arch_kwargs(record: ModelRecord) -> Dict[str, Any]:
    """Subset of hyperparams accepted by library.src.model.create_model."""
    cfg = record.hyperparams
    # Prefer a model_config key if present (saved by save_best_model), else
    # pick ARCH_KEYS from the flat hyperparams dict (sweep summary.json shape).
    if all(k in cfg for k in ARCH_KEYS):
        out = {k: cfg[k] for k in ARCH_KEYS}
    else:
        out = {k: cfg[k] for k in ARCH_KEYS if k in cfg}
    out["model_type"] = record.model_type
    return out
