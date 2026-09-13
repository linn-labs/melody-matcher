#!/usr/bin/env python3
"""Dataset Experiments — train one model per dataset theory.

Historical explicit-execution research tooling; not a supported workflow.
Read docs/COMPONENTS.md and docs/DATA_AND_RIGHTS.md before considering use.
"""

import argparse
import csv
import gc
import json
import logging
import random
import sys
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    DATA_DIR,
    EMBEDDINGS_PATH,
    TRAINING_DATA_DIR,
    TRAIN_NUM_WORKERS,
    TRAIN_PREFETCH_FACTOR,
    MIN_LIBRARY_SIZE,
    MIN_SUPER_GENRES,
    MAX_TOP5_CONCENTRATION,
)
from src.db import get_connection
from library.src.dataset import get_dataloaders
from library.src.model import create_model
from library.src.preprocessing import (
    DatasetVariant,
    build_embedding_index,
    load_embedding_index,
    prepare_dataset_variant,
    save_embedding_index,
)
from library.scripts.train import (
    get_cosine_schedule_with_warmup,
    get_device,
    save_best_model,
    train_epoch,
    validate,
)


SWEEPS_DIR = DATA_DIR / "sweeps"

logger = logging.getLogger("dataset_experiments")


# ============================================================================
# Fixed training recipe — based on trial_13_epochs_long winner from stage1_20260416,
# with two deltas tuned for the larger snowball corpus + CUDA box:
#   - batch_size 32 → 128 (4× fewer steps/epoch; GPU has VRAM headroom)
#   - patience 8 → 4 (val loss is plateauing in noise after ~25 epochs;
#                     no need to keep training for a harness-comparison sweep)
# All 17 variants use this same recipe, so cross-variant comparisons stay fair.
# ============================================================================

FIXED_RECIPE: Dict = {
    "lr": 1e-4,
    "weight_decay": 0.01,
    "batch_size": 128,
    "warmup_ratio": 0.05,
    "grad_clip": 1.0,
    "epochs": 40,
    "patience": 4,
    "dropout": 0.1,
    "d_model": 512,
    "num_heads": 8,
    "enc_layers": 3,
    "dec_layers": 2,
    "dim_feedforward": 2048,
    "context_ratio": 0.7,
    "window_size": 300,
}


# ============================================================================
# The 17 dataset variants — one per testable theory
# ============================================================================
#
# Run order is deliberate: front-loaded with the most informative comparisons
# (control + clean-signal + overlap) so an interrupted run still yields useful
# theory-level conclusions.
# ============================================================================

DATASET_VARIANTS: List[DatasetVariant] = [
    # --- Group A — User-pool composition ---
    DatasetVariant(
        name="baseline_all_diverse",
        description="Control: all users passing diversity filter, no cohort restriction.",
    ),
    DatasetVariant(
        name="original_17k_only",
        description="Replicates trial-13's training set (cohort='original').",
        user_filter_sql="u.cohort = 'original'",
    ),
    # --- Group B (early) — Track filtering on baseline pool ---
    DatasetVariant(
        name="drop_low_play_tracks",
        description="Drop tracks with play_count < 3 per user. Tests noise-tail removal.",
        track_filter="min_play_3",
    ),
    # --- Group C (early) — Score normalization on baseline pool ---
    DatasetVariant(
        name="rank_percentile_scores",
        description="Per-user rank percentile in place of log-normalized scores.",
        score_method="rank_percentile",
    ),
    # --- Group A (continued) — overlap-based cohorts ---
    DatasetVariant(
        name="high_overlap_broad",
        description="Volume of moderately-aligned users: overlap_track ≥0.20 OR overlap_artist ≥0.30 OR cohort='niche-deep'.",
        user_filter_sql=(
            "(u.overlap_track_pct >= 0.20 OR u.overlap_artist_pct >= 0.30 "
            "OR u.cohort = 'niche-deep')"
        ),
    ),
    DatasetVariant(
        name="high_overlap_strict",
        description="Density-over-volume: cohort='core-breadth' (overlap_track ≥0.50 OR overlap_artist ≥0.65).",
        user_filter_sql="u.cohort = 'core-breadth'",
    ),
    DatasetVariant(
        name="niche_deep_only",
        description="Users seeded from the target library's rank-50–150 artists (cohort='niche-deep'). Genre-internal dimensions.",
        user_filter_sql="u.cohort = 'niche-deep'",
    ),
    DatasetVariant(
        name="diverse_listeners_strict",
        description="Highly diverse listeners (genre_count≥5, play_entropy≥6.0, top5<0.40). Tests dimension-selectivity thesis.",
        user_filter_sql="u.play_entropy >= 6.0",
        min_genres=5,
        max_top5_conc=0.40,
    ),
    DatasetVariant(
        name="engaged_users",
        description="Heavy listeners only (total_scrobbles ≥ 10000). Tests engagement quality.",
        user_filter_sql="u.total_scrobbles >= 10000",
    ),
    DatasetVariant(
        name="non_original_snowball_only",
        description="Excludes the original 17K. Tests whether snowball alone produces a better me-centric model.",
        user_filter_sql="u.cohort != 'original'",
    ),
    # --- Group B (continued) — track filtering ---
    DatasetVariant(
        name="drop_one_play_tracks",
        description="Drop tracks with play_count == 1 per user. Less aggressive than min_play_3.",
        track_filter="drop_one_play",
    ),
    DatasetVariant(
        name="top_100_per_user",
        description="Keep only top 100 tracks per user by play_count.",
        track_filter="top_100",
    ),
    # --- Group C (continued) — score normalization ---
    DatasetVariant(
        name="binary_top_quartile",
        description="Score = 1.0 if track in user's top 25% by play_count, else 0.0.",
        score_method="binary_top_quartile",
    ),
    DatasetVariant(
        name="z_score_per_user",
        description="Per-user z-score of log(1+play_count), clipped and min-max-scaled to [0,1].",
        score_method="z_score",
    ),
    # --- Group D — Stacked combinations ---
    DatasetVariant(
        name="me_overlap_clean",
        description="Stacks A4 + B2: high overlap users + drop play_count<3.",
        user_filter_sql=(
            "(u.overlap_track_pct >= 0.20 OR u.overlap_artist_pct >= 0.30 "
            "OR u.cohort = 'niche-deep')"
        ),
        track_filter="min_play_3",
    ),
    DatasetVariant(
        name="me_centric_maximal",
        description="Aggressive me-centric stack: niche-deep/core-breadth/overlap≥0.30, library≥100, play_count≥3.",
        user_filter_sql=(
            "(u.cohort IN ('core-breadth', 'niche-deep') "
            "OR u.overlap_artist_pct >= 0.30)"
        ),
        track_filter="min_play_3",
        min_library_size=100,
    ),
    DatasetVariant(
        name="diverse_engaged_clean",
        description="High-quality data without overlap bias: genre_count≥5, play_entropy≥6, total_scrobbles≥10K, play_count≥3.",
        user_filter_sql="u.play_entropy >= 6.0 AND u.total_scrobbles >= 10000",
        track_filter="min_play_3",
        min_genres=5,
        max_top5_conc=0.40,
    ),
]


# ============================================================================
# Logging setup
# ============================================================================

def setup_root_logger() -> None:
    """Configure root logger; detach train.py's data/train.log handler if present."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename).name == "train.log":
            root.removeHandler(h)
            h.close()

    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
        root.addHandler(sh)
    root.setLevel(logging.INFO)


def attach_trial_log(trial_dir: Path) -> logging.FileHandler:
    handler = logging.FileHandler(trial_dir / "train.log")
    handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def detach_trial_log(handler: logging.FileHandler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()


# ============================================================================
# Seeding
# ============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Per-variant training
# ============================================================================

def run_variant(
    trial_id: int,
    variant: DatasetVariant,
    sweep_dir: Path,
    embedding_index: Dict[str, int],
    device: torch.device,
    seed: int,
    epoch_override: Optional[int],
    num_workers: int,
    prefetch_factor: int,
) -> Dict:
    """Prepare data, train one model on this variant, save best.pt + summary.json."""
    name = variant.name
    slug = f"exp_{name}"
    trial_dir = sweep_dir / slug
    trial_dir.mkdir(parents=True, exist_ok=True)

    trial_handler = attach_trial_log(trial_dir)
    started_at = datetime.now(timezone.utc)
    wall_start = time.time()

    summary: Dict = {
        "trial_id": trial_id,
        "name": name,
        "description": variant.description,
        "stage": "dataset_experiment",
        "status": "running",
        "config": {**FIXED_RECIPE, "dataset": variant.to_serializable()},
        "seed": seed,
        "device": str(device),
        "started_at": started_at.isoformat(),
        "weights_path": None,
    }

    # Note in summary that Group C (score normalization) is not directly comparable
    # cross-variant via val Spearman — use harness top-50 instead.
    if variant.score_method != "log_norm":
        summary["score_method_note"] = (
            "Score normalization variant — val Spearman is computed against this "
            "variant's own normalized targets and is NOT cross-variant comparable. "
            "Use harness top-50 inspection for cross-variant comparison."
        )

    # Holders so the finally-block can clean up regardless of where we fail
    model = None
    optimizer = None
    scheduler = None
    train_loader = val_loader = test_loader = None
    epoch = -1

    try:
        logger.info("=" * 70)
        logger.info("Variant %d: %s", trial_id, name)
        logger.info("  %s", variant.description)
        logger.info("=" * 70)

        seed_everything(seed)

        # Step 1 — prepare per-variant training data (resumable / idempotent)
        variant_training_dir = TRAINING_DATA_DIR / f"exp_{name}"
        prep_result = prepare_dataset_variant(
            variant=variant,
            training_dir=variant_training_dir,
            embedding_index=embedding_index,
            conn=get_connection(),
        )
        summary["dataset_prep"] = prep_result

        # Step 2 — build per-variant dataloaders (no caching across variants)
        train_loader, val_loader, test_loader = get_dataloaders(
            training_dir=variant_training_dir,
            embeddings_path=EMBEDDINGS_PATH,
            batch_size=FIXED_RECIPE["batch_size"],
            context_ratio=FIXED_RECIPE["context_ratio"],
            window_size=FIXED_RECIPE["window_size"],
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=prefetch_factor,
        )
        logger.info(
            "Variant %d | Train batches: %d | Val batches: %d | Test batches: %d",
            trial_id, len(train_loader), len(val_loader), len(test_loader),
        )

        if len(train_loader) == 0 or len(val_loader) == 0:
            raise RuntimeError(
                f"Variant {name} has insufficient data: "
                f"train_batches={len(train_loader)}, val_batches={len(val_loader)}"
            )

        # Step 3 — model
        model = create_model(
            d_model=FIXED_RECIPE["d_model"],
            num_heads=FIXED_RECIPE["num_heads"],
            enc_layers=FIXED_RECIPE["enc_layers"],
            dec_layers=FIXED_RECIPE["dec_layers"],
            dropout=FIXED_RECIPE["dropout"],
            dim_feedforward=FIXED_RECIPE["dim_feedforward"],
        ).to(device)
        total, trainable = model.count_parameters()
        logger.info("Parameters: %d total, %d trainable", total, trainable)
        summary["param_count"] = total

        # Step 4 — optimizer + cosine schedule
        epochs = epoch_override if epoch_override is not None else FIXED_RECIPE["epochs"]
        if epoch_override is not None:
            logger.info(
                "Epoch override: running %d epoch(s) instead of %d",
                epoch_override, FIXED_RECIPE["epochs"],
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=FIXED_RECIPE["lr"],
            weight_decay=FIXED_RECIPE["weight_decay"],
        )
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * FIXED_RECIPE["warmup_ratio"])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        logger.info("Total steps: %d | Warmup steps: %d", total_steps, warmup_steps)

        # Step 5 — training loop
        metrics_path = trial_dir / "metrics.jsonl"
        metrics_path.write_text("")

        best_val_loss = float("inf")
        best_val_spearman = float("-inf")
        best_epoch = -1
        patience_counter = 0
        early_stopped = False
        weights_path = trial_dir / "best.pt"

        final_train_loss = float("nan")
        final_train_spearman = float("nan")

        for epoch in range(epochs):
            epoch_start = time.time()

            train_loss, train_spearman, _ = train_epoch(
                model, train_loader, optimizer, scheduler, device,
                FIXED_RECIPE["grad_clip"],
            )
            val_loss, val_spearman, _ = validate(model, val_loader, device)

            elapsed = time.time() - epoch_start
            current_lr = scheduler.get_last_lr()[0]

            logger.info(
                "Variant %d | Epoch %02d | Train Loss: %.4f | Train Spearman: %.4f | "
                "Val Loss: %.4f | Val Spearman: %.4f | LR: %.2e | Time: %.1fs",
                trial_id, epoch + 1, train_loss, train_spearman,
                val_loss, val_spearman, current_lr, elapsed,
            )

            with metrics_path.open("a") as f:
                f.write(json.dumps({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_spearman": train_spearman,
                    "val_loss": val_loss,
                    "val_spearman": val_spearman,
                    "lr": current_lr,
                    "elapsed_s": elapsed,
                }) + "\n")

            final_train_loss = train_loss
            final_train_spearman = train_spearman

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_spearman = val_spearman
                best_epoch = epoch + 1
                patience_counter = 0
                save_best_model(model, weights_path)
                logger.info(
                    "Variant %d: new best val loss %.4f (Spearman %.4f) at epoch %d",
                    trial_id, best_val_loss, best_val_spearman, best_epoch,
                )
            else:
                patience_counter += 1
                if patience_counter >= FIXED_RECIPE["patience"]:
                    logger.info("Variant %d: early stopping at epoch %d", trial_id, epoch + 1)
                    early_stopped = True
                    break

        # Step 6 — test on best weights
        if weights_path.exists():
            best_ckpt = torch.load(weights_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
            test_loss, test_spearman, _ = validate(model, test_loader, device)
            logger.info(
                "Variant %d | Test Loss: %.4f | Test Spearman: %.4f",
                trial_id, test_loss, test_spearman,
            )
            summary["weights_path"] = str(weights_path.relative_to(DATA_DIR.parent))
        else:
            logger.warning("Variant %d: no weights saved (val loss never improved)", trial_id)
            test_loss = float("nan")
            test_spearman = float("nan")

        summary.update({
            "status": "complete",
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_spearman": best_val_spearman,
            "test_loss": test_loss,
            "test_spearman": test_spearman,
            "final_train_loss": final_train_loss,
            "final_train_spearman": final_train_spearman,
            "epochs_run": epoch + 1,
            "early_stopped": early_stopped,
        })

    except Exception as exc:
        logger.exception("Variant %d (%s) failed: %s", trial_id, name, exc)
        summary.update({
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })

    finally:
        wall_time = time.time() - wall_start
        summary["wall_time_s"] = wall_time
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()

        with (trial_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2, default=str)

        detach_trial_log(trial_handler)

        # Free memory and close h5/dataloader handles before next variant
        for obj_name in ("model", "optimizer", "scheduler", "train_loader", "val_loader", "test_loader"):
            try:
                del_obj = locals().get(obj_name)
                if del_obj is not None:
                    del del_obj
            except Exception:
                pass
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return summary


# ============================================================================
# Results aggregation
# ============================================================================

RESULT_COLUMNS = [
    "trial_id", "name", "description", "status",
    "best_epoch", "best_val_loss", "best_val_spearman",
    "test_loss", "test_spearman",
    "final_train_loss", "final_train_spearman",
    "epochs_run", "early_stopped", "param_count",
    "wall_time_s", "started_at", "finished_at",
    "device", "seed", "weights_path",
]
DATASET_COLUMNS = [
    "dataset_user_filter_sql", "dataset_track_filter", "dataset_score_method",
    "dataset_min_genres", "dataset_max_top5_conc",
    "dataset_min_library_size", "dataset_min_library_size_after_filter",
]


def append_result(sweep_dir: Path, summary: Dict) -> None:
    """Append one variant's summary to results.jsonl + results.csv."""
    jsonl_path = sweep_dir / "results.jsonl"
    csv_path = sweep_dir / "results.csv"

    with jsonl_path.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")

    row = {k: summary.get(k) for k in RESULT_COLUMNS}
    ds = summary.get("config", {}).get("dataset", {}) or {}
    row["dataset_user_filter_sql"] = ds.get("user_filter_sql")
    row["dataset_track_filter"] = ds.get("track_filter")
    row["dataset_score_method"] = ds.get("score_method")
    row["dataset_min_genres"] = ds.get("min_genres")
    row["dataset_max_top5_conc"] = ds.get("max_top5_conc")
    row["dataset_min_library_size"] = ds.get("min_library_size")
    row["dataset_min_library_size_after_filter"] = ds.get("min_library_size_after_filter")

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS + DATASET_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_completed_names(sweep_dir: Path) -> Set[str]:
    """Names of variants whose summary.json shows status=complete."""
    path = sweep_dir / "results.jsonl"
    if not path.exists():
        return set()
    done: Set[str] = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "complete":
                done.add(row["name"])
    return done


# ============================================================================
# Main
# ============================================================================

def default_sweep_id() -> str:
    return f"dataset_experiments_{datetime.now().strftime('%Y%m%d')}"


def ensure_embedding_index() -> Dict[str, int]:
    """Build or load the shared embedding index (lastfm_id -> embedding row)."""
    index_path = TRAINING_DATA_DIR / "embedding_index.json"
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        logger.info("Loading existing embedding index from %s", index_path)
        return load_embedding_index(index_path)
    logger.info("Building embedding index (one-time)")
    index = build_embedding_index(EMBEDDINGS_PATH)
    save_embedding_index(index, index_path)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one model per dataset theory")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    parser.add_argument("--sweep-id", default=None,
                        help="Sweep directory name (default: dataset_experiments_YYYYMMDD)")
    parser.add_argument("--only", default=None,
                        help="Comma-separated variant names to run (skip others)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned variants and exit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch-override", type=int, default=None,
                        help="Override epochs in every variant (for smoke tests)")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Only run the first N planned variants")
    parser.add_argument("--num-workers", type=int, default=TRAIN_NUM_WORKERS,
                        help="DataLoader workers")
    parser.add_argument("--prefetch-factor", type=int, default=TRAIN_PREFETCH_FACTOR,
                        help="Batches each DataLoader worker preloads")
    args = parser.parse_args()

    setup_root_logger()

    only_set = set(s.strip() for s in args.only.split(",")) if args.only else None
    variants = list(DATASET_VARIANTS)
    if only_set is not None:
        unknown = only_set - {v.name for v in variants}
        if unknown:
            logger.warning("Unknown variant names ignored: %s", sorted(unknown))
        variants = [v for v in variants if v.name in only_set]
    if args.max_trials is not None:
        variants = variants[: args.max_trials]

    sweep_id = args.sweep_id if args.sweep_id else default_sweep_id()
    sweep_dir = SWEEPS_DIR / sweep_id

    if args.dry_run:
        logger.info("=" * 70)
        logger.info("Dry-run | Sweep ID: %s | %d variant(s) planned", sweep_id, len(variants))
        logger.info("Output would be: %s", sweep_dir)
        logger.info("=" * 70)
        for i, v in enumerate(variants, 1):
            logger.info("[dry-run] Variant %d: %s", i, v.name)
            logger.info("           %s", v.description)
            logger.info("           user_filter_sql: %s", v.user_filter_sql or "(none)")
            logger.info("           track_filter: %s | score_method: %s",
                        v.track_filter, v.score_method)
            if v.min_genres != 3 or v.max_top5_conc != 0.60 or v.min_library_size != 50:
                logger.info(
                    "           overrides: min_genres=%d, max_top5_conc=%.2f, min_library_size=%d",
                    v.min_genres, v.max_top5_conc, v.min_library_size,
                )
        return

    device = get_device(args.device)
    logger.info("Device: %s", device)

    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the full plan for reproducibility
    with (sweep_dir / "sweep_config.json").open("w") as f:
        json.dump({
            "sweep_id": sweep_id,
            "seed": args.seed,
            "device": str(device),
            "epoch_override": args.epoch_override,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "fixed_recipe": FIXED_RECIPE,
            "variants": [v.to_serializable() for v in variants],
        }, f, indent=2, default=str)

    logger.info("=" * 70)
    logger.info("Sweep ID: %s | %d variant(s)", sweep_id, len(variants))
    logger.info("Output: %s", sweep_dir)
    logger.info("=" * 70)

    done = read_completed_names(sweep_dir)
    if done:
        logger.info("Resuming: %d variant(s) already complete: %s",
                    len(done), ", ".join(sorted(done)))

    # Build / load shared embedding index once
    embedding_index = ensure_embedding_index()

    for trial_id, variant in enumerate(variants, 1):
        if variant.name in done:
            logger.info("Skipping already-complete variant: %s", variant.name)
            continue

        summary = run_variant(
            trial_id=trial_id,
            variant=variant,
            sweep_dir=sweep_dir,
            embedding_index=embedding_index,
            device=device,
            seed=args.seed,
            epoch_override=args.epoch_override,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
        )
        append_result(sweep_dir, summary)

    logger.info("=" * 70)
    logger.info("All variants finished. Results: %s", sweep_dir / "results.jsonl")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
