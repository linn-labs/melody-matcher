#!/usr/bin/env python3
"""Negative-sampling + hurdle-model sweep.

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
from dataclasses import asdict
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
    evaluate_ndcg,
    get_cosine_schedule_with_warmup,
    get_device,
    save_best_model,
    train_epoch,
    validate,
)


SWEEPS_DIR = DATA_DIR / "sweeps"
logger = logging.getLogger("negative_sampling_experiments")


# ============================================================================
# Fixed training recipe — same as dataset_experiments so cross-sweep numbers
# can be compared epoch-for-epoch on the same dataset.
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
# Shared base dataset: baseline_all_diverse. All variants reuse its prepared
# training dir, since num_negatives is sampled at dataloader time and doesn't
# require re-preparing the per-user HDF5.
# ============================================================================

BASE_DATASET = DatasetVariant(
    name="baseline_all_diverse",
    description="Control: all users passing diversity filter, no cohort restriction.",
)


def _v(name: str, description: str, **overrides) -> DatasetVariant:
    """Spawn a DatasetVariant from BASE_DATASET with overrides applied."""
    base = asdict(BASE_DATASET)
    base.pop("name")
    base.pop("description")
    # user_filter_params is a tuple post-asdict it'll be a list; restore.
    if "user_filter_params" in base and isinstance(base["user_filter_params"], list):
        base["user_filter_params"] = tuple(base["user_filter_params"])
    base.update(overrides)
    return DatasetVariant(name=name, description=description, **base)


NEGATIVE_VARIANTS: List[DatasetVariant] = [
    _v("neg_0",   "Baseline — no negative sampling (current behavior).",
       num_negatives=0),
    _v("neg_30",  "Single-head MSE with K=30 sampled negatives (~1:3 pos:neg).",
       num_negatives=30),
    _v("neg_90",  "Single-head MSE with K=90 sampled negatives (~1:1 pos:neg).",
       num_negatives=90),
    _v("neg_270", "Single-head MSE with K=270 sampled negatives (~3:1 pos:neg).",
       num_negatives=270),
    _v("neg_900", "Single-head MSE with K=900 sampled negatives (~10:1 pos:neg).",
       num_negatives=900),
    _v("hurdle_neg_270",
       "Two-head hurdle (BCE + alpha*MSE-on-positives) with K=270 negatives.",
       num_negatives=270, model_type="hurdle", hurdle_alpha=1.0),
]


# ============================================================================
# Logging setup (same pattern as dataset_experiments.py)
# ============================================================================

def setup_root_logger() -> None:
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
    batch_size_override: Optional[int],
) -> Dict:
    name = variant.name
    slug = f"exp_{name}"
    trial_dir = sweep_dir / slug
    trial_dir.mkdir(parents=True, exist_ok=True)

    trial_handler = attach_trial_log(trial_dir)
    started_at = datetime.now(timezone.utc)
    wall_start = time.time()

    batch_size = batch_size_override or FIXED_RECIPE["batch_size"]

    summary: Dict = {
        "trial_id": trial_id,
        "name": name,
        "description": variant.description,
        "stage": "negative_sampling_experiment",
        "status": "running",
        "config": {**FIXED_RECIPE, "batch_size": batch_size,
                   "dataset": variant.to_serializable()},
        "seed": seed,
        "device": str(device),
        "started_at": started_at.isoformat(),
        "weights_path": None,
    }

    model = None
    optimizer = None
    scheduler = None
    train_loader = val_loader = test_loader = None
    epoch = -1

    try:
        logger.info("=" * 70)
        logger.info("Variant %d: %s", trial_id, name)
        logger.info("  %s", variant.description)
        logger.info("  num_negatives=%d, model_type=%s, alpha=%.2f, batch_size=%d",
                    variant.num_negatives, variant.model_type,
                    variant.hurdle_alpha, batch_size)
        logger.info("=" * 70)

        seed_everything(seed)

        # All variants share the same prepared training data — only the loss
        # axis varies. Prepare once (idempotent / resumable).
        variant_training_dir = TRAINING_DATA_DIR / f"exp_{BASE_DATASET.name}"
        prep_result = prepare_dataset_variant(
            variant=BASE_DATASET,
            training_dir=variant_training_dir,
            embedding_index=embedding_index,
            conn=get_connection(),
        )
        summary["dataset_prep"] = prep_result

        # Dataloaders — num_negatives lives here, not in the prepared data.
        train_loader, val_loader, test_loader = get_dataloaders(
            training_dir=variant_training_dir,
            embeddings_path=EMBEDDINGS_PATH,
            batch_size=batch_size,
            context_ratio=FIXED_RECIPE["context_ratio"],
            window_size=FIXED_RECIPE["window_size"],
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=prefetch_factor,
            num_negatives=variant.num_negatives,
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

        # Model — dispatch on model_type.
        model = create_model(
            model_type=variant.model_type,
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

        epochs = epoch_override if epoch_override is not None else FIXED_RECIPE["epochs"]
        if epoch_override is not None:
            logger.info("Epoch override: %d (vs %d)", epoch_override, FIXED_RECIPE["epochs"])

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=FIXED_RECIPE["lr"],
            weight_decay=FIXED_RECIPE["weight_decay"],
        )
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * FIXED_RECIPE["warmup_ratio"])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        logger.info("Total steps: %d | Warmup steps: %d", total_steps, warmup_steps)

        metrics_path = trial_dir / "metrics.jsonl"
        metrics_path.write_text("")

        best_val_loss = float("inf")
        best_val_spearman = float("-inf")
        best_val_auc = float("-inf")
        best_val_bce = float("nan")
        best_val_mse_pos = float("nan")
        best_epoch = -1
        patience_counter = 0
        early_stopped = False
        weights_path = trial_dir / "best.pt"

        final_train_loss = float("nan")
        final_train_spearman = float("nan")

        alpha = variant.hurdle_alpha

        for epoch in range(epochs):
            epoch_start = time.time()

            train_loss, train_spearman, train_extras = train_epoch(
                model, train_loader, optimizer, scheduler, device,
                FIXED_RECIPE["grad_clip"], alpha=alpha,
            )
            val_loss, val_spearman, val_extras = validate(
                model, val_loader, device, alpha=alpha,
            )

            elapsed = time.time() - epoch_start
            current_lr = scheduler.get_last_lr()[0]

            extra_log = ""
            if "auc" in val_extras and not np.isnan(val_extras["auc"]):
                extra_log += f" | Val AUC: {val_extras['auc']:.4f}"
            if "bce" in val_extras:
                extra_log += f" | BCE: {val_extras['bce']:.4f}"
            if "mse_pos" in val_extras:
                extra_log += f" | MSE+: {val_extras['mse_pos']:.4f}"

            logger.info(
                "Variant %d | Epoch %02d | Train Loss: %.4f | Train Spearman: %.4f | "
                "Val Loss: %.4f | Val Spearman: %.4f%s | LR: %.2e | Time: %.1fs",
                trial_id, epoch + 1, train_loss, train_spearman,
                val_loss, val_spearman, extra_log, current_lr, elapsed,
            )

            with metrics_path.open("a") as f:
                f.write(json.dumps({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_spearman": train_spearman,
                    "train_auc": train_extras.get("auc"),
                    "val_loss": val_loss,
                    "val_spearman": val_spearman,
                    "val_auc": val_extras.get("auc"),
                    "val_bce": val_extras.get("bce"),
                    "val_mse_pos": val_extras.get("mse_pos"),
                    "lr": current_lr,
                    "elapsed_s": elapsed,
                }) + "\n")

            final_train_loss = train_loss
            final_train_spearman = train_spearman

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_spearman = val_spearman
                best_val_auc = val_extras.get("auc", float("nan"))
                best_val_bce = val_extras.get("bce", float("nan"))
                best_val_mse_pos = val_extras.get("mse_pos", float("nan"))
                best_epoch = epoch + 1
                patience_counter = 0
                save_best_model(model, weights_path)
                logger.info(
                    "Variant %d: new best val loss %.4f (Sp %.4f, AUC %.4f) at epoch %d",
                    trial_id, best_val_loss, best_val_spearman,
                    best_val_auc, best_epoch,
                )
            else:
                patience_counter += 1
                if patience_counter >= FIXED_RECIPE["patience"]:
                    logger.info("Variant %d: early stopping at epoch %d", trial_id, epoch + 1)
                    early_stopped = True
                    break

        # Final eval on best weights — test loss/Spearman + NDCG@50.
        if weights_path.exists():
            best_ckpt = torch.load(weights_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
            test_loss, test_spearman, test_extras = validate(
                model, test_loader, device, alpha=alpha,
            )
            ndcg = evaluate_ndcg(model, test_loader, device, k=50)
            logger.info(
                "Variant %d | Test Loss: %.4f | Test Spearman: %.4f | "
                "Test AUC: %.4f | NDCG@50: %.4f",
                trial_id, test_loss, test_spearman,
                test_extras.get("auc", float("nan")), ndcg,
            )
            summary["weights_path"] = str(weights_path.relative_to(DATA_DIR.parent))
        else:
            logger.warning("Variant %d: no weights saved (val loss never improved)", trial_id)
            test_loss = float("nan")
            test_spearman = float("nan")
            test_extras = {}
            ndcg = float("nan")

        summary.update({
            "status": "complete",
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_val_spearman": best_val_spearman,
            "best_val_auc": best_val_auc,
            "best_val_bce": best_val_bce,
            "best_val_mse_pos": best_val_mse_pos,
            "test_loss": test_loss,
            "test_spearman": test_spearman,
            "test_auc": test_extras.get("auc"),
            "test_bce": test_extras.get("bce"),
            "test_mse_pos": test_extras.get("mse_pos"),
            "test_ndcg50": ndcg,
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

        for obj_name in ("model", "optimizer", "scheduler",
                         "train_loader", "val_loader", "test_loader"):
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
# Results aggregation (same shape as dataset_experiments for tooling reuse)
# ============================================================================

RESULT_COLUMNS = [
    "trial_id", "name", "description", "status",
    "best_epoch", "best_val_loss", "best_val_spearman", "best_val_auc",
    "best_val_bce", "best_val_mse_pos",
    "test_loss", "test_spearman", "test_auc", "test_ndcg50",
    "final_train_loss", "final_train_spearman",
    "epochs_run", "early_stopped", "param_count",
    "wall_time_s", "started_at", "finished_at",
    "device", "seed", "weights_path",
]
DATASET_COLUMNS = [
    "dataset_num_negatives", "dataset_model_type", "dataset_hurdle_alpha",
]


def append_result(sweep_dir: Path, summary: Dict) -> None:
    jsonl_path = sweep_dir / "results.jsonl"
    csv_path = sweep_dir / "results.csv"

    with jsonl_path.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")

    row = {k: summary.get(k) for k in RESULT_COLUMNS}
    ds = summary.get("config", {}).get("dataset", {}) or {}
    row["dataset_num_negatives"] = ds.get("num_negatives")
    row["dataset_model_type"] = ds.get("model_type")
    row["dataset_hurdle_alpha"] = ds.get("hurdle_alpha")

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS + DATASET_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_completed_names(sweep_dir: Path) -> Set[str]:
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
    return f"negative_experiments_{datetime.now().strftime('%Y%m%d')}"


def ensure_embedding_index() -> Dict[str, int]:
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
    parser = argparse.ArgumentParser(description="Negative-sampling + hurdle sweep")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    parser.add_argument("--sweep-id", default=None,
                        help="Sweep dir name (default: negative_experiments_YYYYMMDD)")
    parser.add_argument("--only", default=None,
                        help="Comma-separated variant names to run")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned variants and exit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch-override", type=int, default=None,
                        help="Override epochs in every variant (smoke tests)")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=TRAIN_NUM_WORKERS)
    parser.add_argument("--prefetch-factor", type=int, default=TRAIN_PREFETCH_FACTOR)
    parser.add_argument("--batch-size-override", type=int, default=None,
                        help="Override batch size (use for neg_900 if OOM).")
    args = parser.parse_args()

    setup_root_logger()

    only_set = set(s.strip() for s in args.only.split(",")) if args.only else None
    variants = list(NEGATIVE_VARIANTS)
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
        logger.info("Dry-run | Sweep ID: %s | %d variant(s) planned",
                    sweep_id, len(variants))
        logger.info("Output would be: %s", sweep_dir)
        logger.info("=" * 70)
        for i, v in enumerate(variants, 1):
            logger.info("[dry-run] Variant %d: %s", i, v.name)
            logger.info("           %s", v.description)
            logger.info("           num_negatives=%d | model_type=%s | alpha=%.2f",
                        v.num_negatives, v.model_type, v.hurdle_alpha)
        return

    device = get_device(args.device)
    logger.info("Device: %s", device)

    sweep_dir.mkdir(parents=True, exist_ok=True)
    with (sweep_dir / "sweep_config.json").open("w") as f:
        json.dump({
            "sweep_id": sweep_id,
            "seed": args.seed,
            "device": str(device),
            "epoch_override": args.epoch_override,
            "num_workers": args.num_workers,
            "prefetch_factor": args.prefetch_factor,
            "batch_size_override": args.batch_size_override,
            "fixed_recipe": FIXED_RECIPE,
            "base_dataset": BASE_DATASET.to_serializable(),
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
            batch_size_override=args.batch_size_override,
        )
        append_result(sweep_dir, summary)

    logger.info("=" * 70)
    logger.info("All variants finished. Results: %s", sweep_dir / "results.jsonl")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
