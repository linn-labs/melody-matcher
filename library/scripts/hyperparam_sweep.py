#!/usr/bin/env python3
"""Hyperparameter Sweep for Taste-Scoring Neural Network

Historical explicit-execution research tooling; not a supported workflow.
Read docs/COMPONENTS.md and docs/DATA_AND_RIGHTS.md before considering use.
"""

import argparse
import copy
import csv
import gc
import json
import logging
import random
import sys
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import DATA_DIR
from library.src.dataset import get_dataloaders
from library.src.model import create_model
from library.scripts.train import (
    compute_spearman,
    get_cosine_schedule_with_warmup,
    get_device,
    masked_mse_loss,
    save_best_model,
    train_epoch,
    validate,
)

SWEEPS_DIR = DATA_DIR / "sweeps"

logger = logging.getLogger("hyperparam_sweep")


# ============================================================================
# Trial configuration
# ============================================================================

BASELINE_CONFIG: Dict = {
    # optimization
    "lr": 1e-4,
    "weight_decay": 0.01,
    "batch_size": 32,
    "warmup_ratio": 0.05,
    "grad_clip": 1.0,
    "epochs": 20,
    "patience": 5,
    # regularization
    "dropout": 0.1,
    # architecture
    "d_model": 512,
    "num_heads": 8,
    "enc_layers": 3,
    "dec_layers": 2,
    "dim_feedforward": 2048,
    # data pipeline
    "context_ratio": 0.7,
    "window_size": 300,
}


def _variant(name: str, **overrides) -> Dict:
    cfg = deepcopy(BASELINE_CONFIG)
    cfg.update(overrides)
    cfg["name"] = name
    return cfg


STAGE1_TRIALS: List[Dict] = [
    _variant("baseline"),
    _variant("lr_high", lr=3e-4),
    _variant("lr_low", lr=3e-5),
    _variant("lr_very_high", lr=1e-3),
    _variant("wd_high", weight_decay=0.1),
    _variant("wd_low", weight_decay=1e-4),
    _variant("bs_small", batch_size=16),
    _variant("bs_large", batch_size=64),
    _variant("warmup_long", warmup_ratio=0.15),
    _variant("warmup_short", warmup_ratio=0.01),
    _variant("dropout_high", dropout=0.3),
    _variant("dropout_low", dropout=0.0),
    _variant("epochs_long", epochs=40, patience=8),
    _variant("ctx_low", context_ratio=0.5),
    _variant("ctx_high", context_ratio=0.85),
    _variant("window_large", window_size=500),
    _variant("combo_best_guess_a", lr=3e-4, weight_decay=0.1, dropout=0.2, epochs=30, patience=7),
    _variant("combo_best_guess_b", lr=3e-4, warmup_ratio=0.1, batch_size=64, epochs=30, patience=7),
]


def build_stage2_trials(recipe: Dict) -> List[Dict]:
    """Build stage-2 trial list by varying only architecture knobs from a fixed recipe."""
    def arch(name: str, **overrides) -> Dict:
        cfg = deepcopy(recipe)
        cfg.update(overrides)
        cfg["name"] = name
        return cfg

    return [
        arch("arch_baseline"),
        arch("d_model_small", d_model=256, dim_feedforward=1024),
        arch("d_model_large", d_model=768, dim_feedforward=3072),
        arch("d_model_xlarge", d_model=1024, dim_feedforward=4096),
        arch("enc_deep", enc_layers=6),
        arch("enc_shallow", enc_layers=1),
        arch("dec_deep", dec_layers=4),
        arch("dec_shallow", dec_layers=1),
        arch("heads_many", num_heads=16),
        arch("heads_few", num_heads=4),
        arch("ffn_wide", dim_feedforward=4096),
        arch("ffn_narrow", dim_feedforward=1024),
        arch("big_model", d_model=768, enc_layers=5, dec_layers=3, dim_feedforward=3072),
        arch("deep_narrow", d_model=384, enc_layers=6, dec_layers=4),
    ]


ARCH_KEYS = ("d_model", "num_heads", "enc_layers", "dec_layers", "dropout", "dim_feedforward")
DATA_KEYS = ("batch_size", "context_ratio", "window_size")


# ============================================================================
# Logging setup
# ============================================================================

def setup_root_logger() -> None:
    """Configure root logger for the sweep. Removes train.py's file handler so
    the sweep doesn't pollute data/train.log."""
    root = logging.getLogger()
    # Detach train.py's FileHandler pointing at data/train.log
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename).name == "train.log":
            root.removeHandler(h)
            h.close()

    # Ensure we have a StreamHandler
    has_stream = any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                     for h in root.handlers)
    if not has_stream:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
        root.addHandler(sh)

    root.setLevel(logging.INFO)


def attach_trial_log(trial_dir: Path) -> logging.FileHandler:
    """Attach a per-trial file handler. Returns the handler so it can be removed after."""
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
# Dataloader cache
# ============================================================================

@dataclass
class DataLoaderCache:
    key: Optional[Tuple] = None
    loaders: Optional[Tuple] = None
    pin_memory: bool = False

    def get(self, batch_size: int, context_ratio: float, window_size: int, pin_memory: bool):
        key = (batch_size, context_ratio, window_size, pin_memory)
        if self.key == key and self.loaders is not None:
            return self.loaders
        logger.info("Building dataloaders (batch_size=%d, context_ratio=%.3f, window_size=%d)",
                    batch_size, context_ratio, window_size)
        loaders = get_dataloaders(
            batch_size=batch_size,
            context_ratio=context_ratio,
            window_size=window_size,
            pin_memory=pin_memory,
        )
        self.key = key
        self.loaders = loaders
        return loaders


# ============================================================================
# Per-trial training
# ============================================================================

def run_trial(
    trial_id: int,
    stage: str,
    config: Dict,
    sweep_dir: Path,
    device: torch.device,
    seed: int,
    loader_cache: DataLoaderCache,
    epoch_override: Optional[int],
) -> Dict:
    """Train one configuration end-to-end and return a summary dict."""
    name = config["name"]
    slug = f"trial_{trial_id:02d}_{name}"
    trial_dir = sweep_dir / slug
    trial_dir.mkdir(parents=True, exist_ok=True)

    trial_handler = attach_trial_log(trial_dir)
    started_at = datetime.now(timezone.utc)
    wall_start = time.time()

    summary: Dict = {
        "trial_id": trial_id,
        "name": name,
        "stage": stage,
        "status": "running",
        "config": {k: v for k, v in config.items() if k != "name"},
        "seed": seed,
        "device": str(device),
        "started_at": started_at.isoformat(),
        "weights_path": None,
    }

    try:
        logger.info("=" * 70)
        logger.info("Trial %d: %s (stage=%s)", trial_id, name, stage)
        logger.info("=" * 70)
        for k, v in config.items():
            if k == "name":
                continue
            logger.info("  %s = %s", k, v)

        seed_everything(seed)

        epochs = config["epochs"] if epoch_override is None else epoch_override
        if epoch_override is not None:
            logger.info("Epoch override in effect: running %d epoch(s) instead of %d",
                        epoch_override, config["epochs"])

        # Dataloaders (cached by data-pipeline key)
        train_loader, val_loader, test_loader = loader_cache.get(
            batch_size=config["batch_size"],
            context_ratio=config["context_ratio"],
            window_size=config["window_size"],
            pin_memory=(device.type == "cuda"),
        )
        logger.info("Train batches: %d | Val batches: %d | Test batches: %d",
                    len(train_loader), len(val_loader), len(test_loader))

        # Model
        model = create_model(
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            enc_layers=config["enc_layers"],
            dec_layers=config["dec_layers"],
            dropout=config["dropout"],
            dim_feedforward=config["dim_feedforward"],
        ).to(device)
        total, trainable = model.count_parameters()
        logger.info("Parameters: %d total, %d trainable", total, trainable)
        summary["param_count"] = total

        # Optimizer + scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * config["warmup_ratio"])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        logger.info("Total steps: %d | Warmup steps: %d", total_steps, warmup_steps)

        # Training loop
        metrics_path = trial_dir / "metrics.jsonl"
        metrics_path.write_text("")  # truncate if resuming a failed trial

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

            train_loss, train_spearman = train_epoch(
                model, train_loader, optimizer, scheduler, device, config["grad_clip"]
            )
            val_loss, val_spearman = validate(model, val_loader, device)

            elapsed = time.time() - epoch_start
            current_lr = scheduler.get_last_lr()[0]

            logger.info(
                "Trial %d | Epoch %02d | Train Loss: %.4f | Train Spearman: %.4f | "
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
                logger.info("Trial %d: new best val loss %.4f (Spearman %.4f)",
                            trial_id, best_val_loss, best_val_spearman)
            else:
                patience_counter += 1
                if patience_counter >= config["patience"]:
                    logger.info("Trial %d: early stopping at epoch %d", trial_id, epoch + 1)
                    early_stopped = True
                    break

        # Final test evaluation using best weights
        if weights_path.exists():
            best_ckpt = torch.load(weights_path, map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
            test_loss, test_spearman = validate(model, test_loader, device)
            logger.info("Trial %d | Test Loss: %.4f | Test Spearman: %.4f",
                        trial_id, test_loss, test_spearman)
            summary["weights_path"] = str(weights_path.relative_to(DATA_DIR.parent))
        else:
            logger.warning("Trial %d: no weights saved (val loss never improved)", trial_id)
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
        logger.exception("Trial %d (%s) failed: %s", trial_id, name, exc)
        summary.update({
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })

    finally:
        wall_time = time.time() - wall_start
        finished_at = datetime.now(timezone.utc)
        summary["wall_time_s"] = wall_time
        summary["finished_at"] = finished_at.isoformat()

        # Write per-trial summary.json
        with (trial_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2, default=str)

        detach_trial_log(trial_handler)

        # Free memory before the next trial
        try:
            del model  # noqa: F821
        except Exception:
            pass
        try:
            del optimizer, scheduler  # noqa: F821
        except Exception:
            pass
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return summary


# ============================================================================
# Results logging
# ============================================================================

RESULT_COLUMNS = [
    "trial_id", "name", "stage", "status",
    "best_epoch", "best_val_loss", "best_val_spearman",
    "test_loss", "test_spearman",
    "final_train_loss", "final_train_spearman",
    "epochs_run", "early_stopped", "param_count",
    "wall_time_s", "started_at", "finished_at",
    "device", "seed", "weights_path",
    # flattened config columns appended after
]
CONFIG_COLUMN_ORDER = [
    "lr", "weight_decay", "batch_size", "warmup_ratio", "grad_clip",
    "epochs", "patience", "dropout",
    "d_model", "num_heads", "enc_layers", "dec_layers", "dim_feedforward",
    "context_ratio", "window_size",
]


def append_result(sweep_dir: Path, summary: Dict) -> None:
    jsonl_path = sweep_dir / "results.jsonl"
    csv_path = sweep_dir / "results.csv"

    with jsonl_path.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")

    row = {k: summary.get(k) for k in RESULT_COLUMNS}
    for k in CONFIG_COLUMN_ORDER:
        row[f"cfg_{k}"] = summary.get("config", {}).get(k)

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS + [f"cfg_{k}" for k in CONFIG_COLUMN_ORDER])
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_completed_names(sweep_dir: Path) -> set:
    """Read results.jsonl and return the set of trial names already completed."""
    path = sweep_dir / "results.jsonl"
    if not path.exists():
        return set()
    done = set()
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
# Stage-2 recipe auto-pick
# ============================================================================

def find_best_stage1_recipe(explicit_path: Optional[Path]) -> Dict:
    """Locate the best stage-1 configuration, either from an explicit path or by
    scanning data/sweeps/ for the most recent directory with stage=optimization
    results.jsonl and picking the trial with max best_val_spearman."""
    if explicit_path is not None:
        logger.info("Loading base config from %s", explicit_path)
        with explicit_path.open() as f:
            cfg = json.load(f)
        if "config" in cfg:  # accept a full results row too
            cfg = cfg["config"]
        cfg.setdefault("name", "from_base_config")
        return cfg

    if not SWEEPS_DIR.exists():
        raise FileNotFoundError(
            f"No sweeps directory at {SWEEPS_DIR}. Run stage optimization first, "
            f"or pass --base-config pointing at a config JSON."
        )

    candidates = sorted(
        [p for p in SWEEPS_DIR.iterdir() if p.is_dir() and (p / "results.jsonl").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for sweep_dir in candidates:
        best_row = None
        with (sweep_dir / "results.jsonl").open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("stage") != "optimization" or row.get("status") != "complete":
                    continue
                if row.get("best_val_spearman") is None:
                    continue
                if best_row is None or row["best_val_spearman"] > best_row["best_val_spearman"]:
                    best_row = row
        if best_row is not None:
            logger.info(
                "Auto-picked stage-1 winner from %s: trial '%s' with val Spearman %.4f",
                sweep_dir.name, best_row["name"], best_row["best_val_spearman"],
            )
            recipe = deepcopy(best_row["config"])
            recipe["name"] = "from_" + best_row["name"]
            return recipe

    raise FileNotFoundError(
        "No completed stage=optimization trials found in any sweep directory. "
        "Run stage optimization first, or pass --base-config."
    )


# ============================================================================
# Weight cleanup
# ============================================================================

def prune_weights(sweep_dir: Path, keep_weights: str, top_k: int) -> None:
    """Delete best.pt files for trials outside the top-k by val Spearman.

    keep_weights: "all" | "top-k" | "none"
    """
    if keep_weights == "all":
        return

    rows = []
    path = sweep_dir / "results.jsonl"
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "complete" and row.get("best_val_spearman") is not None:
                rows.append(row)

    if keep_weights == "none":
        keep_ids = set()
    else:  # top-k
        rows.sort(key=lambda r: r["best_val_spearman"], reverse=True)
        keep_ids = {r["trial_id"] for r in rows[:top_k]}

    for row in rows:
        if row["trial_id"] in keep_ids:
            continue
        slug = f"trial_{row['trial_id']:02d}_{row['name']}"
        wpath = sweep_dir / slug / "best.pt"
        if wpath.exists():
            wpath.unlink()
            logger.info("Pruned %s", wpath)


# ============================================================================
# Main
# ============================================================================

def default_sweep_id(stage: str) -> str:
    return f"{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for taste-scoring NN")
    parser.add_argument("--stage", choices=["optimization", "architecture", "both"],
                        default="optimization")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    parser.add_argument("--sweep-id", default=None,
                        help="Sweep directory name (default: timestamped per stage)")
    parser.add_argument("--base-config", type=Path, default=None,
                        help="Path to config.json to use as stage-2 recipe (overrides auto-pick)")
    parser.add_argument("--max-trials", type=int, default=None,
                        help="Only run the first N planned trials")
    parser.add_argument("--only", default=None,
                        help="Comma-separated trial names to run (skip others)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned trials and exit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch-override", type=int, default=None,
                        help="Override epochs in every trial (for smoke tests)")
    parser.add_argument("--keep-weights", choices=["all", "top-k", "none"], default="top-k",
                        help="After sweep finishes, which best.pt files to retain")
    parser.add_argument("--top-k", type=int, default=5,
                        help="K for --keep-weights top-k")
    args = parser.parse_args()

    setup_root_logger()

    stages_to_run: List[str]
    if args.stage == "both":
        stages_to_run = ["optimization", "architecture"]
    else:
        stages_to_run = [args.stage]

    only_set = set(s.strip() for s in args.only.split(",")) if args.only else None

    device = get_device(args.device)
    logger.info("Device: %s", device)

    stage2_recipe_for_both: Optional[Dict] = None

    for stage in stages_to_run:
        if stage == "optimization":
            trials = STAGE1_TRIALS
        else:
            if stage2_recipe_for_both is not None:
                recipe = stage2_recipe_for_both
            else:
                try:
                    recipe = find_best_stage1_recipe(args.base_config)
                except FileNotFoundError:
                    if not args.dry_run:
                        raise
                    logger.warning(
                        "No stage-1 recipe found; dry-run will use BASELINE_CONFIG "
                        "as a placeholder. Real runs still require a recipe."
                    )
                    recipe = deepcopy(BASELINE_CONFIG)
                    recipe["name"] = "placeholder_baseline"
            trials = build_stage2_trials(recipe)

        if only_set is not None:
            trials = [t for t in trials if t["name"] in only_set]
        if args.max_trials is not None:
            trials = trials[:args.max_trials]

        sweep_id = args.sweep_id if args.sweep_id else default_sweep_id(stage)
        sweep_dir = SWEEPS_DIR / sweep_id

        if args.dry_run:
            logger.info("=" * 70)
            logger.info("Stage: %s | Sweep ID: %s | %d trial(s) planned (dry-run)",
                        stage, sweep_id, len(trials))
            logger.info("Output would be: %s", sweep_dir)
            logger.info("=" * 70)
            for i, cfg in enumerate(trials, 1):
                logger.info("[dry-run] Trial %d: %s", i, cfg["name"])
                for k, v in cfg.items():
                    if k == "name":
                        continue
                    logger.info("           %s = %s", k, v)
            continue

        sweep_dir.mkdir(parents=True, exist_ok=True)

        # Snapshot full planned config for reproducibility
        with (sweep_dir / "sweep_config.json").open("w") as f:
            json.dump({
                "stage": stage,
                "sweep_id": sweep_id,
                "seed": args.seed,
                "device": str(device),
                "epoch_override": args.epoch_override,
                "trials": trials,
            }, f, indent=2, default=str)

        logger.info("=" * 70)
        logger.info("Stage: %s | Sweep ID: %s | %d trial(s) planned",
                    stage, sweep_id, len(trials))
        logger.info("Output: %s", sweep_dir)
        logger.info("=" * 70)

        done = read_completed_names(sweep_dir)
        if done:
            logger.info("Resuming: %d trial(s) already complete: %s",
                        len(done), ", ".join(sorted(done)))

        loader_cache = DataLoaderCache()

        best_of_stage: Optional[Dict] = None

        for trial_id, cfg in enumerate(trials, 1):
            if cfg["name"] in done:
                logger.info("Skipping already-complete trial: %s", cfg["name"])
                continue

            summary = run_trial(
                trial_id=trial_id,
                stage=stage,
                config=cfg,
                sweep_dir=sweep_dir,
                device=device,
                seed=args.seed,
                loader_cache=loader_cache,
                epoch_override=args.epoch_override,
            )
            append_result(sweep_dir, summary)

            if (
                stage == "optimization"
                and summary.get("status") == "complete"
                and summary.get("best_val_spearman") is not None
            ):
                if (best_of_stage is None
                        or summary["best_val_spearman"] > best_of_stage["best_val_spearman"]):
                    best_of_stage = summary

        prune_weights(sweep_dir, args.keep_weights, args.top_k)

        logger.info("=" * 70)
        logger.info("Stage '%s' finished. Results: %s", stage, sweep_dir / "results.jsonl")
        logger.info("=" * 70)

        if stage == "optimization" and "architecture" in stages_to_run and best_of_stage is not None:
            recipe = deepcopy(best_of_stage["config"])
            recipe["name"] = "from_" + best_of_stage["name"]
            stage2_recipe_for_both = recipe
            logger.info(
                "Carrying forward stage-1 winner '%s' (val Spearman %.4f) into stage 2",
                best_of_stage["name"], best_of_stage["best_val_spearman"],
            )


if __name__ == "__main__":
    main()
