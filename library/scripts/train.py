#!/usr/bin/env python3
"""Training Script for Taste-Scoring Neural Network

Historical explicit-execution research tooling; not a supported workflow.
Read docs/COMPONENTS.md and docs/DATA_AND_RIGHTS.md before considering use.
"""

import sys
import argparse
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, rankdata

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    TRAIN_BATCH_SIZE,
    TRAIN_CONTEXT_RATIO,
    TRAIN_WINDOW_SIZE,
    TRAIN_LEARNING_RATE,
    TRAIN_WEIGHT_DECAY,
    TRAIN_WARMUP_RATIO,
    TRAIN_EPOCHS,
    TRAIN_PATIENCE,
    TRAIN_GRAD_CLIP,
    TRAIN_NUM_WORKERS,
    TRAIN_PREFETCH_FACTOR,
    CHECKPOINT_DIR,
    DATA_DIR,
)
from library.src.dataset import get_dataloaders
from library.src.model import create_model

# Logging setup
LOG_DIR = Path(__file__).parent.parent.parent / "data"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "train.log"),
    ],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Loss and Metrics
# ============================================================================

def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute MSE loss only on non-padded positions.

    Args:
        pred: (B, M, 1) predicted scores
        target: (B, M, 1) actual scores
        mask: (B, M) True where padded (positions to ignore)

    Returns:
        Scalar MSE loss averaged over valid positions
    """
    valid_mask = ~mask  # True where valid
    valid_mask = valid_mask.unsqueeze(-1)  # (B, M, 1)

    squared_error = (pred - target) ** 2
    masked_error = squared_error * valid_mask

    num_valid = valid_mask.sum()
    if num_valid == 0:
        return torch.tensor(0.0, device=pred.device)

    return masked_error.sum() / num_valid


def hurdle_loss(
    play_logits: torch.Tensor,
    score_pred: torch.Tensor,
    target_scores: torch.Tensor,
    is_positive: torch.Tensor,
    target_mask: torch.Tensor,
    alpha: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """BCE over (positive vs negative) + alpha * MSE over positives only.

    Args:
        play_logits: (B, M, 1) — logits for "would the user play this?"
        score_pred:  (B, M, 1) — predicted score | played
        target_scores: (B, M, 1) — true scores (only meaningful for positives)
        is_positive:   (B, M)   — True for real library tracks, False for negs
        target_mask:   (B, M)   — True where padding
        alpha: BCE + alpha * MSE weighting

    Returns:
        (total_loss, {"bce": float, "mse_pos": float})
    """
    valid = (~target_mask)  # (B, M)
    valid_f = valid.float()
    is_pos_f = is_positive.float()

    bce_per = F.binary_cross_entropy_with_logits(
        play_logits.squeeze(-1), is_pos_f, reduction="none"
    )  # (B, M)
    n_valid = valid_f.sum().clamp(min=1.0)
    bce_loss = (bce_per * valid_f).sum() / n_valid

    pos_valid = (is_positive & valid).float()  # (B, M)
    mse_per = ((score_pred - target_scores) ** 2).squeeze(-1)
    n_pos = pos_valid.sum()
    if n_pos > 0:
        mse_loss = (mse_per * pos_valid).sum() / n_pos
    else:
        mse_loss = torch.zeros((), device=play_logits.device)

    total = bce_loss + alpha * mse_loss
    return total, {"bce": float(bce_loss.detach().item()),
                   "mse_pos": float(mse_loss.detach().item())}


def batch_auc(scalar_pred: torch.Tensor, is_positive: torch.Tensor,
              target_mask: torch.Tensor) -> float:
    """ROC-AUC of (scalar prediction) ranked against (is_positive label).

    Hand-rolled via rank-sum so we don't pull in sklearn just for this:
        AUC = (sum_of_ranks_of_positives - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    Ties are handled by `rankdata` (average rank).
    Returns NaN if the batch has no contrast (all-pos or all-neg).
    """
    valid = ~target_mask
    preds = scalar_pred.squeeze(-1)[valid].detach().cpu().numpy().astype(np.float64)
    labels = is_positive[valid].detach().cpu().numpy().astype(np.int64)
    n = labels.size
    if n < 2:
        return float("nan")
    n_pos = int(labels.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(preds, method="average")
    sum_pos = float(ranks[labels == 1].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_spearman(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute Spearman correlation on non-padded positions.

    Spearman correlation measures ranking quality:
    - +1 = perfect ranking
    - 0 = random
    - -1 = perfectly inverted

    Args:
        pred: (B, M, 1) predicted scores
        target: (B, M, 1) actual scores
        mask: (B, M) True where padded

    Returns:
        Spearman correlation coefficient (float)
    """
    valid_mask = ~mask
    all_preds = []
    all_targets = []

    for i in range(pred.shape[0]):
        valid = valid_mask[i]
        if valid.sum() > 1:  # Need at least 2 points for correlation
            all_preds.extend(pred[i, valid, 0].cpu().numpy())
            all_targets.extend(target[i, valid, 0].cpu().numpy())

    if len(all_preds) < 2:
        return 0.0

    correlation, _ = spearmanr(all_preds, all_targets)

    # Handle NaN (can happen if all predictions are identical)
    if np.isnan(correlation):
        return 0.0

    return correlation


# ============================================================================
# Learning Rate Scheduler
# ============================================================================

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Create a learning rate scheduler with linear warmup and cosine decay.

    The learning rate:
    1. Starts at 0
    2. Linearly increases to the base LR over warmup_steps
    3. Follows a cosine curve down to 0 over the remaining steps

    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of steps for linear warmup
        total_steps: Total number of training steps

    Returns:
        A LambdaLR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase: linear increase from 0 to 1
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)

        # Cosine decay phase: from 1 to 0
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Checkpointing
# ============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    val_loss: float,
    best_val_loss: float,
    path: Path,
):
    """Save a training checkpoint.

    Saves everything needed to resume training:
    - Model weights
    - Optimizer state (momentum, adaptive learning rates)
    - Scheduler state
    - Training progress
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'model_config': model.config,
    }, path)

    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
):
    """Load a training checkpoint.

    Returns:
        Tuple of (start_epoch, best_val_loss)
    """
    logger.info("Loading checkpoint from %s", path)

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    best_val_loss = checkpoint['best_val_loss']

    logger.info("Resumed from epoch %d, best val loss: %.4f", checkpoint['epoch'], best_val_loss)

    return start_epoch, best_val_loss


def save_best_model(model: nn.Module, path: Path):
    """Save just the model weights (for inference)."""
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.config,
    }, path)

    logger.info("Saved best model to %s", path)


# ============================================================================
# Training and Validation
# ============================================================================

def _step_forward(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    alpha: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Forward + loss, branching on model output shape.

    Returns:
        (loss, scalar_pred, parts)
        scalar_pred: (B, M, 1) — the value used for ranking/AUC.
            For scoring models: the regression output.
            For hurdle models:  sigmoid(play_logits) (the "is this a real
            candidate?" probability). Spearman on positives compares this
            to target_scores — meaningful as a ranking signal, but the
            absolute value isn't directly comparable to log_norm scores.
        parts: extra scalar metrics ("bce", "mse_pos") for hurdle, {} else.
    """
    out = model.forward_batch(batch)
    if isinstance(out, dict):
        loss, parts = hurdle_loss(
            out["play_logits"], out["score_pred"],
            batch["target_scores"], batch["target_is_positive"],
            batch["target_mask"], alpha=alpha,
        )
        scalar_pred = torch.sigmoid(out["play_logits"])
        return loss, scalar_pred, parts
    loss = masked_mse_loss(out, batch["target_scores"], batch["target_mask"])
    return loss, out, {}


def _positives_only_spearman(
    pred: torch.Tensor,
    target: torch.Tensor,
    pad_mask: torch.Tensor,
    is_positive: torch.Tensor,
) -> float:
    """Spearman on (positives only), so the metric stays comparable across
    neg_0 / neg_K / hurdle variants."""
    valid = (~pad_mask) & is_positive
    valid = valid.unsqueeze(-1)  # (B, M, 1)
    preds_flat = pred[valid].detach().cpu().numpy()
    targets_flat = target[valid].detach().cpu().numpy()
    if preds_flat.size < 2:
        return 0.0
    correlation, _ = spearmanr(preds_flat, targets_flat)
    if np.isnan(correlation):
        return 0.0
    return float(correlation)


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    grad_clip: float,
    alpha: float = 1.0,
) -> Tuple[float, float, Dict[str, float]]:
    """Run one training epoch.

    Returns:
        (average_loss, positives_only_spearman, extras)
        extras: {"auc": ..., "bce": ..., "mse_pos": ...} where available.
    """
    model.train()
    total_loss = 0.0
    sum_bce = 0.0
    sum_mse_pos = 0.0
    parts_n = 0
    all_preds = []
    all_targets = []
    all_masks = []
    all_is_pos = []
    num_batches = 0

    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        loss, scalar_pred, parts = _step_forward(model, batch, alpha=alpha)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        if parts:
            sum_bce += parts.get("bce", 0.0)
            sum_mse_pos += parts.get("mse_pos", 0.0)
            parts_n += 1

        all_preds.append(scalar_pred.detach())
        all_targets.append(batch["target_scores"].detach())
        all_masks.append(batch["target_mask"].detach())
        all_is_pos.append(batch["target_is_positive"].detach())

    avg_loss = total_loss / max(1, num_batches)

    # Spearman / AUC on the last few batches (memory).
    tail = slice(-10, None)
    sp = _positives_only_spearman(
        torch.cat(all_preds[tail], dim=0),
        torch.cat(all_targets[tail], dim=0),
        torch.cat(all_masks[tail], dim=0),
        torch.cat(all_is_pos[tail], dim=0),
    )
    auc = batch_auc(
        torch.cat(all_preds[tail], dim=0),
        torch.cat(all_is_pos[tail], dim=0),
        torch.cat(all_masks[tail], dim=0),
    )

    extras: Dict[str, float] = {"auc": auc}
    if parts_n > 0:
        extras["bce"] = sum_bce / parts_n
        extras["mse_pos"] = sum_mse_pos / parts_n
    return avg_loss, sp, extras


def validate(
    model: nn.Module,
    val_loader,
    device: torch.device,
    alpha: float = 1.0,
) -> Tuple[float, float, Dict[str, float]]:
    """Run validation.

    Returns:
        (average_loss, positives_only_spearman, extras)
    """
    model.eval()
    total_loss = 0.0
    sum_bce = 0.0
    sum_mse_pos = 0.0
    parts_n = 0
    all_preds = []
    all_targets = []
    all_masks = []
    all_is_pos = []
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, scalar_pred, parts = _step_forward(model, batch, alpha=alpha)

            total_loss += loss.item()
            num_batches += 1
            if parts:
                sum_bce += parts.get("bce", 0.0)
                sum_mse_pos += parts.get("mse_pos", 0.0)
                parts_n += 1

            all_preds.append(scalar_pred)
            all_targets.append(batch["target_scores"])
            all_masks.append(batch["target_mask"])
            all_is_pos.append(batch["target_is_positive"])

    avg_loss = total_loss / max(1, num_batches)

    cat_preds = torch.cat(all_preds, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)
    cat_masks = torch.cat(all_masks, dim=0)
    cat_is_pos = torch.cat(all_is_pos, dim=0)

    sp = _positives_only_spearman(cat_preds, cat_targets, cat_masks, cat_is_pos)
    auc = batch_auc(cat_preds, cat_is_pos, cat_masks)

    extras: Dict[str, float] = {"auc": auc}
    if parts_n > 0:
        extras["bce"] = sum_bce / parts_n
        extras["mse_pos"] = sum_mse_pos / parts_n
    return avg_loss, sp, extras


def evaluate_ndcg(
    model: nn.Module,
    loader,
    device: torch.device,
    k: int = 50,
) -> float:
    """NDCG@k on a mixed positives+negatives ranking task.

    Each user in each batch becomes one ranking instance: pool all their
    target positions (positives + sampled negatives), rank by scalar_pred,
    compute NDCG@k using target_scores as graded relevance (negatives have
    score=0 by construction, so they get relevance 0).

    Returns the average NDCG@k across all (user, batch) instances. Requires
    the loader to have num_negatives > 0 in its dataset, else this measures
    only library-internal ranking quality.
    """
    model.eval()
    ndcg_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k_: v.to(device) for k_, v in batch.items()}
            _, scalar_pred, _ = _step_forward(model, batch, alpha=1.0)
            scalar_pred = scalar_pred.squeeze(-1).cpu().numpy()       # (B, M)
            relevance = batch["target_scores"].squeeze(-1).cpu().numpy()  # (B, M)
            mask = batch["target_mask"].cpu().numpy()                  # True = pad
            B = scalar_pred.shape[0]
            for i in range(B):
                valid = ~mask[i]
                if valid.sum() < 2:
                    continue
                preds_i = scalar_pred[i, valid]
                rels_i = relevance[i, valid]
                # Top-k by predicted score
                topk_idx = np.argsort(-preds_i)[:k]
                gains = (np.power(2.0, rels_i[topk_idx]) - 1.0)
                discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
                dcg = float((gains * discounts).sum())
                # Ideal DCG: sort by true relevance
                ideal_idx = np.argsort(-rels_i)[:k]
                ideal_gains = (np.power(2.0, rels_i[ideal_idx]) - 1.0)
                ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_gains.size + 2))
                idcg = float((ideal_gains * ideal_discounts).sum())
                if idcg <= 0:
                    continue
                ndcg_sum += dcg / idcg
                n += 1
    return ndcg_sum / max(1, n)


# ============================================================================
# Main
# ============================================================================

def get_device(device_str: str) -> torch.device:
    """Parse device string and return torch device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def main():
    parser = argparse.ArgumentParser(description="Train taste-scoring model")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=TRAIN_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=TRAIN_NUM_WORKERS, help="DataLoader workers")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=TRAIN_PREFETCH_FACTOR,
        help="Batches each worker preloads",
    )
    args = parser.parse_args()

    # Setup
    device = get_device(args.device)
    logger.info("=" * 60)
    logger.info("Training Taste-Scoring Model")
    logger.info("=" * 60)
    logger.info("Device: %s", device)
    logger.info("Epochs: %d", args.epochs)
    logger.info("Learning rate: %g", args.lr)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("DataLoader workers: %d", args.num_workers)
    logger.info("Prefetch factor: %d", args.prefetch_factor)
    logger.info("Weight decay: %g", TRAIN_WEIGHT_DECAY)
    logger.info("Warmup ratio: %g", TRAIN_WARMUP_RATIO)
    logger.info("Gradient clip: %g", TRAIN_GRAD_CLIP)
    logger.info("Early stopping patience: %d", TRAIN_PATIENCE)

    # Data
    logger.info("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        context_ratio=TRAIN_CONTEXT_RATIO,
        window_size=TRAIN_WINDOW_SIZE,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=args.prefetch_factor,
    )
    logger.info("Train batches: %d", len(train_loader))
    logger.info("Val batches: %d", len(val_loader))

    # Model
    logger.info("Creating model...")
    model = create_model()
    model = model.to(device)
    total_params, trainable_params = model.count_parameters()
    logger.info("Parameters: %d total, %d trainable", total_params, trainable_params)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=TRAIN_WEIGHT_DECAY,
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * TRAIN_WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    logger.info("Total steps: %d, warmup steps: %d", total_steps, warmup_steps)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        start_epoch, best_val_loss = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, device
        )

    # Training loop
    patience_counter = 0

    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_loss, train_spearman, _ = train_epoch(
            model, train_loader, optimizer, scheduler, device, TRAIN_GRAD_CLIP
        )

        val_loss, val_spearman, _ = validate(model, val_loader, device)

        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch %02d | Train Loss: %.4f | Train Spearman: %.4f | "
            "Val Loss: %.4f | Val Spearman: %.4f | LR: %.2e | Time: %.1fs",
            epoch + 1, train_loss, train_spearman,
            val_loss, val_spearman, current_lr, epoch_time
        )

        # Save checkpoint
        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch + 1:02d}.pt"
        save_checkpoint(
            model, optimizer, scheduler, epoch, val_loss, best_val_loss, checkpoint_path
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_best_model(model, DATA_DIR / "best_model.pt")
            logger.info("New best validation loss: %.4f", best_val_loss)
        else:
            patience_counter += 1
            logger.info("No improvement for %d epoch(s)", patience_counter)

            if patience_counter >= TRAIN_PATIENCE:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                break

    # Final evaluation on test set
    logger.info("=" * 60)
    logger.info("Evaluating on test set...")
    logger.info("=" * 60)

    # Load best model for final evaluation
    best_checkpoint = torch.load(DATA_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])

    test_loss, test_spearman, _ = validate(model, test_loader, device)
    logger.info("Test Loss: %.4f | Test Spearman: %.4f", test_loss, test_spearman)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Best model saved to: %s", DATA_DIR / "best_model.pt")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
