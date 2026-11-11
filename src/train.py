"""
Training pipeline with AdamW, Cosine Annealing, Early Stopping,
and Power BI training history logging.

DISCLAIMER: This project is for educational and portfolio purposes only.
It is NOT a medical device and must NOT be used for clinical diagnosis.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.config import (
    CHECKPOINTS_DIR,
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    FIGURES_DIR,
    POWERBI_DATA_DIR,
    RANDOM_SEED,
    SPLITS_DIR,
    SUPPORTED_MODELS,
    get_device,
)
from src.dataset import CTScanDataset, get_eval_transforms, get_train_transforms
from src.model import build_model, count_parameters
from src.utils import plot_training_curves, set_seed, setup_logger

logger = setup_logger("train")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train model for a single epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    val_loss = running_loss / max(total, 1)
    val_acc = correct / max(total, 1)
    return val_loss, val_acc


def train_model(
    model_name: str = "mobilenet_v3_large",
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    patience: int = 7,
    seed: int = RANDOM_SEED,
) -> Tuple[Path, pd.DataFrame, float]:
    """
    Main training execution loop on full datasets.
    Returns: (best_checkpoint_path, history_dataframe, total_training_time_minutes)
    """
    set_seed(seed)
    device = get_device()
    logger.info(f"Starting training run on FULL splits | Model: {model_name} | Device: {device} | Epochs: {epochs}")

    # Load full splits
    train_csv = SPLITS_DIR / "train.csv"
    val_csv = SPLITS_DIR / "val.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(f"Splits not found at {SPLITS_DIR}. Run splits pipeline first.")

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    logger.info(f"Loaded {len(train_df)} training samples and {len(val_df)} validation samples.")

    train_dataset = CTScanDataset(train_df, transform=get_train_transforms())
    val_dataset = CTScanDataset(val_df, transform=get_eval_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows-friendly multiprocessing
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Initialize model
    model = build_model(model_name, num_classes=len(CLASS_NAMES), pretrained=True)
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_checkpoint_path = CHECKPOINTS_DIR / f"best_{model_name}.pth"

    history_records = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        history_records.append({
            "model": model_name,
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "lr": round(optimizer.param_groups[0]["lr"], 7),
            "train_time_min": round((time.time() - start_time) / 60.0, 2),
        })

        # Save best model by val accuracy
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": model_name,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                best_checkpoint_path,
            )
            logger.info(f"--> Saved new best checkpoint to {best_checkpoint_path} (Val Acc: {best_val_acc:.4f})")
        else:
            epochs_no_improve += 1

        # Early stopping check
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs).")
            break

    total_training_minutes = round((time.time() - start_time) / 60.0, 2)
    history_df = pd.DataFrame(history_records)

    # Plot training curves
    plot_training_curves(history_df, FIGURES_DIR / f"training_curves_{model_name}.png", model_name=model_name)

    # Append to cumulative training_history.csv for Power BI
    history_csv = POWERBI_DATA_DIR / "training_history.csv"
    if history_csv.exists():
        existing_history = pd.read_csv(history_csv)
        # Remove old records for this model to avoid duplication
        existing_history = existing_history[existing_history["model"] != model_name]
        updated_history = pd.concat([existing_history, history_df], ignore_index=True)
    else:
        updated_history = history_df

    updated_history.to_csv(history_csv, index=False)
    logger.info(f"Training completed in {total_training_minutes} min. History logged to {history_csv}")

    return best_checkpoint_path, history_df, total_training_minutes


def parse_args():
    parser = argparse.ArgumentParser(description="Train COVID-CT classifier models.")
    parser.add_argument("--model", type=str, default="resnet18", choices=SUPPORTED_MODELS)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--all-models", action="store_true", help="Train all supported architectures sequentially")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    models_sequence = ["resnet18", "mobilenet_v3_large", "efficientnet_b0"]
    if args.all_models:
        for m in models_sequence:
            logger.info(f"=== Sequentially Training Architecture: {m} ===")
            train_model(
                model_name=m,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
            )
    else:
        train_model(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )
