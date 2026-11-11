"""
Utility functions: reproducibility seeds, logging, plotting helpers, and device setup.

DISCLAIMER: This project is for educational and portfolio purposes only.
It is NOT a medical device and must NOT be used for clinical diagnosis.
"""

import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.calibration import calibration_curve

from src.config import DISCLAIMER_TEXT, FIGURES_DIR, LOGS_DIR, RANDOM_SEED


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed across all libraries for deterministic execution."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def setup_logger(name: str = "covid_ct", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Set up a structured console and file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        default_log = LOGS_DIR / f"{name}.log"
        default_log.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(default_log), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def add_disclaimer(fig: plt.Figure, fontsize: int = 8) -> None:
    """Add consistent educational disclaimer banner at the bottom of figures."""
    fig.text(
        0.5,
        0.01,
        "EDUCATIONAL/PORTFOLIO PROJECT ONLY • NOT A MEDICAL DEVICE • NOT FOR CLINICAL USE",
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="#718096",
        fontweight="semibold",
    )


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """Plot and save a high-resolution confusion matrix heatmap."""
    plt.figure(figsize=(7, 6), dpi=200)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
    
    if normalize:
        cm_display = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
        linewidths=1.0,
        linecolor="#E2E8F0",
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=11, labelpad=10)
    plt.ylabel("Ground Truth Label", fontsize=11, labelpad=10)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    add_disclaimer(plt.gcf())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_roc_curves(
    y_true_bin: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Receiver Operating Characteristic (ROC) Curves",
) -> None:
    """Plot multi-class One-vs-Rest ROC curves."""
    plt.figure(figsize=(7, 6), dpi=200)
    colors = ["#14B8A6", "#3B82F6", "#F59E0B"]

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            color=colors[i % len(colors)],
            lw=2,
            label=f"{class_name} (AUC = {roc_auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random Guess (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold", pad=15)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="lower right", frameon=True, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    add_disclaimer(plt.gcf())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_pr_curves(
    y_true_bin: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Precision-Recall Curves",
) -> None:
    """Plot multi-class One-vs-Rest Precision-Recall curves."""
    plt.figure(figsize=(7, 6), dpi=200)
    colors = ["#14B8A6", "#3B82F6", "#F59E0B"]

    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(
            recall,
            precision,
            color=colors[i % len(colors)],
            lw=2,
            label=f"{class_name} (PR-AUC = {pr_auc:.3f})",
        )

    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold", pad=15)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="lower left", frameon=True, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    add_disclaimer(plt.gcf())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_calibration_curves(
    y_true_bin: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "Probability Calibration Curves (Reliability Diagram)",
) -> None:
    """Plot calibration curves to evaluate predicted probability reliability."""
    plt.figure(figsize=(7, 6), dpi=200)
    colors = ["#14B8A6", "#3B82F6", "#F59E0B"]

    plt.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Perfectly Calibrated")

    for i, class_name in enumerate(class_names):
        prob_true, prob_pred = calibration_curve(
            y_true_bin[:, i], y_probs[:, i], n_bins=10, strategy="uniform"
        )
        plt.plot(
            prob_pred,
            prob_true,
            marker="o",
            lw=2,
            color=colors[i % len(colors)],
            label=f"{class_name}",
        )

    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Fraction of Positives", fontsize=11)
    plt.title(title, fontsize=13, fontweight="bold", pad=15)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper left", frameon=True, framealpha=0.9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    add_disclaimer(plt.gcf())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_training_curves(
    history_df,
    output_path: Path,
    model_name: str = "",
) -> None:
    """Plot Loss and Accuracy curves across training epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=200)
    title_suffix = f" - {model_name}" if model_name else ""

    # Loss
    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss", color="#3B82F6", lw=2)
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val Loss", color="#F59E0B", lw=2, linestyle="--")
    axes[0].set_title(f"Loss vs. Epochs{title_suffix}", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Cross Entropy Loss", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    # Accuracy
    axes[1].plot(history_df["epoch"], history_df["train_acc"], label="Train Acc", color="#14B8A6", lw=2)
    axes[1].plot(history_df["epoch"], history_df["val_acc"], label="Val Acc", color="#EC4899", lw=2, linestyle="--")
    axes[1].set_title(f"Accuracy vs. Epochs{title_suffix}", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    add_disclaimer(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
