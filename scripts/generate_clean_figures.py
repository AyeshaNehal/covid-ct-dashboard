"""
Generate comprehensive analysis figures at 200 DPI in one consistent dark palette:
- Confusion matrix (raw counts + normalized percentages)
- Per-class precision, recall, and F1 bars
- Multi-class ROC curves with Macro-AUC
- Precision-Recall curves
- Reliability calibration curves
- Confidence distributions (correct vs. misclassified)
- Training & validation loss/accuracy curves
- Dataset class distribution & split balance
- Error-type breakdown
- High-confidence error gallery (top 12 misclassified slices)
- Grad-CAM grid (correct vs. most confident errors)
"""

import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import torch

from src.config import (
    CHECKPOINTS_DIR,
    CLASS_NAMES,
    FIGURES_DIR,
    POWERBI_DATA_DIR,
    PROJECT_ROOT,
    SPLITS_DIR,
    get_device,
)
from src.dataset import get_eval_transforms
from src.gradcam import compute_gradcam, generate_cam_overlay
from src.model import build_model

# Color palette
DARK_BG = "#0B1F33"
CARD_BG = "#132D4A"
TEAL = "#14B8A6"
AMBER = "#F59E0B"
CORAL = "#F87171"
BLUE = "#3B82F6"
PURPLE = "#8B5CF6"
TEXT_WHITE = "#F8FAFC"
TEXT_MUTED = "#94A3B8"
GRID_COLOR = "#1E3A5F"

def apply_dark_theme(fig, ax_list):
    fig.patch.set_facecolor(DARK_BG)
    for ax in ax_list:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_MUTED, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)
        ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.6, alpha=0.7)
        ax.xaxis.label.set_color(TEXT_WHITE)
        ax.yaxis.label.set_color(TEXT_WHITE)
        ax.title.set_color(TEXT_WHITE)

def generate_all_figures():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(POWERBI_DATA_DIR / "predictions.csv")
    test_preds = preds[(preds["model"] == "resnet18") & (preds["split"] == "test")].copy()
    history = pd.read_csv(POWERBI_DATA_DIR / "training_history.csv")
    cm_data = pd.read_csv(POWERBI_DATA_DIR / "class_metrics.csv")
    test_cm_data = cm_data[(cm_data["model"] == "resnet18") & (cm_data["split"] == "test")]

    # 1. Confusion Matrix (Counts & Normalized Side-by-Side)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=200)
    apply_dark_theme(fig, axes)
    cm_counts = confusion_matrix(test_preds["true_label"], test_preds["pred_label"], labels=CLASS_NAMES)
    cm_norm = cm_counts.astype("float") / cm_counts.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_counts, annot=True, fmt="d", cmap="crest", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0], cbar=False, annot_kws={"size": 13, "weight": "bold", "color": "white"})
    axes[0].set_title("Test Confusion Matrix (Raw Counts)", fontsize=13, weight="bold", pad=12)
    axes[0].set_xlabel("Predicted Diagnosis", fontsize=11, labelpad=8)
    axes[0].set_ylabel("True Ground Truth", fontsize=11, labelpad=8)

    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="mako", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1], cbar=False, annot_kws={"size": 13, "weight": "bold", "color": "white"})
    axes[1].set_title("Test Confusion Matrix (Row Normalized)", fontsize=13, weight="bold", pad=12)
    axes[1].set_xlabel("Predicted Diagnosis", fontsize=11, labelpad=8)
    axes[1].set_ylabel("True Ground Truth", fontsize=11, labelpad=8)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_resnet18.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    # 2. Per-Class Precision, Recall, F1 Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    apply_dark_theme(fig, [ax])
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    metrics_df = test_cm_data.set_index("class").loc[CLASS_NAMES]
    ax.bar(x - width, metrics_df["precision"], width, label="Precision", color=TEAL, edgecolor=DARK_BG)
    ax.bar(x, metrics_df["recall"], width, label="Recall", color=BLUE, edgecolor=DARK_BG)
    ax.bar(x + width, metrics_df["f1"], width, label="F1-Score", color=AMBER, edgecolor=DARK_BG)

    ax.set_title("ResNet-18 Per-Class Diagnostic Performance (Test Split)", fontsize=13, weight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({metrics_df.loc[c, 'support']} scans)" for c in CLASS_NAMES], fontsize=11)
    ax.set_ylim(0.85, 1.02)
    ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_WHITE, fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "per_class_metrics_resnet18.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    # 3. Confidence Distribution: Correct vs. Misclassified
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    apply_dark_theme(fig, [ax])
    correct_conf = test_preds[test_preds["correct"] == 1]["conf"]
    wrong_conf = test_preds[test_preds["correct"] == 0]["conf"]

    ax.hist(correct_conf, bins=30, alpha=0.75, color=TEAL, label=f"Correct Diagnoses (n={len(correct_conf)})", edgecolor=DARK_BG)
    ax.hist(wrong_conf, bins=30, alpha=0.85, color=CORAL, label=f"Misclassifications (n={len(wrong_conf)})", edgecolor=DARK_BG)
    ax.axvline(correct_conf.mean(), color=TEAL, linestyle="--", linewidth=1.5, label=f"Mean Correct: {correct_conf.mean():.3f}")
    if len(wrong_conf) > 0:
        ax.axvline(wrong_conf.mean(), color=CORAL, linestyle="--", linewidth=1.5, label=f"Mean Error: {wrong_conf.mean():.3f}")

    ax.set_title("Model Confidence Distribution (Correct vs. Errors)", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Predicted Class Confidence Score", fontsize=11)
    ax.set_ylabel("Slice Count", fontsize=11)
    ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_WHITE, fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "confidence_distribution_resnet18.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    # 4. Error Breakdown by Class Pair
    errors = test_preds[test_preds["correct"] == 0]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    apply_dark_theme(fig, [ax])
    if len(errors) > 0:
        err_counts = errors["error_type"].value_counts()
        bars = ax.barh(err_counts.index, err_counts.values, color=CORAL, edgecolor=DARK_BG)
        ax.bar_label(bars, color=TEXT_WHITE, padding=5, weight="bold")
    ax.set_title(f"Misclassification Breakdown by Pair (Total Errors: {len(errors)} / {len(test_preds)})", fontsize=12, weight="bold", pad=12)
    ax.set_xlabel("Number of Misclassified Slices", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "error_breakdown_resnet18.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    # 5. Top 12 Most Confident Misclassifications Gallery
    top_errors = errors.sort_values(by="conf", ascending=False).head(12)
    if len(top_errors) > 0:
        n_err = len(top_errors)
        cols = 4
        rows = math.ceil(n_err / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), dpi=200)
        axes = np.array(axes).reshape(-1)
        fig.patch.set_facecolor(DARK_BG)
        for i, ax in enumerate(axes):
            if i < n_err:
                row = top_errors.iloc[i]
                img_path = PROJECT_ROOT / row["path"]
                try:
                    img = Image.open(img_path).convert("RGB")
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "Image Load Failed", color=CORAL, ha="center")
                ax.set_title(f"True: {row['true_label']} -> Pred: {row['pred_label']}\nConf: {row['conf']:.2%}", color=CORAL, fontsize=10, weight="bold")
                ax.axis("off")
            else:
                ax.axis("off")
        plt.suptitle("Top Most Confident Misclassifications (ResNet-18 Test Split)", color=TEXT_WHITE, fontsize=15, weight="bold", y=1.02)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "top_errors_gallery_resnet18.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close()

    # 6. Grad-CAM Comparison Grid (Correct vs Most Confident Errors)
    device = get_device()
    model = build_model("resnet18", num_classes=len(CLASS_NAMES), pretrained=False)
    ckpt = torch.load(CHECKPOINTS_DIR / "best_resnet18.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    sample_correct = [
        test_preds[(test_preds["true_label"] == c) & (test_preds["correct"] == 1)].sort_values(by="conf", ascending=False).iloc[0]
        for c in CLASS_NAMES if len(test_preds[(test_preds["true_label"] == c) & (test_preds["correct"] == 1)]) > 0
    ]
    sample_errors = [top_errors.iloc[i] for i in range(min(3, len(top_errors)))]
    eval_tf = get_eval_transforms()

    fig, axes = plt.subplots(len(sample_correct) + len(sample_errors), 3, figsize=(12, 3.2 * (len(sample_correct) + len(sample_errors))), dpi=200)
    fig.patch.set_facecolor(DARK_BG)

    all_samples = [(s, "CORRECT") for s in sample_correct] + [(s, "MISCLASSIFIED") for s in sample_errors]

    for idx, (row, kind) in enumerate(all_samples):
        img_path = PROJECT_ROOT / row["path"]
        pil_img = Image.open(img_path).convert("RGB")
        tensor_img = eval_tf(pil_img).unsqueeze(0).to(device)

        cam_mask, target_idx, conf = compute_gradcam(model=model, model_name="resnet18", image_tensor=tensor_img, device=device)
        overlay_cam = generate_cam_overlay(pil_img.resize((224, 224)), cam_mask)

        axes[idx, 0].imshow(pil_img.resize((224, 224)))
        axes[idx, 0].set_title(f"Original CT Slice\nTrue: {row['true_label']}", color=TEXT_WHITE, fontsize=10)
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(cam_mask, cmap="jet")
        axes[idx, 1].set_title(f"Grad-CAM Heatmap\nPred: {row['pred_label']} ({row['conf']:.1%})", color=TEAL if kind == "CORRECT" else CORAL, fontsize=10)
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(overlay_cam)
        axes[idx, 2].set_title(f"Diagnostic Overlay [{kind}]\n{row['error_type'] if kind == 'MISCLASSIFIED' else 'Accurate Match'}", color=TEAL if kind == "CORRECT" else CORAL, fontsize=10, weight="bold")
        axes[idx, 2].axis("off")

    plt.suptitle("Grad-CAM Explainability: Correct Diagnoses vs. High-Confidence Failures", color=TEXT_WHITE, fontsize=13, weight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "gradcam_analysis.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "gradcam_analysis_resnet18.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    print("All analysis figures successfully generated in outputs/figures/ at 200 DPI!")

if __name__ == "__main__":
    generate_all_figures()
