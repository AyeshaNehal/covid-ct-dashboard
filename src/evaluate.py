"""
Comprehensive evaluation pipeline: multi-class metrics, ROC/PR curves,
calibration curves, confusion matrices, and Power BI data exports.

DISCLAIMER: This project is for educational and portfolio purposes only.
It is NOT a medical device and must NOT be used for clinical diagnosis.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    CHECKPOINTS_DIR,
    CLASS_NAMES,
    CLASS_TO_IDX,
    FIGURES_DIR,
    IDX_TO_CLASS,
    POWERBI_DATA_DIR,
    PROJECT_ROOT,
    SPLITS_DIR,
    SUPPORTED_MODELS,
    get_device,
)
from src.dataset import CTScanDataset, get_eval_transforms
from src.model import build_model, count_parameters
from src.utils import (
    plot_calibration_curves,
    plot_confusion_matrix,
    plot_pr_curves,
    plot_roc_curves,
    setup_logger,
)

logger = setup_logger("evaluate")


def extract_image_metadata(img_rel_path: str) -> Tuple[int, int, float]:
    """Return (width, height, mean_brightness [0, 1]) for an image."""
    full_path = PROJECT_ROOT / img_rel_path
    try:
        with Image.open(full_path) as img:
            w, h = img.size
            gray = img.convert("L")
            arr = np.asarray(gray, dtype=np.float32) / 255.0
            mean_bright = float(arr.mean())
            return w, h, round(mean_bright, 4)
    except Exception as e:
        logger.warning(f"Could not read metadata for {full_path}: {e}")
        return 224, 224, 0.5


def evaluate_split(
    model: nn.Module,
    model_name: str,
    split_name: str,
    df_split: pd.DataFrame,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run forward pass on a specific split (train, val, or test).
    Returns (predictions_dataframe, summary_metrics_dict).
    """
    model.eval()
    dataset = CTScanDataset(df_split, transform=get_eval_transforms())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds = []
    all_targets = []
    all_probs = []
    all_paths = []

    with torch.no_grad():
        for images, targets, paths in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_probs.append(probs)
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())
            all_paths.extend(paths)

    all_probs = np.vstack(all_probs)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Build per-prediction records
    records = []
    for i, path_str in enumerate(all_paths):
        true_cls = IDX_TO_CLASS[all_targets[i]]
        pred_cls = IDX_TO_CLASS[all_preds[i]]
        conf = float(all_probs[i, all_preds[i]])
        correct = 1 if true_cls == pred_cls else 0
        err_type = "None" if correct == 1 else f"{true_cls}->{pred_cls}"
        w, h, bright = extract_image_metadata(path_str)

        records.append({
            "image_id": Path(path_str).stem,
            "path": path_str,
            "split": split_name,
            "model": model_name,
            "true_label": true_cls,
            "pred_label": pred_cls,
            "conf": round(conf, 4),
            "p_nCT": round(float(all_probs[i, CLASS_TO_IDX["nCT"]]), 4),
            "p_pCT": round(float(all_probs[i, CLASS_TO_IDX["pCT"]]), 4),
            "p_NiCT": round(float(all_probs[i, CLASS_TO_IDX["NiCT"]]), 4),
            "correct": correct,
            "error_type": err_type,
            "width": w,
            "height": h,
            "mean_brightness": bright,
        })

    pred_df = pd.DataFrame(records)

    # Compute aggregate metrics
    acc = float(np.mean(all_preds == all_targets))
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, average=None, labels=[0, 1, 2], zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )

    metrics = {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i in range(len(CLASS_NAMES))
        },
        "confusion_matrix": confusion_matrix(all_targets, all_preds, labels=[0, 1, 2]),
        "y_true": all_targets,
        "y_probs": all_probs,
    }

    return pred_df, metrics


def run_full_evaluation(
    model_names: Optional[List[str]] = None,
    batch_size: int = 64,
) -> None:
    """
    Run evaluation across FULL train, val, and test splits for all specified models.
    Generates predictions.csv (~19,685 rows per model), model_comparison.csv,
    class_metrics.csv, and publication figures.
    """
    if model_names is None:
        model_names = SUPPORTED_MODELS

    device = get_device()
    logger.info(f"Running evaluation on device: {device} for models: {model_names} across FULL splits.")

    splits = {
        "train": pd.read_csv(SPLITS_DIR / "train.csv"),
        "val": pd.read_csv(SPLITS_DIR / "val.csv"),
        "test": pd.read_csv(SPLITS_DIR / "test.csv"),
    }
    for s_name, s_df in splits.items():
        logger.info(f"Split '{s_name}' contains {len(s_df)} ground-truth rows.")

    all_pred_dfs = []
    comparison_records = []
    class_metric_records = []

    # Check existing training history for real wall-clock train times
    history_csv = POWERBI_DATA_DIR / "training_history.csv"
    train_times = {}
    if history_csv.exists():
        h_df = pd.read_csv(history_csv)
        if "train_time_min" in h_df.columns:
            for m, grp in h_df.groupby("model"):
                train_times[m] = round(grp["train_time_min"].max(), 2)

    for model_name in model_names:
        ckpt_path = CHECKPOINTS_DIR / f"best_{model_name}.pth"
        if not ckpt_path.exists():
            logger.warning(f"Checkpoint not found for {model_name} at {ckpt_path}. Skipping.")
            continue

        logger.info(f"Loading checkpoint {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = build_model(model_name, num_classes=len(CLASS_NAMES), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        total_params, _ = count_parameters(model)

        for split_name, df_split in splits.items():
            pred_df, metrics = evaluate_split(
                model=model,
                model_name=model_name,
                split_name=split_name,
                df_split=df_split,
                device=device,
                batch_size=batch_size,
            )
            all_pred_dfs.append(pred_df)

            # Record model comparison
            comparison_records.append({
                "model": model_name,
                "split": split_name,
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "params": total_params,
                "train_time_min": train_times.get(model_name, 0.0),
            })

            # Record class metrics
            for cls_name, c_met in metrics["per_class"].items():
                class_metric_records.append({
                    "model": model_name,
                    "split": split_name,
                    "class": cls_name,
                    "precision": c_met["precision"],
                    "recall": c_met["recall"],
                    "f1": c_met["f1"],
                    "support": c_met["support"],
                })

            # Generate figures for test split
            if split_name == "test":
                logger.info(f"Generating test figures for {model_name}...")
                # Confusion matrix
                plot_confusion_matrix(
                    cm=metrics["confusion_matrix"],
                    class_names=CLASS_NAMES,
                    output_path=FIGURES_DIR / f"confusion_matrix_{model_name}.png",
                    title=f"Test Confusion Matrix - {model_name}",
                )

                # ROC Curves
                y_bin = label_binarize(metrics["y_true"], classes=[0, 1, 2])
                plot_roc_curves(
                    y_true_bin=y_bin,
                    y_probs=metrics["y_probs"],
                    class_names=CLASS_NAMES,
                    output_path=FIGURES_DIR / f"roc_curves_{model_name}.png",
                    title=f"Test ROC Curves - {model_name}",
                )

                # PR Curves
                plot_pr_curves(
                    y_true_bin=y_bin,
                    y_probs=metrics["y_probs"],
                    class_names=CLASS_NAMES,
                    output_path=FIGURES_DIR / f"pr_curves_{model_name}.png",
                    title=f"Test Precision-Recall Curves - {model_name}",
                )

                # Calibration Curves
                plot_calibration_curves(
                    y_true_bin=y_bin,
                    y_probs=metrics["y_probs"],
                    class_names=CLASS_NAMES,
                    output_path=FIGURES_DIR / f"calibration_curves_{model_name}.png",
                    title=f"Test Calibration Curves - {model_name}",
                )

    # Save / Update consolidated Power BI CSVs incrementally
    if all_pred_dfs:
        new_pred_df = pd.concat(all_pred_dfs, ignore_index=True)
        pred_csv = POWERBI_DATA_DIR / "predictions.csv"
        if pred_csv.exists():
            existing_preds = pd.read_csv(pred_csv)
            # Remove any previous records for the evaluated models
            existing_preds = existing_preds[~existing_preds["model"].isin(model_names)]
            full_pred_df = pd.concat([existing_preds, new_pred_df], ignore_index=True)
        else:
            full_pred_df = new_pred_df
        full_pred_df.to_csv(pred_csv, index=False)
        logger.info(f"Updated {pred_csv} (now {len(full_pred_df)} total prediction records)")

    if comparison_records:
        new_comp_df = pd.DataFrame(comparison_records)
        comp_csv = POWERBI_DATA_DIR / "model_comparison.csv"
        if comp_csv.exists():
            existing_comp = pd.read_csv(comp_csv)
            existing_comp = existing_comp[~existing_comp["model"].isin(model_names)]
            full_comp_df = pd.concat([existing_comp, new_comp_df], ignore_index=True)
        else:
            full_comp_df = new_comp_df
        full_comp_df.to_csv(comp_csv, index=False)
        logger.info(f"Updated model comparison table in {comp_csv}")
        print("\n=== MODEL TEST COMPARISON ===")
        print(full_comp_df[full_comp_df["split"] == "test"].to_string(index=False))

    if class_metric_records:
        new_class_df = pd.DataFrame(class_metric_records)
        class_csv = POWERBI_DATA_DIR / "class_metrics.csv"
        if class_csv.exists():
            existing_class = pd.read_csv(class_csv)
            existing_class = existing_class[~existing_class["model"].isin(model_names)]
            full_class_df = pd.concat([existing_class, new_class_df], ignore_index=True)
        else:
            full_class_df = new_class_df
        full_class_df.to_csv(class_csv, index=False)
        logger.info(f"Updated per-class metrics table in {class_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate COVID-CT classifiers on full splits.")
    parser.add_argument("--model", type=str, default=None, choices=SUPPORTED_MODELS)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    models = [args.model] if args.model else SUPPORTED_MODELS
    run_full_evaluation(model_names=models, batch_size=args.batch_size)
