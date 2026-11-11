"""
Unit tests verifying schema and structural integrity of Power BI data exports.
"""

from pathlib import Path
import pandas as pd
import pytest
from src.config import POWERBI_DATA_DIR


def test_dataset_summary_schema():
    csv_path = POWERBI_DATA_DIR / "dataset_summary.csv"
    if not csv_path.exists():
        pytest.skip("dataset_summary.csv not generated yet")

    df = pd.read_csv(csv_path)
    expected_cols = {
        "class",
        "total_slices",
        "unique_groups",
        "mean_width",
        "mean_height",
        "mean_brightness",
        "corrupt_images",
        "exact_duplicates",
    }
    assert expected_cols.issubset(df.columns)
    assert len(df) >= 3


def test_predictions_schema():
    csv_path = POWERBI_DATA_DIR / "predictions.csv"
    if not csv_path.exists():
        pytest.skip("predictions.csv not generated yet")

    df = pd.read_csv(csv_path)
    expected_cols = {
        "image_id",
        "path",
        "split",
        "model",
        "true_label",
        "pred_label",
        "conf",
        "p_nCT",
        "p_pCT",
        "p_NiCT",
        "correct",
        "error_type",
        "width",
        "height",
        "mean_brightness",
    }
    assert expected_cols.issubset(df.columns)


def test_model_comparison_schema():
    csv_path = POWERBI_DATA_DIR / "model_comparison.csv"
    if not csv_path.exists():
        pytest.skip("model_comparison.csv not generated yet")

    df = pd.read_csv(csv_path)
    expected_cols = {
        "model",
        "split",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "params",
        "train_time_min",
    }
    assert expected_cols.issubset(df.columns)


def test_predictions_row_counts_match_splits():
    """
    Requirement 4: Verify that predictions.csv row counts per model and split
    exactly match the raw ground truth row counts in data/splits/*.csv.
    """
    from src.config import SPLITS_DIR
    pred_path = POWERBI_DATA_DIR / "predictions.csv"
    if not pred_path.exists():
        pytest.skip("predictions.csv not generated yet")

    df_preds = pd.read_csv(pred_path)
    models = df_preds["model"].unique()
    assert len(models) >= 1, "No models present in predictions.csv"

    for split_name in ["train", "val", "test"]:
        split_file = SPLITS_DIR / f"{split_name}.csv"
        assert split_file.exists(), f"Split file missing: {split_file}"
        expected_count = len(pd.read_csv(split_file))

        for m in models:
            actual_count = len(df_preds[(df_preds["model"] == m) & (df_preds["split"] == split_name)])
            assert actual_count == expected_count, (
                f"Count mismatch for model '{m}' on split '{split_name}': "
                f"expected {expected_count} rows from {split_name}.csv, got {actual_count} rows in predictions.csv"
            )
