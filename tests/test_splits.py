"""
Unit tests for data splits, leakage prevention, and class representation.
"""

from pathlib import Path
import pandas as pd
import pytest
from src.config import CLASS_NAMES, SPLITS_DIR


def test_splits_files_exist():
    for split_name in ["train", "val", "test"]:
        split_path = SPLITS_DIR / f"{split_name}.csv"
        assert split_path.exists(), f"Split file missing: {split_path}"


def test_splits_schema_and_classes():
    for split_name in ["train", "val", "test"]:
        df = pd.read_csv(SPLITS_DIR / f"{split_name}.csv")
        assert {"path", "label", "split"}.issubset(df.columns), f"Columns missing in {split_name}.csv"
        assert len(df) > 0, f"Split {split_name} is empty"

        # Check all 3 classes represented
        classes_present = set(df["label"].unique())
        assert classes_present == set(CLASS_NAMES), f"Expected classes {CLASS_NAMES}, got {classes_present}"


def test_no_group_leakage_across_splits():
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    if "group_id" in train_df.columns and "group_id" in test_df.columns:
        train_groups = set(train_df["group_id"].unique())
        val_groups = set(val_df["group_id"].unique())
        test_groups = set(test_df["group_id"].unique())

        assert train_groups.isdisjoint(val_groups), "Leakage: Train and Validation share groups!"
        assert train_groups.isdisjoint(test_groups), "Leakage: Train and Test share groups!"
        assert val_groups.isdisjoint(test_groups), "Leakage: Validation and Test share groups!"
