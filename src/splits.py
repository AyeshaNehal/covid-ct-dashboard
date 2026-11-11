"""
Leakage-aware dataset splitting logic for COVID CT slices.
Ensures slices from the same proxy patient volume / sequence block
do not cross train/validation/test splits.

DISCLAIMER: This project is for educational and portfolio purposes only.
It is NOT a medical device and must NOT be used for clinical diagnosis.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    CLASS_NAMES,
    PROJECT_ROOT,
    RANDOM_SEED,
    RAW_DATA_DIR,
    SPLITS_DIR,
)
from src.utils import setup_logger

logger = setup_logger("splits")


def extract_slice_number(filename: str) -> int:
    """Extract numeric slice index from filename (e.g. 'nCT1042.jpg' -> 1042)."""
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else 0


def collect_raw_dataset(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Scan raw data directory and build initial metadata table."""
    records = []
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    for class_name in CLASS_NAMES:
        class_folder = raw_dir / class_name
        if not class_folder.exists():
            logger.warning(f"Class folder not found: {class_folder}")
            continue

        for p in class_folder.iterdir():
            if p.is_file() and p.suffix.lower() in valid_exts:
                rel_path = p.relative_to(PROJECT_ROOT).as_posix()
                slice_num = extract_slice_number(p.name)
                records.append({
                    "filename": p.name,
                    "path": rel_path,
                    "label": class_name,
                    "slice_num": slice_num,
                })

    if not records:
        raise FileNotFoundError(f"No valid images found in {raw_dir}")

    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} total raw image records across {df['label'].nunique()} classes.")
    return df


def assign_patient_or_proxy_groups(df: pd.DataFrame, block_size: int = 25) -> Tuple[pd.DataFrame, str]:
    """
    Assign group_id to each image to prevent slice-level data leakage.
    Since Kaggle filenames are normalized to '<class><index>.jpg' without native DICOM patient tags,
    we group adjacent consecutive slices (standard CT scan acquisition volume size of ~25 slices)
    into proxy patient volumes: f"{label}_vol_{block_index}".
    """
    df = df.copy()
    
    # Sort slices by class and sequential slice number
    df = df.sort_values(by=["label", "slice_num"]).reset_index(drop=True)
    
    # Assign contiguous sequence block as proxy patient volume
    groups = []
    for (label), group_df in df.groupby("label", sort=False):
        n = len(group_df)
        block_ids = np.repeat(np.arange(int(np.ceil(n / block_size))), block_size)[:n]
        for b_id in block_ids:
            groups.append(f"{label}_vol_{b_id:04d}")

    df["group_id"] = groups
    method = "proxy_sequence_block_clustering"
    logger.info(f"Assigned {df['group_id'].nunique()} proxy patient volume groups (block size: {block_size} slices).")
    return df, method


def make_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train (70%), val (15%), test (15%) ensuring no group crosses splits.
    Stratified across classes by applying GroupShuffleSplit within each class.
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Split ratios must sum to 1.0"

    train_dfs = []
    val_dfs = []
    test_dfs = []

    # Stratified per-class split while respecting group boundaries
    for cls_name in CLASS_NAMES:
        cls_df = df[df["label"] == cls_name].copy().reset_index(drop=True)
        groups = cls_df["group_id"].values
        temp_ratio = val_ratio + test_ratio

        # Stage 1: Train vs (Val + Test)
        gss1 = GroupShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=seed)
        train_idx, temp_idx = next(gss1.split(cls_df, groups=groups))

        train_part = cls_df.iloc[train_idx].copy()
        temp_part = cls_df.iloc[temp_idx].copy()

        # Stage 2: Val vs Test
        val_share = val_ratio / temp_ratio
        gss2 = GroupShuffleSplit(n_splits=1, test_size=1.0 - val_share, random_state=seed)
        val_idx, test_idx = next(gss2.split(temp_part, groups=temp_part["group_id"].values))

        val_part = temp_part.iloc[val_idx].copy()
        test_part = temp_part.iloc[test_idx].copy()

        train_dfs.append(train_part)
        val_dfs.append(val_part)
        test_dfs.append(test_part)

    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Leakage check: groups must be mutually exclusive
    train_groups = set(train_df["group_id"])
    val_groups = set(val_df["group_id"])
    test_groups = set(test_df["group_id"])

    assert train_groups.isdisjoint(val_groups), "Data leakage detected: train and val share groups!"
    assert train_groups.isdisjoint(test_groups), "Data leakage detected: train and test share groups!"
    assert val_groups.isdisjoint(test_groups), "Data leakage detected: val and test share groups!"

    logger.info("Verification passed: Strict zero group overlap across train, val, and test.")
    return train_df, val_df, test_df


def save_and_log_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path = SPLITS_DIR,
) -> None:
    """Save split CSVs (path, label, split) and log detailed class distribution."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cols = ["path", "label", "split", "group_id"]

    train_df[cols].to_csv(output_dir / "train.csv", index=False)
    val_df[cols].to_csv(output_dir / "val.csv", index=False)
    test_df[cols].to_csv(output_dir / "test.csv", index=False)

    logger.info(f"Splits saved to {output_dir}")

    # Summary table
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    summary = pd.crosstab(combined["label"], combined["split"], margins=True)
    logger.info("\nClass Balance across Splits (Counts):\n" + summary.to_string())

    pct_summary = pd.crosstab(combined["label"], combined["split"], normalize="columns") * 100
    logger.info("\nClass Distribution per Split (Percentages):\n" + pct_summary.round(2).to_string())


def run_split_pipeline(raw_dir: Path = RAW_DATA_DIR, output_dir: Path = SPLITS_DIR) -> Tuple[str, int]:
    """Execute end-to-end split generation pipeline."""
    df = collect_raw_dataset(raw_dir)
    df, method = assign_patient_or_proxy_groups(df)
    train_df, val_df, test_df = make_splits(df)
    save_and_log_splits(train_df, val_df, test_df, output_dir)
    return method, len(df)


if __name__ == "__main__":
    run_split_pipeline()
