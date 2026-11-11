"""
Exploratory Data Analysis (EDA) Script.
Analyzes class counts, image geometry, integrity/corruption, byte duplicates,
perceptual hash clustering, and filename metadata.
Exports powerbi/data/dataset_summary.csv and publication figures.

DISCLAIMER: This project is for educational and portfolio purposes only.
It is NOT a medical device and must NOT be used for clinical diagnosis.
"""

import hashlib
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.config import (
    CLASS_NAMES,
    FIGURES_DIR,
    POWERBI_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
)
from src.utils import add_disclaimer, setup_logger

logger = setup_logger("eda")


def compute_md5(file_path: Path) -> str:
    """Compute MD5 hash of raw file bytes for exact duplicate detection."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compute_dhash(img_pil: Image.Image, hash_size: int = 8) -> str:
    """Compute 64-bit difference hash for perceptual similarity / near duplicates."""
    img_gray = img_pil.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(img_gray)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return hashlib.md5(diff.tobytes()).hexdigest()[:12]


def run_eda(raw_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    logger.info(f"Starting EDA on raw dataset at: {raw_dir}")

    records = []
    seen_md5 = set()
    md5_duplicates = 0
    corrupt_files = 0

    class_stats = {
        c: {
            "total_slices": 0,
            "widths": [],
            "heights": [],
            "brightness": [],
            "corrupt": 0,
            "duplicates": 0,
            "dhashes": set(),
        }
        for c in CLASS_NAMES
    }

    sample_images = {c: [] for c in CLASS_NAMES}

    for c in CLASS_NAMES:
        class_folder = raw_dir / c
        if not class_folder.exists():
            logger.warning(f"Class folder missing: {class_folder}")
            continue

        files = sorted(list(class_folder.glob("*.*")))
        class_stats[c]["total_slices"] = len(files)
        logger.info(f"Scanning {c}: {len(files)} files...")

        # Process all files
        for p in tqdm(files, desc=f"Auditing {c}"):
            # MD5 check
            try:
                f_md5 = compute_md5(p)
                if f_md5 in seen_md5:
                    class_stats[c]["duplicates"] += 1
                    md5_duplicates += 1
                else:
                    seen_md5.add(f_md5)
            except Exception:
                class_stats[c]["corrupt"] += 1
                corrupt_files += 1
                continue

            # Image open check
            try:
                with Image.open(p) as img:
                    w, h = img.size
                    mode = img.mode
                    class_stats[c]["widths"].append(w)
                    class_stats[c]["heights"].append(h)

                    # Brightness
                    gray = img.convert("L")
                    arr = np.asarray(gray, dtype=np.float32) / 255.0
                    b_val = float(arr.mean())
                    class_stats[c]["brightness"].append(b_val)

                    # dHash
                    dh = compute_dhash(img)
                    class_stats[c]["dhashes"].add(dh)

                    # Save up to 5 sample images for plotting
                    if len(sample_images[c]) < 5:
                        sample_images[c].append((p.name, img.copy()))

                    records.append({
                        "filename": p.name,
                        "class": c,
                        "width": w,
                        "height": h,
                        "mode": mode,
                        "brightness": round(b_val, 4),
                        "md5": f_md5,
                        "dhash": dh,
                    })
            except Exception as e:
                logger.error(f"Corrupt image {p}: {e}")
                class_stats[c]["corrupt"] += 1
                corrupt_files += 1

    df_records = pd.DataFrame(records)
    logger.info(f"Audited {len(df_records)} valid images. Corrupt: {corrupt_files}, Exact Duplicates: {md5_duplicates}")

    # Build dataset_summary.csv
    summary_rows = []
    total_slices_all = 0
    total_groups_all = 0
    total_corrupt_all = 0
    total_dup_all = 0

    for c in CLASS_NAMES:
        st = class_stats[c]
        n_slices = st["total_slices"]
        n_groups = len(st["dhashes"])
        m_w = float(np.mean(st["widths"])) if st["widths"] else 0.0
        m_h = float(np.mean(st["heights"])) if st["heights"] else 0.0
        m_b = float(np.mean(st["brightness"])) if st["brightness"] else 0.0

        total_slices_all += n_slices
        total_groups_all += n_groups
        total_corrupt_all += st["corrupt"]
        total_dup_all += st["duplicates"]

        summary_rows.append({
            "class": c,
            "total_slices": n_slices,
            "unique_groups": n_groups,
            "mean_width": round(m_w, 1),
            "mean_height": round(m_h, 1),
            "mean_brightness": round(m_b, 4),
            "corrupt_images": st["corrupt"],
            "exact_duplicates": st["duplicates"],
        })

    # Total row
    summary_rows.append({
        "class": "Total",
        "total_slices": total_slices_all,
        "unique_groups": total_groups_all,
        "mean_width": round(float(df_records["width"].mean()), 1),
        "mean_height": round(float(df_records["height"].mean()), 1),
        "mean_brightness": round(float(df_records["brightness"].mean()), 4),
        "corrupt_images": total_corrupt_all,
        "exact_duplicates": total_dup_all,
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = POWERBI_DATA_DIR / "dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved dataset summary to {summary_path}")
    print("\n=== DATASET SUMMARY AUDIT ===")
    print(summary_df.to_string(index=False))

    # Plot 1: Sample Grid (3 rows x 5 columns)
    fig, axes = plt.subplots(3, 5, figsize=(15, 9), dpi=200)
    for row_idx, c in enumerate(CLASS_NAMES):
        samples = sample_images[c]
        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            if col_idx < len(samples):
                fname, img = samples[col_idx]
                ax.imshow(img, cmap="bone" if img.mode == "L" else None)
                ax.set_title(f"{c}\n{fname}", fontsize=9)
            ax.axis("off")

    plt.suptitle("Chest CT Slice Exemplars by Pathology Class (Raw Dataset)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    add_disclaimer(fig)
    sample_grid_path = FIGURES_DIR / "eda_samples_grid.png"
    plt.savefig(sample_grid_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved sample grid to {sample_grid_path}")

    # Plot 2: Brightness Distribution by Class
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    for c in CLASS_NAMES:
        c_bright = df_records[df_records["class"] == c]["brightness"]
        ax.hist(c_bright, bins=30, alpha=0.6, label=c, density=True)

    ax.set_title("Slice Brightness (Intensity) Probability Density by Class", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Grayscale Intensity (0.0 = Black, 1.0 = White)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    add_disclaimer(fig)
    bright_dist_path = FIGURES_DIR / "eda_brightness_distribution.png"
    plt.savefig(bright_dist_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved brightness distribution to {bright_dist_path}")

    return df_records


if __name__ == "__main__":
    run_eda()
