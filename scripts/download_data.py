"""
Download and unpack Kaggle dataset 'azaemon/preprocessed-ct-scans-for-covid19'.
Organizes slices into data/raw/{nCT, pCT, NiCT}.

DISCLAIMER: This project is for educational and portfolio purposes only.
It is NOT a medical device and must NOT be used for clinical diagnosis.
"""

import os
import shutil
import sys
import zipfile
from pathlib import Path


def download_dataset():
    data_dir = Path("data/raw")
    temp_dir = Path("data/tmp_download")
    dataset_name = "azaemon/preprocessed-ct-scans-for-covid19"

    print("=== COVID-CT Dataset Downloader ===")
    print(f"Target directory: {data_dir.resolve()}")

    try:
        import kaggle
    except ImportError:
        print("ERROR: kaggle package not found. Run: pip install kaggle")
        sys.exit(1)

    temp_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_name}...")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_name, path=str(temp_dir), unzip=True)

    print("Inspecting downloaded files...")
    classes = ["nCT", "pCT", "NiCT"]
    
    # Check for original vs preprocessed folders
    # Note: Kaggle dataset may have folders like 'preprocessed_images' and 'original_images' or directly nCT, pCT, NiCT
    # We choose original images if available, or whatever classes are provided.
    found_map = {}
    for root, dirs, files in os.walk(temp_dir):
        for d in dirs:
            for c in classes:
                if d.lower() == c.lower() and c not in found_map:
                    # Prefer original over preprocessed if both exist
                    full_p = Path(root) / d
                    if "preprocessed" not in str(full_p).lower() or c not in found_map:
                        found_map[c] = full_p

    print("Found class directories:", found_map)

    for c in classes:
        target_class_dir = data_dir / c
        target_class_dir.mkdir(parents=True, exist_ok=True)
        if c in found_map:
            src_dir = found_map[c]
            copied = 0
            for item in src_dir.iterdir():
                if item.is_file() and item.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                    shutil.copy2(item, target_class_dir / item.name)
                    copied += 1
            print(f"Copied {copied} images to {target_class_dir}")
        else:
            print(f"WARNING: Class directory {c} not found in downloaded archive!")

    # Clean up temp
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    print("\nVerification of data/raw:")
    for c in classes:
        p = data_dir / c
        count = len(list(p.glob("*.*"))) if p.exists() else 0
        print(f"  {c}: {count} images")


if __name__ == "__main__":
    download_dataset()
