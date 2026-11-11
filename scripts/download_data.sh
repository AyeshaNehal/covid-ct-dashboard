#!/usr/bin/env bash
# scripts/download_data.sh
# Downloads the Kaggle dataset 'azaemon/preprocessed-ct-scans-for-covid19'
# and organizes it into data/raw/{nCT, pCT, NiCT}.
#
# Dataset: https://www.kaggle.com/datasets/azaemon/preprocessed-ct-scans-for-covid19
# License: CC BY 4.0 (Images from iCTCF, Ning et al. 2020)
#
# DISCLAIMER: This is a learning/portfolio project, NOT a medical device.
# Do not use for clinical diagnosis or treatment.

set -euo pipefail

DATA_DIR="data/raw"
TEMP_DIR="data/tmp_download"
DATASET_NAME="azaemon/preprocessed-ct-scans-for-covid19"

echo "=== COVID-CT Dataset Downloader ==="
echo "Target directory: $DATA_DIR"

# Check kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "ERROR: 'kaggle' CLI not found. Please install it (pip install kaggle) and ensure ~/.kaggle/kaggle.json exists."
    exit 1
fi

mkdir -p "$DATA_DIR"
mkdir -p "$TEMP_DIR"

echo "Downloading dataset $DATASET_NAME from Kaggle..."
kaggle datasets download -d "$DATASET_NAME" -p "$TEMP_DIR" --unzip

echo "Inspecting downloaded files..."
# The dataset contains preprocessed and original images.
# If both exist, we choose the original / full-resolution images as specified in the project specs,
# or map the classes to nCT, pCT, NiCT in data/raw/.
python -c "
import os, shutil, glob

temp_dir = '$TEMP_DIR'
target_dir = '$DATA_DIR'

# Locate directories matching nCT, pCT, NiCT
class_names = ['nCT', 'pCT', 'NiCT']
for c in class_names:
    target_c = os.path.join(target_dir, c)
    os.makedirs(target_c, exist_ok=True)
    
    # Search for folder named c (case-insensitive)
    found = False
    for root, dirs, files in os.walk(temp_dir):
        for d in dirs:
            if d.lower() == c.lower():
                src_path = os.path.join(root, d)
                print(f'Moving images from {src_path} to {target_c}...')
                for f in os.listdir(src_path):
                    f_src = os.path.join(src_path, f)
                    if os.path.isfile(f_src):
                        shutil.copy2(f_src, os.path.join(target_c, f))
                found = True
                break
        if found:
            break

print('Data organization complete.')
"

# Clean up temp files
rm -rf "$TEMP_DIR"

echo "Verifying classes in $DATA_DIR:"
for c in nCT pCT NiCT; do
    if [ -d "$DATA_DIR/$c" ]; then
        count=$(find "$DATA_DIR/$c" -type f | wc -l)
        echo "  $c: $count files"
    else
        echo "  WARNING: $DATA_DIR/$c not found"
    fi
done

echo "Done!"
