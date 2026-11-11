"""
Extract original images from 'archive (10).zip' into data/raw/{nCT, pCT, NiCT}.
Documenting choice: We explicitly select the 'Original CT Scans' folder rather than
the preprocessed one to retain full raw resolution and dynamic range, applying our
own reproducible, standardized ImageNet pipeline in PyTorch.

DISCLAIMER: This project is for educational and portfolio purposes only.
It is NOT a medical device and must NOT be used for clinical diagnosis.
"""

import sys
import zipfile
from pathlib import Path

ZIP_PATH = Path("archive (10).zip")
DEST_DIR = Path("data/raw")

CLASSES = {
    "Original CT Scans/nCT": DEST_DIR / "nCT",
    "Original CT Scans/pCT": DEST_DIR / "pCT",
    "Original CT Scans/NiCT": DEST_DIR / "NiCT",
}

for d in CLASSES.values():
    d.mkdir(parents=True, exist_ok=True)

print(f"Extracting Original CT Scans from {ZIP_PATH} to {DEST_DIR}...")

counts = {"nCT": 0, "pCT": 0, "NiCT": 0}

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    infolist = z.infolist()
    total = len(infolist)
    print(f"Total entries in archive: {total}")

    for idx, info in enumerate(infolist):
        filename = info.filename
        for prefix, target_dir in CLASSES.items():
            if filename.startswith(prefix) and not info.is_dir():
                basename = Path(filename).name
                if basename:
                    target_file = target_dir / basename
                    # Extract single file
                    with z.open(info) as src, open(target_file, "wb") as dst:
                        dst.write(src.read())
                    cls_key = target_dir.name
                    counts[cls_key] += 1
                    break

        if (idx + 1) % 5000 == 0 or idx == total - 1:
            print(f"Processed {idx+1}/{total} archive items... Counts so far: {counts}")

print("\nExtraction complete! Final counts in data/raw:")
for cls_name, count in counts.items():
    print(f"  {cls_name}: {count} files")
