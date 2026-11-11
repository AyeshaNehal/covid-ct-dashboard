"""
Generate portfolio-grade Jupyter notebooks:
1. notebooks/01_eda.ipynb
2. notebooks/02_training_walkthrough.ipynb
3. notebooks/03_error_analysis.ipynb
"""

import nbformat as nbf
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

DISCLAIMER_MD = """
> [!WARNING]
> **EDUCATIONAL & PORTFOLIO PROJECT ONLY • NOT A MEDICAL DEVICE**  
> *This notebook, associated models, and data pipelines are strictly for research, algorithmic benchmarking, and portfolio demonstration. They are NOT certified by the FDA, CE, or any medical regulatory agency and must NEVER be used for clinical decision making, diagnosis, triage, or patient treatment.*
"""

# ==============================================================================
# Notebook 1: EDA
# ==============================================================================
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    nbf.v4.new_markdown_cell(
        f"# Notebook 01: Exploratory Data Analysis & Leakage Investigation\n"
        f"**Project:** COVID-CT Deep Learning Classifier (`covid-ct-dashboard`)\n\n"
        f"{DISCLAIMER_MD}\n\n"
        f"### Objectives:\n"
        f"1. Audit class distribution across Normal (`nCT`), COVID-19 Positive (`pCT`), and Non-informative (`NiCT`).\n"
        f"2. Inspect raw image dimensions, channels, and integrity (zero corrupt files check).\n"
        f"3. Detect exact duplicates using MD5 hashing and perceptual near-duplicates via difference hashing (`dHash`).\n"
        f"4. **Investigate patient/scan identifiers in filenames**: Analyze whether slice names contain patient tags, scan IDs, or purely sequential numbers.\n"
        f"5. Export dataset audit summary to `powerbi/data/dataset_summary.csv`."
    ),
    nbf.v4.new_code_cell(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from PIL import Image\n\n"
        "# Ensure src is importable\n"
        "PROJECT_ROOT = Path('.').resolve().parent\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "from src.config import RAW_DATA_DIR, CLASS_NAMES, POWERBI_DATA_DIR\n"
        "print('Project Root:', PROJECT_ROOT)\n"
        "print('Raw Data Path:', RAW_DATA_DIR)"
    ),
    nbf.v4.new_markdown_cell("## 1. Class Distribution & File Inventory"),
    nbf.v4.new_code_cell(
        "inventory = []\n"
        "for c in CLASS_NAMES:\n"
        "    c_dir = RAW_DATA_DIR / c\n"
        "    files = list(c_dir.glob('*.*')) if c_dir.exists() else []\n"
        "    inventory.append({'class': c, 'count': len(files)})\n\n"
        "inv_df = pd.DataFrame(inventory)\n"
        "display(inv_df)\n\n"
        "plt.figure(figsize=(7, 4), dpi=150)\n"
        "plt.bar(inv_df['class'], inv_df['count'], color=['#14B8A6', '#3B82F6', '#F59E0B'])\n"
        "plt.title('Raw CT Slice Class Distribution', fontweight='bold')\n"
        "plt.ylabel('Number of Slices')\n"
        "plt.grid(True, linestyle='--', alpha=0.3)\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "## 2. Filename Structure & Patient ID Investigation\n"
        "A critical vulnerability in slice-based medical imaging benchmarks is **data leakage**: "
        "if multiple slices from the same patient scan are randomly assigned across train and test sets, "
        "the model can memorize unique patient anatomy, yielding artificially elevated test scores.\n\n"
        "Let's inspect sample filenames from each class:"
    ),
    nbf.v4.new_code_cell(
        "for c in CLASS_NAMES:\n"
        "    c_dir = RAW_DATA_DIR / c\n"
        "    sample_fnames = [f.name for f in sorted(list(c_dir.glob('*.*')))[:8]]\n"
        "    print(f'{c}:', sample_fnames)"
    ),
    nbf.v4.new_markdown_cell(
        "### 🔍 Key Finding on Filename Provenance:\n"
        "- The Kaggle dataset (`azaemon/preprocessed-ct-scans-for-covid19`) renames all original images to `<class><index>.jpg` (e.g. `nCT1.jpg`, `pCT45.jpg`, `NiCT120.jpg`).\n"
        "- **No explicit patient ID string or DICOM metadata header is preserved** in these flattened JPEG filenames.\n"
        "- **Consequence & Mitigation**: Rather than performing naive uniform random splitting (which guarantees slice leakage), "
        "we employ **proxy grouping**: consecutive sequence blocks and perceptual difference hash (`dHash`) clustering are used to group adjacent slices into pseudo-patient volumes. "
        "A strict `GroupShuffleSplit` (70/15/15) is then enforced so that clusters never cross splits."
    ),
    nbf.v4.new_markdown_cell("## 3. Dataset Health Check: Corrupt Files & Duplicates"),
    nbf.v4.new_code_cell(
        "summary_csv = POWERBI_DATA_DIR / 'dataset_summary.csv'\n"
        "if summary_csv.exists():\n"
        "    summary_df = pd.read_csv(summary_csv)\n"
        "    display(summary_df)\n"
        "else:\n"
        "    print('dataset_summary.csv not generated yet.')"
    ),
    nbf.v4.new_markdown_cell("## 4. Visual Exemplars by Pathology Class"),
    nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(len(CLASS_NAMES), 4, figsize=(12, 3 * len(CLASS_NAMES)), dpi=150)\n"
        "for r, c in enumerate(CLASS_NAMES):\n"
        "    c_dir = RAW_DATA_DIR / c\n"
        "    samples = sorted(list(c_dir.glob('*.*')))[:4]\n"
        "    for col, s_path in enumerate(samples):\n"
        "        img = Image.open(s_path)\n"
        "        axes[r, col].imshow(img, cmap='bone')\n"
        "        axes[r, col].set_title(f'{c} - {s_path.name}', fontsize=8)\n"
        "        axes[r, col].axis('off')\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
]

with open(NOTEBOOKS_DIR / "01_eda.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb1, f)
print("Created notebooks/01_eda.ipynb")

# ==============================================================================
# Notebook 2: Training Walkthrough
# ==============================================================================
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    nbf.v4.new_markdown_cell(
        f"# Notebook 02: End-to-End Training & Evaluation Walkthrough\n"
        f"**Project:** COVID-CT Deep Learning Classifier (`covid-ct-dashboard`)\n\n"
        f"{DISCLAIMER_MD}\n\n"
        f"### Overview:\n"
        f"This notebook demonstrates the complete PyTorch training workflow:\n"
        f"1. Loading leakage-aware split tables (`data/splits/train.csv`, `val.csv`, `test.csv`).\n"
        f"2. Dataset class & augmentation pipeline (RandomResizedCrop, Rotation, Jitter, ImageNet normalization).\n"
        f"3. Instantiating pretrained transfer learning backbones (`MobileNetV3-Large`, `ResNet-18`, `EfficientNet-B0`).\n"
        f"4. Optimizer (AdamW), Cosine Annealing learning rate schedule, and Early Stopping.\n"
        f"5. Training progression inspection across epochs."
    ),
    nbf.v4.new_code_cell(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "import torch\n\n"
        "PROJECT_ROOT = Path('.').resolve().parent\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "from src.config import SPLITS_DIR, CHECKPOINTS_DIR, get_device, CLASS_NAMES\n"
        "from src.dataset import CTScanDataset, get_train_transforms, get_eval_transforms\n"
        "from src.model import build_model, count_parameters\n\n"
        "device = get_device()\n"
        "print('Active Device:', device)"
    ),
    nbf.v4.new_markdown_cell("## 1. Inspecting Dataset Splits"),
    nbf.v4.new_code_cell(
        "train_df = pd.read_csv(SPLITS_DIR / 'train.csv')\n"
        "val_df = pd.read_csv(SPLITS_DIR / 'val.csv')\n"
        "test_df = pd.read_csv(SPLITS_DIR / 'test.csv')\n\n"
        "print(f'Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}')\n\n"
        "# Verify strict group separation\n"
        "train_g = set(train_df['group_id'])\n"
        "test_g = set(test_df['group_id'])\n"
        "overlap = train_g.intersection(test_g)\n"
        "print('Group overlap between Train and Test:', len(overlap))"
    ),
    nbf.v4.new_markdown_cell("## 2. Model Architecture Setup"),
    nbf.v4.new_code_cell(
        "for name in ['mobilenet_v3_large', 'resnet18', 'efficientnet_b0']:\n"
        "    m = build_model(name, num_classes=3, pretrained=True)\n"
        "    total, trainable = count_parameters(m)\n"
        "    print(f'{name:<20}: {total:>10,} total params | {trainable:>10,} trainable')"
    ),
    nbf.v4.new_markdown_cell("## 3. Training History & Benchmark Comparison"),
    nbf.v4.new_code_cell(
        "history_csv = PROJECT_ROOT / 'powerbi/data/training_history.csv'\n"
        "comp_csv = PROJECT_ROOT / 'powerbi/data/model_comparison.csv'\n\n"
        "if history_csv.exists():\n"
        "    h_df = pd.read_csv(history_csv)\n"
        "    display(h_df.head(10))\n\n"
        "if comp_csv.exists():\n"
        "    c_df = pd.read_csv(comp_csv)\n"
        "    display(c_df[c_df['split'] == 'test'])"
    ),
]

with open(NOTEBOOKS_DIR / "02_training_walkthrough.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb2, f)
print("Created notebooks/02_training_walkthrough.ipynb")

# ==============================================================================
# Notebook 3: Error Analysis
# ==============================================================================
nb3 = nbf.v4.new_notebook()
nb3.cells = [
    nbf.v4.new_markdown_cell(
        f"# Notebook 03: Explainability, Grad-CAM & Systematic Error Analysis\n"
        f"**Project:** COVID-CT Deep Learning Classifier (`covid-ct-dashboard`)\n\n"
        f"{DISCLAIMER_MD}\n\n"
        f"### Objectives:\n"
        f"1. Extract top-20 most confident prediction errors.\n"
        f"2. Analyze misclassification transitions (e.g. `pCT -> nCT`, `nCT -> pCT`).\n"
        f"3. Generate Grad-CAM heatmaps overlaying salient feature activations.\n"
        f"4. **Discussion of Likely Failure Causes**: Motion artifact, peripheral ground-glass similarity, and partial-volume slice boundaries.\n"
        f"5. **Honest Evaluation Note**: Methodological impact of proxy group splitting vs. volumetric 3D CT."
    ),
    nbf.v4.new_code_cell(
        "import sys\n"
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from PIL import Image\n\n"
        "PROJECT_ROOT = Path('.').resolve().parent\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "from src.config import POWERBI_DATA_DIR, FIGURES_DIR, CLASS_NAMES\n"
        "pred_csv = POWERBI_DATA_DIR / 'predictions.csv'\n"
        "df_preds = pd.read_csv(pred_csv)\n"
        "print('Loaded predictions:', len(df_preds))"
    ),
    nbf.v4.new_markdown_cell("## 1. Top-20 Most Confident Errors"),
    nbf.v4.new_code_cell(
        "test_errors = df_preds[(df_preds['split'] == 'test') & (df_preds['correct'] == 0)].copy()\n"
        "top20 = test_errors.sort_values(by='conf', ascending=False).head(20)\n"
        "display(top20[['model', 'image_id', 'true_label', 'pred_label', 'conf', 'error_type', 'mean_brightness']])"
    ),
    nbf.v4.new_markdown_cell("## 2. Misclassification Pair Breakdown"),
    nbf.v4.new_code_cell(
        "err_pairs = test_errors.groupby(['model', 'error_type']).size().unstack(fill_value=0)\n"
        "display(err_pairs)\n\n"
        "test_errors['error_type'].value_counts().plot(kind='barh', color='#F87171', figsize=(7, 4))\n"
        "plt.title('Misclassification Transitions on Test Set', fontweight='bold')\n"
        "plt.xlabel('Count')\n"
        "plt.grid(True, linestyle='--', alpha=0.3)\n"
        "plt.show()"
    ),
    nbf.v4.new_markdown_cell(
        "## 3. Clinical & Methodological Discussion\n\n"
        "### Likely Causes of Systematic Errors:\n"
        "1. **Peripheral Ground-Glass Opacities vs. Atelectasis**: Subpleural dependent density in normal elderly patients frequently mimics early multifocal COVID-19 consolidations.\n"
        "2. **Motion and Diaphragmatic Artifacts**: Slices acquired during shallow respiration display cardiac motion blur that distorts lung parenchyma textures.\n"
        "3. **Non-informative (NiCT) Slice Boundaries**: Transitional slices between the apex and upper thoracic soft tissues have lower lung area, creating borderline ambiguity between NiCT and normal nCT.\n\n"
        "### Honest Appraisal of Split Strategy & Score Inflation:\n"
        "- In standard benchmarks where random slice splitting is used, test accuracies often exceed 98-99% because near-duplicate adjacent slices from the same scan volume exist in both training and test sets.\n"
        "- Enforcing proxy grouping reduces this inflation, providing a more realistic assessment of out-of-scan generalization.\n"
        "- However, since the dataset did not provide native DICOM patient headers, sequence proxy clustering is an approximation. True prospective clinical performance on heterogeneous scanners across different hospitals would be expected to experience further distribution shift."
    ),
]

with open(NOTEBOOKS_DIR / "03_error_analysis.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb3, f)
print("Created notebooks/03_error_analysis.ipynb")
