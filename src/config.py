"""
Project configuration, paths, constants, and hyperparameters.

DISCLAIMER: This project is for educational and portfolio purposes only.
It is NOT a medical device and must NOT be used for clinical diagnosis.
"""

from pathlib import Path
import torch

# Base Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
FIGURES_DIR = OUTPUTS_DIR / "figures"
LOGS_DIR = OUTPUTS_DIR / "logs"

POWERBI_DIR = PROJECT_ROOT / "powerbi"
POWERBI_DATA_DIR = POWERBI_DIR / "data"
POWERBI_DOCS_DIR = POWERBI_DIR / "docs"
POWERBI_THEME_DIR = POWERBI_DIR / "theme"
POWERBI_SCREENSHOTS_DIR = POWERBI_DIR / "screenshots"

# Ensure runtime output directories exist
for directory in [
    SPLITS_DIR,
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    POWERBI_DATA_DIR,
    POWERBI_DOCS_DIR,
    POWERBI_THEME_DIR,
    POWERBI_SCREENSHOTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# Classes and Labels
# nCT = Normal, pCT = COVID-19 Positive, NiCT = Non-informative CT
CLASS_NAMES = ["nCT", "pCT", "NiCT"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}
CLASS_DESCRIPTIONS = {
    "nCT": "Normal Chest CT",
    "pCT": "COVID-19 Positive Chest CT",
    "NiCT": "Non-informative CT Scan",
}

# Image Preprocessing & Normalization (ImageNet stats)
IMAGE_SIZE = 224
EVAL_RESIZE = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Reproducibility
RANDOM_SEED = 42

# Training Defaults
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 25
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
SUPPORTED_MODELS = ["mobilenet_v3_large", "resnet18", "efficientnet_b0"]

# Medical Disclaimer
DISCLAIMER_TEXT = (
    "DISCLAIMER: This project is strictly for research, portfolio demonstration, "
    "and educational purposes. It is NOT a certified medical device and must NOT "
    "be used for medical diagnosis, treatment planning, or clinical decision making."
)


def get_device() -> torch.device:
    """
    Auto-detect device in precedence order:
    1. MPS (Apple Silicon)
    2. CUDA (NVIDIA GPU)
    3. CPU (Fallback)
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
