"""
PyTorch Dataset class and transformation pipelines for chest CT slice classification.

DISCLAIMER: This project is for educational and portfolio purposes only.
It is NOT a medical device and must NOT be used for clinical diagnosis.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from src.config import (
    CLASS_TO_IDX,
    EVAL_RESIZE,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PROJECT_ROOT,
)


def get_train_transforms(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    Data augmentations for training:
    - RandomResizedCrop(224)
    - RandomHorizontalFlip(p=0.5)
    - RandomRotation(degrees=10)
    - ColorJitter(brightness=0.15, contrast=0.15)
    - ImageNet normalization
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(
    image_size: int = IMAGE_SIZE, resize_size: int = EVAL_RESIZE
) -> transforms.Compose:
    """
    Evaluation transforms for validation, testing, and inference:
    - Resize(256)
    - CenterCrop(224)
    - ImageNet normalization
    """
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class CTScanDataset(Dataset):
    """
    PyTorch Dataset returning (tensor, label_idx, path).
    Ensures images are converted to 3-channel RGB.
    """

    def __init__(
        self,
        data_source: Union[pd.DataFrame, str, Path],
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            data_source: pandas DataFrame or path to CSV containing 'path' and 'label' columns.
            transform: torchvision Compose transform callable.
        """
        if isinstance(data_source, (str, Path)):
            self.df = pd.read_csv(data_source)
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy().reset_index(drop=True)
        else:
            raise ValueError(f"Unsupported data_source type: {type(data_source)}")

        required_cols = {"path", "label"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"Dataframe missing required columns: {required_cols - set(self.df.columns)}")

        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        img_path_str = str(row["path"])
        
        # Resolve path relative to project root if not absolute
        img_path = Path(img_path_str)
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        # Open and ensure 3-channel RGB
        with Image.open(img_path) as img:
            img = img.convert("RGB")

        if self.transform is not None:
            tensor = self.transform(img)
        else:
            # Fallback to standard eval transforms
            tensor = get_eval_transforms()(img)

        label_name = str(row["label"])
        label_idx = CLASS_TO_IDX[label_name]

        return tensor, label_idx, img_path_str
