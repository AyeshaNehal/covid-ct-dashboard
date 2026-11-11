"""
Unit tests for CTScanDataset and transformation pipelines.
"""

from pathlib import Path
from PIL import Image
import pandas as pd
import pytest
import torch

from src.dataset import CTScanDataset, get_eval_transforms, get_train_transforms


def test_dataset_transforms_and_output_shapes(tmp_path: Path):
    # Create dummy dummy image
    img_path = tmp_path / "dummy_slice.png"
    img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    img.save(img_path)

    df = pd.DataFrame([{"path": str(img_path), "label": "pCT", "split": "train"}])

    dataset = CTScanDataset(df, transform=get_eval_transforms())
    assert len(dataset) == 1

    tensor, label_idx, p_str = dataset[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224), f"Expected (3, 224, 224), got {tensor.shape}"
    assert label_idx == 1  # pCT is index 1
    assert p_str == str(img_path)
