"""
Unit tests for model architecture factory and forward pass.
"""

import pytest
import torch
from src.config import SUPPORTED_MODELS
from src.model import build_model, count_parameters


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_model_build_and_parameters(model_name: str):
    model = build_model(model_name, num_classes=3, pretrained=False)
    total_params, trainable_params = count_parameters(model)
    assert total_params > 0, "Total parameters must be positive"
    assert trainable_params > 0, "Trainable parameters must be positive"
    assert trainable_params == total_params, "All parameters should be trainable for full fine-tuning"


@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_model_forward_pass_shape(model_name: str):
    model = build_model(model_name, num_classes=3, pretrained=False)
    model.eval()
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (batch_size, 3), f"Expected shape ({batch_size}, 3), got {output.shape}"
