"""
Model architecture factory supporting MobileNetV3-Large, ResNet-18, and EfficientNet-B0.
Pretrained backbones with 3-class output heads for full fine-tuning.

DISCLAIMER: This project is for educational and portfolio purposes only.
It is NOT a medical device and must NOT be used for clinical diagnosis.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torchvision.models as models

from src.config import CLASS_NAMES, SUPPORTED_MODELS


def build_model(name: str, num_classes: int = len(CLASS_NAMES), pretrained: bool = True) -> nn.Module:
    """
    Build and return a PyTorch model with a customized 3-class classification head.

    Args:
        name: Name of architecture ('mobilenet_v3_large', 'resnet18', 'efficientnet_b0')
        num_classes: Number of output classes (default: 3)
        pretrained: Whether to load ImageNet pre-trained weights

    Returns:
        nn.Module model ready for full fine-tuning.
    """
    name_clean = name.lower().strip()
    if name_clean not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model architecture '{name}'. Choose from: {SUPPORTED_MODELS}"
        )

    if name_clean == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif name_clean == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif name_clean == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a PyTorch model.

    Returns:
        (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_target_gradcam_layer(model: nn.Module, model_name: str) -> nn.Module:
    """
    Return the recommended target convolutional layer for Grad-CAM.
    """
    name_clean = model_name.lower().strip()
    if name_clean == "resnet18":
        return model.layer4[-1]
    elif name_clean == "mobilenet_v3_large":
        return model.features[-1]
    elif name_clean == "efficientnet_b0":
        return model.features[-1]
    else:
        raise ValueError(f"Unknown target layer for model: {model_name}")
