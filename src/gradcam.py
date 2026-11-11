"""
Grad-CAM explainability pipeline for CT slice predictions.
Visualizes salient anatomical / pathological regions driving model decisions.

DISCLAIMER: This project is for educational and portfolio purposes only.
It is NOT a medical device and must NOT be used for clinical diagnosis.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.config import (
    CLASS_NAMES,
    FIGURES_DIR,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PROJECT_ROOT,
    get_device,
)
from src.dataset import get_eval_transforms
from src.model import get_target_gradcam_layer
from src.utils import add_disclaimer, setup_logger

logger = setup_logger("gradcam")


def compute_gradcam(
    model: nn.Module,
    model_name: str,
    image_tensor: torch.Tensor,
    target_category: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, int, float]:
    """
    Compute Grad-CAM heatmap for a single image tensor.

    Args:
        model: Evaluated PyTorch model
        model_name: Architecture name to identify target conv layer
        image_tensor: Normalized tensor of shape (1, 3, H, W) or (3, H, W)
        target_category: Target class index. If None, uses model's top prediction
        device: Torch device

    Returns:
        (cam_mask [H, W], predicted_category, confidence_probability)
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    if image_tensor.dim() == 3:
        input_tensor = image_tensor.unsqueeze(0).to(device)
    else:
        input_tensor = image_tensor.to(device)

    # Get model logits and predictions
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

    if target_category is None:
        target_category = pred_idx

    target_layer = get_target_gradcam_layer(model, model_name)
    targets = [ClassifierOutputTarget(target_category)]

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # Shape: (H, W), values in [0, 1]

    return grayscale_cam, pred_idx, confidence


def generate_cam_overlay(
    raw_img_pil: Image.Image,
    cam_mask: np.ndarray,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """
    Resize image to image_size and overlay Grad-CAM heatmap.
    Returns RGB uint8 numpy image.
    """
    # Resize PIL image to match CAM mask
    img_resized = raw_img_pil.convert("RGB").resize((image_size, image_size))
    img_np = np.float32(img_resized) / 255.0
    cam_resized = cv2.resize(cam_mask, (image_size, image_size))
    overlay = show_cam_on_image(img_np, cam_resized, use_rgb=True)
    return overlay


def plot_gradcam_grid(
    samples: List[dict],
    output_path: Path = FIGURES_DIR / "gradcam_grid.png",
    title: str = "Grad-CAM Explainability: Correct vs. High-Confidence Error Slices",
) -> None:
    """
    Plot a grid of slices: Raw Image, Grad-CAM Overlay, True/Pred labels, and Confidence.
    """
    n_samples = len(samples)
    if n_samples == 0:
        logger.warning("No samples provided for Grad-CAM grid.")
        return

    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3.2 * n_samples), dpi=200)
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i, s in enumerate(samples):
        # 1. Raw image
        axes[i, 0].imshow(s["raw_img"])
        axes[i, 0].set_title(f"True: {s['true_label']}\n{Path(s['path']).name}", fontsize=10)
        axes[i, 0].axis("off")

        # 2. Grad-CAM Overlay
        axes[i, 1].imshow(s["overlay"])
        pred_status = "CORRECT" if s["is_correct"] else "ERROR"
        color = "#14B8A6" if s["is_correct"] else "#EF4444"
        axes[i, 1].set_title(
            f"Pred: {s['pred_label']} ({s['conf']:.1%})\n[{pred_status}]",
            fontsize=10,
            color=color,
            fontweight="bold",
        )
        axes[i, 1].axis("off")

        # 3. Class Probabilities Bar
        bars = axes[i, 2].barh(CLASS_NAMES, s["probs"], color=["#14B8A6", "#3B82F6", "#F59E0B"])
        axes[i, 2].set_xlim(0, 1.0)
        axes[i, 2].set_xlabel("Probability", fontsize=8)
        axes[i, 2].set_title("Softmax Distribution", fontsize=9)
        axes[i, 2].tick_params(axis="both", labelsize=8)
        axes[i, 2].grid(True, axis="x", linestyle="--", alpha=0.3)

    plt.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    add_disclaimer(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved Grad-CAM grid to {output_path}")
