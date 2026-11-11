"""
Generate Grad-CAM comparison grids (correct vs. most-confident errors)
and save to outputs/figures/gradcam_analysis.png.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch

from src.config import (
    CHECKPOINTS_DIR,
    CLASS_NAMES,
    FIGURES_DIR,
    POWERBI_DATA_DIR,
    PROJECT_ROOT,
    get_device,
)
from src.dataset import get_eval_transforms
from src.gradcam import compute_gradcam, generate_cam_overlay, plot_gradcam_grid
from src.model import build_model
from src.utils import setup_logger

logger = setup_logger("gradcam_gen")


def run_gradcam_analysis(model_name: str = "mobilenet_v3_large"):
    device = get_device()
    pred_csv = POWERBI_DATA_DIR / "predictions.csv"
    if not pred_csv.exists():
        logger.warning(f"Predictions not found at {pred_csv}")
        return

    df_preds = pd.read_csv(pred_csv)
    sub = df_preds[(df_preds["model"] == model_name) & (df_preds["split"] == "test")].copy()
    if sub.empty:
        sub = df_preds[df_preds["model"] == model_name].copy()

    # Load model
    ckpt_path = CHECKPOINTS_DIR / f"best_{model_name}.pth"
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found at {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(model_name, num_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Pick 2 correct and 2 most confident errors
    correct_samples = sub[sub["correct"] == 1].sort_values(by="conf", ascending=False).head(2)
    error_samples = sub[sub["correct"] == 0].sort_values(by="conf", ascending=False).head(2)

    selected_rows = pd.concat([correct_samples, error_samples], ignore_index=True)

    samples = []
    eval_transform = get_eval_transforms()

    for _, row in selected_rows.iterrows():
        img_p = Path(row["path"])
        if not img_p.is_absolute():
            img_p = PROJECT_ROOT / img_p

        if not img_p.exists():
            continue

        raw_img = Image.open(img_p).convert("RGB")
        tensor = eval_transform(raw_img).unsqueeze(0).to(device)

        cam_mask, pred_idx, conf = compute_gradcam(
            model=model,
            model_name=model_name,
            image_tensor=tensor[0],
            target_category=None,
            device=device,
        )

        overlay = generate_cam_overlay(raw_img, cam_mask)

        probs = [row["p_nCT"], row["p_pCT"], row["p_NiCT"]]

        samples.append({
            "path": str(img_p),
            "raw_img": raw_img.resize((224, 224)),
            "overlay": overlay,
            "true_label": row["true_label"],
            "pred_label": row["pred_label"],
            "conf": row["conf"],
            "is_correct": bool(row["correct"] == 1),
            "probs": probs,
        })

    plot_gradcam_grid(
        samples=samples,
        output_path=FIGURES_DIR / f"gradcam_analysis_{model_name}.png",
        title=f"Grad-CAM Attention Maps: Correct vs. Top Errors ({model_name})",
    )
    # Also save primary gradcam_analysis.png
    plot_gradcam_grid(
        samples=samples,
        output_path=FIGURES_DIR / "gradcam_analysis.png",
        title=f"Grad-CAM Attention Maps: Correct vs. Top Errors ({model_name})",
    )


if __name__ == "__main__":
    run_gradcam_analysis()
