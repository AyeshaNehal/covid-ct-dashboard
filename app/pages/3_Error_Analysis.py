"""
Page 3: Systematic Error Analysis & Failure Mode Taxonomy
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
from PIL import Image
import torch

from app.components.charts import plot_confidence_distribution_plotly
from app.components.theme import (
    apply_custom_css,
    render_disclaimer,
    render_header,
    render_kpi,
    TEAL,
    AMBER,
    CORAL,
)
from src.config import CHECKPOINTS_DIR, CLASS_NAMES, PROJECT_ROOT, get_device
from src.dataset import get_eval_transforms
from src.gradcam import compute_gradcam, generate_cam_overlay
from src.model import build_model

st.set_page_config(page_title="Error Analysis - COVID-CT", page_icon="⚠️", layout="wide")
apply_custom_css()
render_header("Systematic Error Analysis & Failure Modes", "Filterable misclassifications, confidence distributions, and Grad-CAM inspection", "⚠️")
render_disclaimer()

@st.cache_data
def load_error_data():
    preds = pd.read_csv("powerbi/data/predictions.csv")
    test_preds = preds[preds["split"] == "test"].copy()
    errors = test_preds[test_preds["correct"] == 0].copy()
    return test_preds, errors

test_preds, errors = load_error_data()

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_kpi("Total Test Errors", f"{len(errors)}", subtext=f"Error rate: {len(errors)/len(test_preds):.2%}", val_color=CORAL)
with c2:
    high_conf_errs = len(errors[errors["conf"] >= 0.90])
    render_kpi("High-Confidence Errors", f"{high_conf_errs}", subtext="Confidence ≥ 90%", val_color=AMBER)
with c3:
    render_kpi("Mean Error Confidence", f"{errors['conf'].mean():.2%}", subtext=f"Correct Avg Conf: {test_preds[test_preds['correct']==1]['conf'].mean():.2%}", val_color=TEAL)
with c4:
    top_pair = errors["error_type"].value_counts().index[0]
    top_pair_cnt = errors["error_type"].value_counts().iloc[0]
    render_kpi("Most Frequent Failure", f"{top_pair}", subtext=f"{top_pair_cnt} occurrences", val_color=CORAL)

st.plotly_chart(plot_confidence_distribution_plotly(test_preds), use_container_width=True)

st.markdown("### 🔍 Interactive Misclassification Filter")
f1, f2, f3 = st.columns(3)
with f1:
    selected_true = st.multiselect("Filter by True Ground Truth:", options=CLASS_NAMES, default=CLASS_NAMES)
with f2:
    selected_pred = st.multiselect("Filter by Predicted Class:", options=CLASS_NAMES, default=CLASS_NAMES)
with f3:
    conf_range = st.slider("Filter by Confidence Range:", min_value=0.30, max_value=1.0, value=(0.40, 1.0))

filtered_errors = errors[
    (errors["true_label"].isin(selected_true))
    & (errors["pred_label"].isin(selected_pred))
    & (errors["conf"] >= conf_range[0])
    & (errors["conf"] <= conf_range[1])
].sort_values(by="conf", ascending=False)

st.caption(f"Showing {len(filtered_errors)} misclassified slices matching criteria.")

st.dataframe(
    filtered_errors[["path", "true_label", "pred_label", "conf", "error_type", "width", "height", "mean_brightness"]].style.format({"conf": "{:.2%}", "mean_brightness": "{:.3f}"}),
    use_container_width=True,
    height=260,
)

st.markdown("---")
st.markdown("### 🔬 Inspect Single Misclassified Slice with Grad-CAM")

if len(filtered_errors) > 0:
    slice_options = [f"#{idx} | True: {row['true_label']} -> Pred: {row['pred_label']} ({row['conf']:.1%}) | {row['path']}" for idx, row in filtered_errors.reset_index().iterrows()]
    selected_slice_str = st.selectbox("Select Slice to Inspect:", slice_options)
    selected_idx = int(selected_slice_str.split("|")[0].replace("#", "").strip())
    selected_row = filtered_errors.iloc[selected_idx]

    img_path = PROJECT_ROOT / selected_row["path"]
    if img_path.exists():
        pil_img = Image.open(img_path).convert("RGB")
        eval_tf = get_eval_transforms()
        device = get_device()

        model = build_model("resnet18", num_classes=3, pretrained=False)
        ckpt = torch.load(CHECKPOINTS_DIR / "best_resnet18.pth", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        tensor_img = eval_tf(pil_img).unsqueeze(0).to(device)
        cam_mask, _, _ = compute_gradcam(model=model, model_name="resnet18", image_tensor=tensor_img, device=device)
        overlay_cam = generate_cam_overlay(pil_img.resize((224, 224)), cam_mask)

        i1, i2, i3 = st.columns(3)
        with i1:
            st.image(pil_img.resize((300, 300)), caption=f"Original Slice: {selected_row['path']}", use_container_width=True)
        with i2:
            st.image(cam_mask, caption=f"Grad-CAM Salience Map (Layer4)", clamp=True, use_container_width=True)
        with i3:
            st.image(overlay_cam, caption=f"Overlay: True {selected_row['true_label']} -> Pred {selected_row['pred_label']} ({selected_row['conf']:.1%})", use_container_width=True)
    else:
        st.error(f"Image not found at {img_path}")
else:
    st.info("No errors match the current filter criteria.")
