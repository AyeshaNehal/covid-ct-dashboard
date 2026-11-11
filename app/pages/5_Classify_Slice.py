"""
Page 5: Live CT Slice Classification & Grad-CAM Explainability Demo
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import torch
import numpy as np
from PIL import Image
import plotly.express as px

from app.components.theme import (
    apply_custom_css,
    render_disclaimer,
    render_header,
    TEAL,
    BLUE,
    AMBER,
    CORAL,
    TEXT_WHITE,
)
from src.config import CHECKPOINTS_DIR, CLASS_NAMES, CLASS_DESCRIPTIONS, PROJECT_ROOT, get_device
from src.dataset import get_eval_transforms
from src.model import build_model
from src.gradcam import compute_gradcam, generate_cam_overlay

st.set_page_config(page_title="Classify CT Slice - COVID-CT", page_icon="🔬", layout="wide")
apply_custom_css()
render_header("Live CT Slice Classification & Grad-CAM", "Interactive deep learning inference with real-time salience heatmaps", "🔬")
render_disclaimer()

@st.cache_resource
def load_resnet_model():
    device = get_device()
    model = build_model("resnet18", num_classes=len(CLASS_NAMES), pretrained=False)
    ckpt_path = CHECKPOINTS_DIR / "best_resnet18.pth"
    if not ckpt_path.exists():
        return None, device
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device

model, device = load_resnet_model()

if model is None:
    st.error("Model checkpoint `outputs/checkpoints/best_resnet18.pth` not found. Please evaluate first.")
    st.stop()

eval_transform = get_eval_transforms()

col_input, col_results = st.columns([1, 1])

input_image = None
image_source = None

with col_input:
    st.markdown("### 📥 Select Input Image")
    input_mode = st.radio("Choose Input Method:", ["Select from Held-Out Test Set", "Upload Custom CT Slice (.jpg / .png)"], horizontal=True)

    if input_mode == "Upload Custom CT Slice (.jpg / .png)":
        uploaded_file = st.file_uploader("Upload Chest CT Axial Slice:", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file).convert("RGB")
            image_source = f"Uploaded: {uploaded_file.name}"
    else:
        test_csv = Path("data/splits/test.csv")
        if test_csv.exists():
            test_df = pd.read_csv(test_csv)
            sample_options = [
                f"{row['label']} | Slice: {row['path']}"
                for _, row in test_df.sample(20, random_state=42).iterrows()
            ]
            chosen_sample = st.selectbox("Choose a sample slice from test set:", sample_options)
            chosen_rel_path = chosen_sample.split("| Slice: ")[-1].strip()
            chosen_full_path = PROJECT_ROOT / chosen_rel_path
            if chosen_full_path.exists():
                input_image = Image.open(chosen_full_path).convert("RGB")
                image_source = f"Test Sample: {chosen_rel_path}"

    if input_image is not None:
        st.image(input_image.resize((260, 260)), caption=image_source, use_container_width=False)

with col_results:
    st.markdown("### 🎯 Model Predictions & Probabilities")
    if input_image is not None:
        tensor_img = eval_transform(input_image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor_img)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_label = CLASS_NAMES[pred_idx]
        pred_conf = probs[pred_idx]

        color_map = {"nCT": TEAL, "pCT": CORAL, "NiCT": AMBER}
        badge_color = color_map.get(pred_label, TEAL)

        st.markdown(
            f"""
            <div class="metric-card" style="border-left: 6px solid {badge_color};">
                <div class="metric-title">Predicted Classification</div>
                <div class="metric-value" style="color: {badge_color};">{pred_label} ({pred_conf:.1%})</div>
                <div style="color: #94A3B8; font-size: 0.9rem;">{CLASS_DESCRIPTIONS[pred_label]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        prob_df = pd.DataFrame({
            "Class": [f"{c} ({CLASS_DESCRIPTIONS[c]})" for c in CLASS_NAMES],
            "Probability": probs,
            "Color": [color_map[c] for c in CLASS_NAMES],
        })

        fig_bars = px.bar(
            prob_df,
            x="Probability",
            y="Class",
            orientation="h",
            text=[f"{p:.1%}" for p in probs],
            range_x=[0, 1.05],
            color="Class",
            color_discrete_sequence=[TEAL, CORAL, AMBER],
        )
        fig_bars.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#132D4A",
            font=dict(color=TEXT_WHITE),
            height=200,
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_bars, use_container_width=True)

        st.markdown("### 🔬 Real-Time Grad-CAM Salience")
        with st.spinner("Generating Grad-CAM heatmap..."):
            cam_mask, _, _ = compute_gradcam(model=model, model_name="resnet18", image_tensor=tensor_img, device=device)
            overlay = generate_cam_overlay(input_image.resize((224, 224)), cam_mask)

        c_cam1, c_cam2 = st.columns(2)
        with c_cam1:
            st.image(cam_mask, caption="Activation Intensity (Layer 4)", clamp=True, use_container_width=True)
        with c_cam2:
            st.image(overlay, caption="Diagnostic Anatomical Overlay", use_container_width=True)
    else:
        st.info("👈 Select or upload a chest CT slice on the left to trigger real-time inference.")
