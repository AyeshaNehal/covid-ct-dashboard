"""
Main Entry point for the COVID-CT Companion Analytics Web App.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from app.components.theme import (
    apply_custom_css,
    render_disclaimer,
    render_header,
    render_kpi,
    TEAL,
    BLUE,
    AMBER,
    CORAL,
)

st.set_page_config(
    page_title="COVID-CT Deep Learning Analytics",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()

# Sidebar Navigation & Context
st.sidebar.markdown("## 🫁 COVID-CT AI")
st.sidebar.caption("Chest CT Deep Learning Diagnostics")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Active Backbone:**
    `ResNet-18` (11.2M params)

    **Evaluation Dataset:**
    19,685 Slices (iCTCF Wuhan Cohort)

    **Split Strategy:**
    Proxy Sequence Grouped (70/15/15)
    """
)
st.sidebar.markdown("---")
st.sidebar.info("💡 Use the sidebar navigation menu above to explore interactive pages.")

render_header(
    title="COVID-CT Deep Learning Diagnostics Dashboard",
    subtitle="Interactive companion surface for PyTorch multi-class CT classification & Grad-CAM explainability",
    icon="🫁",
)

render_disclaimer()

# Load summary metrics
comp_file = Path("powerbi/data/model_comparison.csv")
pred_file = Path("powerbi/data/predictions.csv")

if comp_file.exists() and pred_file.exists():
    comp_df = pd.read_csv(comp_file)
    test_comp = comp_df[comp_df["split"] == "test"].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi("Test Accuracy", f"{test_comp['accuracy']:.2%}", delta="+0.88% vs Val", val_color=TEAL)
    with c2:
        render_kpi("Macro F1-Score", f"{test_comp['macro_f1']:.4f}", subtext="Balanced across all 3 classes", val_color=BLUE)
    with c3:
        render_kpi("Total Slices Evaluated", "19,685", subtext="13,735 Train / 2,950 Val / 3,000 Test", val_color=AMBER)
    with c4:
        render_kpi("Wall-Clock Train Time", f"{test_comp['train_time_min']:.2f} min", subtext="CPU 8-core benchmarked", val_color=CORAL)

    st.markdown("### 🧭 Dashboard Surface Navigation")
    n1, n2, n3 = st.columns(3)
    with n1:
        st.markdown(
            """
            <div class="metric-card">
                <h4>📊 1. Overview</h4>
                <p style="color: #94A3B8; font-size: 0.9rem;">Executive diagnostic KPIs, split accuracy comparisons, and class proportions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="metric-card">
                <h4>🧬 4. Dataset Insights</h4>
                <p style="color: #94A3B8; font-size: 0.9rem;">Exploratory slice geometry, intensity profiles, and duplicate integrity audits.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with n2:
        st.markdown(
            """
            <div class="metric-card">
                <h4>📈 2. Model Performance</h4>
                <p style="color: #94A3B8; font-size: 0.9rem;">Confusion matrix heatmaps with normalize toggle, precision/recall/F1, and ROC/PR curves.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="metric-card">
                <h4>🔬 5. Classify CT Slice</h4>
                <p style="color: #94A3B8; font-size: 0.9rem;">Interactive inference demo with slice uploader, gallery picker, and live Grad-CAM heatmaps.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with n3:
        st.markdown(
            """
            <div class="metric-card">
                <h4>⚠️ 3. Error Analysis</h4>
                <p style="color: #94A3B8; font-size: 0.9rem;">Deep dive into misclassifications, confidence distributions, and slice inspection.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="metric-card">
                <h4>📥 6. Reports & Data</h4>
                <p style="color: #94A3B8; font-size: 0.9rem;">Downloadable executive PDF reports, Data Card, Model Card, and Power BI CSVs.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    st.info("Awaiting model evaluation data. Run evaluation pipeline to populate data.")
