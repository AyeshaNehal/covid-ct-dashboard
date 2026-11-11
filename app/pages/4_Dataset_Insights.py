"""
Page 4: Exploratory Data Analysis & Quality Audit
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px

from app.components.theme import (
    apply_custom_css,
    render_disclaimer,
    render_header,
    render_kpi,
    TEAL,
    BLUE,
    AMBER,
    TEXT_WHITE,
)

st.set_page_config(page_title="Dataset Insights - COVID-CT", page_icon="🧬", layout="wide")
apply_custom_css()
render_header("Dataset Insights & Exploratory Audit", "Class proportions, slice geometry, brightness distribution, and split integrity", "🧬")
render_disclaimer()

@st.cache_data
def load_eda_data():
    summary_df = pd.read_csv("powerbi/data/dataset_summary.csv")
    preds_df = pd.read_csv("powerbi/data/predictions.csv")
    return summary_df, preds_df

summary_df, preds_df = load_eda_data()

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_kpi("Total Original Slices", f"{summary_df['total_slices'].sum():,}", subtext="From Kaggle iCTCF cohort", val_color=TEAL)
with c2:
    render_kpi("Corrupt Files Detected", f"{summary_df['corrupt_images'].sum()}", subtext="Zero corrupt scans in raw archive", val_color=BLUE)
with c3:
    render_kpi("Exact Byte Duplicates", f"{summary_df['exact_duplicates'].sum()}", subtext="All isolated to NiCT slices", val_color=AMBER)
with c4:
    render_kpi("Proxy Sequence Groups", f"{summary_df['unique_groups'].sum():,}", subtext="Volume blocks preventing slice leakage", val_color=TEAL)

col1, col2 = st.columns([1, 1])

with col1:
    fig_bar = px.bar(
        summary_df,
        x="class",
        y="total_slices",
        title="Class Volume Distribution across All Scans",
        color="class",
        color_discrete_map={"nCT": TEAL, "pCT": "#F87171", "NiCT": AMBER},
        text="total_slices",
    )
    fig_bar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#132D4A",
        font=dict(color=TEXT_WHITE),
        height=340,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    fig_bright = px.histogram(
        preds_df,
        x="mean_brightness",
        color="true_label",
        nbins=40,
        barmode="overlay",
        title="Slice Mean Pixel Intensity by Diagnostic Class",
        color_discrete_map={"nCT": TEAL, "pCT": "#F87171", "NiCT": AMBER},
    )
    fig_bright.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#132D4A",
        font=dict(color=TEXT_WHITE),
        height=340,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_bright, use_container_width=True)

st.markdown("### 📋 Dataset Audit Table")
st.dataframe(
    summary_df.style.format({
        "total_slices": "{:,}",
        "unique_groups": "{:,}",
        "mean_width": "{:.1f} px",
        "mean_height": "{:.1f} px",
        "mean_brightness": "{:.3f}",
        "corrupt_images": "{:,}",
        "exact_duplicates": "{:,}",
    }),
    use_container_width=True,
)
