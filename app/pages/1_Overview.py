"""
Page 1: Executive Overview Dashboard
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px

from app.components.charts import plot_split_accuracy_plotly
from app.components.theme import (
    apply_custom_css,
    render_disclaimer,
    render_header,
    render_kpi,
    TEAL,
    BLUE,
    AMBER,
    CORAL,
    TEXT_WHITE,
)

st.set_page_config(page_title="Executive Overview - COVID-CT", page_icon="📊", layout="wide")
apply_custom_css()
render_header("Executive Overview & Diagnostics", "High-level diagnostic KPIs, split metrics, and class distribution", "📊")
render_disclaimer()

@st.cache_data
def load_data():
    comp_df = pd.read_csv("powerbi/data/model_comparison.csv")
    preds_df = pd.read_csv("powerbi/data/predictions.csv")
    return comp_df, preds_df

comp_df, preds_df = load_data()
test_preds = preds_df[preds_df["split"] == "test"]
test_metrics = comp_df[comp_df["split"] == "test"].iloc[0]

c1, c2, c3, c4 = st.columns(4)
with c1:
    render_kpi("Test Accuracy", f"{test_metrics['accuracy']:.2%}", delta="+0.88% vs Val", val_color=TEAL)
with c2:
    render_kpi("Macro F1-Score", f"{test_metrics['macro_f1']:.4f}", subtext="Harmonic Mean of Precision & Recall", val_color=BLUE)
with c3:
    err_count = len(test_preds[test_preds["correct"] == 0])
    render_kpi("Test Error Count", f"{err_count}", subtext=f"Out of {len(test_preds):,} test slices ({err_count/len(test_preds):.2%})", val_color=CORAL)
with c4:
    render_kpi("Avg Model Confidence", f"{test_preds['conf'].mean():.2%}", subtext=f"Errors Avg Conf: {test_preds[test_preds['correct']==0]['conf'].mean():.2%}", val_color=AMBER)

col_left, col_right = st.columns([1, 1])

with col_left:
    st.plotly_chart(plot_split_accuracy_plotly(comp_df), use_container_width=True)

with col_right:
    class_counts = test_preds["true_label"].value_counts().reset_index()
    class_counts.columns = ["Class", "Count"]
    fig_donut = px.pie(
        class_counts,
        values="Count",
        names="Class",
        hole=0.45,
        title="Test Split Class Composition (3,000 Slices)",
        color="Class",
        color_discrete_map={"nCT": TEAL, "pCT": CORAL, "NiCT": AMBER},
    )
    fig_donut.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#132D4A",
        font=dict(color=TEXT_WHITE),
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_donut, use_container_width=True)

st.markdown("### 📋 Evaluated Split Summary Table")
st.dataframe(comp_df.style.format({"accuracy": "{:.2%}", "macro_precision": "{:.4f}", "macro_recall": "{:.4f}", "macro_f1": "{:.4f}", "params": "{:,}", "train_time_min": "{:.2f}"}), use_container_width=True)
