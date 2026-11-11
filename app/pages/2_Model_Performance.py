"""
Page 2: Model Performance Diagnostics & Metrics
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go

from app.components.charts import (
    plot_confusion_matrix_plotly,
    plot_per_class_metrics_plotly,
    plot_roc_curves_plotly,
    plot_pr_curves_plotly,
)
from app.components.theme import (
    apply_custom_css,
    render_disclaimer,
    render_header,
    TEAL,
    BLUE,
    AMBER,
    CORAL,
)

st.set_page_config(page_title="Model Performance - COVID-CT", page_icon="📈", layout="wide")
apply_custom_css()
render_header("Model Performance & Diagnostics", "Interactive confusion matrix, per-class metrics, and ROC/PR analysis", "📈")
render_disclaimer()

@st.cache_data
def load_perf_data():
    preds = pd.read_csv("powerbi/data/predictions.csv")
    test_preds = preds[preds["split"] == "test"]
    class_df = pd.read_csv("powerbi/data/class_metrics.csv")
    test_class_df = class_df[class_df["split"] == "test"]
    history_df = pd.read_csv("powerbi/data/training_history.csv")
    return test_preds, test_class_df, history_df

test_preds, test_class_df, history_df = load_perf_data()

t1, t2 = st.tabs(["🔲 Confusion Matrix & Metrics", "📉 ROC & PR Diagnostic Curves"])

with t1:
    c_toggle, c_info = st.columns([1, 3])
    with c_toggle:
        norm_toggle = st.toggle("Normalize Matrix by Row (%)", value=False)
    with c_info:
        st.caption("Displays test split performance (3,000 held-out slices). Use toggle to switch between absolute counts and row percentages.")

    col1, col2 = st.columns([1, 1])
    with col1:
        cm = confusion_matrix(test_preds["true_label"], test_preds["pred_label"], labels=["nCT", "pCT", "NiCT"])
        st.plotly_chart(plot_confusion_matrix_plotly(cm, ["nCT", "pCT", "NiCT"], normalize=norm_toggle), use_container_width=True)

    with col2:
        st.plotly_chart(plot_per_class_metrics_plotly(test_class_df), use_container_width=True)

    st.markdown("### 📊 Tabular Per-Class Performance")
    st.dataframe(
        test_class_df.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}", "support": "{:,}"}),
        use_container_width=True,
    )

with t2:
    filter_col, _ = st.columns([1, 2])
    with filter_col:
        selected_class = st.selectbox("Select Class to Highlight:", ["All", "nCT", "pCT", "NiCT"])

    r1, r2 = st.columns(2)
    with r1:
        st.plotly_chart(plot_roc_curves_plotly(test_preds, target_class=selected_class), use_container_width=True)
    with r2:
        st.plotly_chart(plot_pr_curves_plotly(test_preds, target_class=selected_class), use_container_width=True)

    st.markdown("### 🏋️ Training & Validation Trajectory (ResNet-18)")
    st.dataframe(history_df.style.format({"train_loss": "{:.4f}", "train_acc": "{:.2%}", "val_loss": "{:.4f}", "val_acc": "{:.2%}", "train_time_min": "{:.2f}"}), use_container_width=True)
