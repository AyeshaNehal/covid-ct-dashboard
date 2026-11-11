"""
Page 6: Reports & Downloadable Assets
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd

from app.components.theme import (
    apply_custom_css,
    render_disclaimer,
    render_header,
    TEAL,
    AMBER,
    CORAL,
)

st.set_page_config(page_title="Reports & Assets - COVID-CT", page_icon="📥", layout="wide")
apply_custom_css()
render_header("Executive Reports & Downloadable Assets", "Download model evaluation reports, data cards, and raw Power BI data tables", "📥")
render_disclaimer()

st.markdown("### 📄 Formal Technical Documentation & Reports")

rep_col1, rep_col2 = st.columns(2)

with rep_col1:
    st.markdown(
        """
        <div class="metric-card">
            <h4>📑 Comprehensive Model Evaluation Report (PDF)</h4>
            <p style="color: #94A3B8; font-size: 0.88rem;">Executive medical imaging AI benchmark report detailing study design, split strategy, authentic empirical metrics, Grad-CAM explainability, and failure mode taxonomy.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    pdf_path = Path("reports/model_report.pdf")
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Model Report (PDF)",
                data=f.read(),
                file_name="COVID_CT_ResNet18_Model_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    else:
        st.info("Generating PDF report...")

with rep_col2:
    st.markdown(
        """
        <div class="metric-card">
            <h4>📋 Markdown Documentation Suite</h4>
            <p style="color: #94A3B8; font-size: 0.88rem;">Full transparency documentation including Dataset Card, Model Card, and In-depth Qualitative Error Taxonomy.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    md_report = Path("reports/model_report.md")
    if md_report.exists():
        with open(md_report, "r", encoding="utf-8") as f:
            st.download_button(
                label="⬇️ Download Technical Report (Markdown)",
                data=f.read(),
                file_name="model_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

st.markdown("---")
st.markdown("### 📊 Power BI & Analytic Data CSV Tables")
st.caption("All tables contain genuine empirical rows evaluated on the held-out partitions.")

d1, d2, d3, d4 = st.columns(4)

with d1:
    p_csv = Path("powerbi/data/predictions.csv")
    if p_csv.exists():
        with open(p_csv, "rb") as f:
            st.download_button("⬇️ predictions.csv (19,685 rows)", f.read(), "predictions.csv", "text/csv", use_container_width=True)

with d2:
    c_csv = Path("powerbi/data/model_comparison.csv")
    if c_csv.exists():
        with open(c_csv, "rb") as f:
            st.download_button("⬇️ model_comparison.csv", f.read(), "model_comparison.csv", "text/csv", use_container_width=True)

with d3:
    m_csv = Path("powerbi/data/class_metrics.csv")
    if m_csv.exists():
        with open(m_csv, "rb") as f:
            st.download_button("⬇️ class_metrics.csv", f.read(), "class_metrics.csv", "text/csv", use_container_width=True)

with d4:
    h_csv = Path("powerbi/data/training_history.csv")
    if h_csv.exists():
        with open(h_csv, "rb") as f:
            st.download_button("⬇️ training_history.csv", f.read(), "training_history.csv", "text/csv", use_container_width=True)
