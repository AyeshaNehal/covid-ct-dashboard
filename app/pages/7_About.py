"""
Page 7: About, Scientific Methodology, Citations, and Clinical Limitations
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from app.components.theme import (
    apply_custom_css,
    render_disclaimer,
    render_header,
    TEAL,
    AMBER,
    CORAL,
)

st.set_page_config(page_title="About & Limitations - COVID-CT", page_icon="📖", layout="wide")
apply_custom_css()
render_header("About, Methodology & Clinical Limitations", "Dataset provenance, split strategy caveats, and scientific citations", "📖")
render_disclaimer()

st.markdown(
    """
    ### 🔬 Scientific Provenance & Objective
    **covid-ct-dashboard** is an end-to-end deep learning classification and diagnostics dashboard
    designed to evaluate chest computed tomography (CT) axial cross-sections for COVID-19 manifestations.
    The primary research objective is to investigate the feasibility of transfer learning while maintaining
    rigorous evaluation practices and honest disclosure of machine learning failure modes.

    ---

    ### ⚠️ Crucial Split Strategy Caveat: Proxy-Grouped Splitting
    In biomedical imaging, random split partitioning across individual slices causes catastrophic data leakage
    because adjacent slices from the same patient scan are nearly identical. 
    
    * **Audit Finding**: In the Kaggle dataset (`azaemon/preprocessed-ct-scans-for-covid19`), DICOM headers and native patient IDs have been stripped, and slices were renamed sequentially (`nCT1.jpg`, `pCT1.jpg`...).
    * **Proxy Grouping Implementation**: To mitigate leakage, slices were clustered into contiguous proxy sequence blocks (25 slices per block) and segregated using two-stage stratified `GroupShuffleSplit` (70% Train, 15% Val, 15% Test).
    * **Remaining Risk**: Because true patient IDs are absent, **patient overlap across splits remains possible**. High empirical test scores (e.g. 98.27% accuracy) must therefore be interpreted as upper-bound benchmarks on this specific dataset rather than proof of generalized clinical efficacy.

    ---

    ### 🚨 Major Clinical Limitations
    1. **Single 2D Slice vs. 3D Volumetric Context**: Clinical chest CT diagnosis relies on volumetric 3D series examination. Assessing isolated 2D slices loses longitudinal depth and volume estimation.
    2. **Geographical & Scanner Bias**: The source cohort originates from Wuhan hospitals during early 2020. Scanner calibration, slice thickness protocols, and patient demographics have not been externally validated on international multicenter cohorts.
    3. **Diagnostic Confounders**: COVID-19 ground-glass opacities (GGO) overlap morphologically with other viral pneumonias (Influenza, Adenovirus) and organizing pneumonia.

    ---

    ### 📚 Scientific Citations & References
    * **Dataset**: Abu Zahid Bin Aziz, *Preprocessed CT Scans for COVID-19*, Kaggle (CC BY 4.0).
    * **iCTCF Cohort**: Ning, C. et al. (2020). *Open-source CT training dataset for COVID-19*.
    * **Deep Residual Learning (ResNet)**: He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. IEEE CVPR.
    * **Grad-CAM**: Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. IEEE ICCV.
    """
)
