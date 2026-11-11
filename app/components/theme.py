"""
UI Styling, modern CSS injection, KPI Cards with deltas, and Disclaimer Components.
Matches project palette: Navy #0B1F33, Teal #14B8A6, Amber #F59E0B, Coral #F87171.
"""

from typing import Optional
import streamlit as st

TEAL = "#14B8A6"
BLUE = "#3B82F6"
AMBER = "#F59E0B"
CORAL = "#F87171"
DARK_NAVY = "#0B1F33"
CARD_BG = "#132D4A"
SIDEBAR_BG = "#081626"
BORDER_COLOR = "#1E3A5F"
TEXT_WHITE = "#F8FAFC"
TEXT_MUTED = "#94A3B8"


def apply_custom_css():
    st.markdown(
        f"""
        <style>
        /* Base typography & theme */
        .stApp {{
            background-color: {DARK_NAVY};
            color: {TEXT_WHITE};
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}
        [data-testid="stSidebar"] {{
            background-color: {SIDEBAR_BG};
            border-right: 1px solid {BORDER_COLOR};
        }}
        [data-testid="stSidebarNav"] {{
            padding-top: 1rem;
        }}
        /* KPI & Metric cards */
        .metric-card {{
            background: {CARD_BG};
            border: 1px solid {BORDER_COLOR};
            border-radius: 12px;
            padding: 20px 22px;
            text-align: left;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            margin-bottom: 16px;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
            border-color: {TEAL};
        }}
        .metric-title {{
            color: {TEXT_MUTED};
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 6px;
        }}
        .metric-value {{
            font-size: 2.1rem;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 6px;
        }}
        .metric-delta {{
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .metric-sub {{
            color: {TEXT_MUTED};
            font-size: 0.8rem;
        }}
        /* Disclaimer banners */
        .disclaimer-banner {{
            background-color: rgba(245, 158, 11, 0.10);
            border-left: 4px solid {AMBER};
            border-radius: 6px;
            padding: 12px 18px;
            margin-bottom: 22px;
            color: #FDE68A;
            font-size: 0.88rem;
            line-height: 1.5;
        }}
        /* Section cards */
        .section-card {{
            background-color: {CARD_BG};
            border: 1px solid {BORDER_COLOR};
            border-radius: 12px;
            padding: 22px;
            margin-bottom: 20px;
        }}
        /* Header titles */
        .page-header {{
            margin-bottom: 22px;
        }}
        .page-header h1 {{
            font-size: 1.9rem;
            font-weight: 700;
            color: {TEXT_WHITE};
            margin-bottom: 4px;
        }}
        .page-header p {{
            font-size: 0.95rem;
            color: {TEXT_MUTED};
            margin-bottom: 0;
        }}
        /* Custom buttons & download anchors */
        .download-btn {{
            display: inline-block;
            background-color: {TEAL};
            color: {DARK_NAVY} !important;
            font-weight: 600;
            padding: 10px 18px;
            border-radius: 8px;
            text-decoration: none;
            margin-right: 12px;
            margin-bottom: 12px;
        }}
        .download-btn:hover {{
            background-color: #2DD4BF;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(title: str, subtitle: str, icon: str = "📊"):
    st.markdown(
        f"""
        <div class="page-header">
            <h1>{icon} {title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer():
    st.markdown(
        """
        <div class="disclaimer-banner">
            <strong>⚠️ NOT A MEDICAL DEVICE / RESEARCH & PORTFOLIO DEMONSTRATION ONLY:</strong><br>
            This system was engineered strictly for deep learning benchmarking and educational explainability.
            It is <strong>NOT</strong> approved by regulatory bodies (FDA, CE, NMPA) and must never be employed
            for diagnostic decision-making, clinical triage, or patient treatment.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(
    title: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: str = TEAL,
    subtext: str = "",
    val_color: str = TEXT_WHITE,
):
    delta_html = ""
    if delta:
        delta_html = f'<div class="metric-delta" style="color: {delta_color};">{delta}</div>'
    sub_html = f'<div class="metric-sub">{subtext}</div>' if subtext else ""

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color: {val_color};">{value}</div>
            {delta_html}
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
