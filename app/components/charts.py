"""
Shared Plotly visualization components styled in the project dark palette.
Palette: Navy #0B1F33, Teal #14B8A6, Amber #F59E0B, Coral #F87171, Blue #3B82F6.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, roc_curve, auc

DARK_BG = "#0B1F33"
CARD_BG = "#132D4A"
TEAL = "#14B8A6"
AMBER = "#F59E0B"
CORAL = "#F87171"
BLUE = "#3B82F6"
PURPLE = "#8B5CF6"
TEXT_WHITE = "#F8FAFC"
TEXT_MUTED = "#94A3B8"
GRID_COLOR = "#1E3A5F"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT_WHITE, family="Segoe UI, Inter, sans-serif"),
    margin=dict(l=40, r=30, t=40, b=40),
)


def plot_confusion_matrix_plotly(cm: np.ndarray, class_names: List[str], normalize: bool = False) -> go.Figure:
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        text_matrix = [[f"{val:.1%}" for val in row] for row in cm_display]
        z_vals = cm_display
        colorscale = [[0.0, "#0B1F33"], [0.5, "#0E7490"], [1.0, TEAL]]
    else:
        text_matrix = [[str(int(val)) for val in row] for row in cm]
        z_vals = cm
        colorscale = [[0.0, "#0B1F33"], [0.5, "#1E3A8A"], [1.0, "#38BDF8"]]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=class_names,
            y=class_names,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=14, color="white"),
            colorscale=colorscale,
            showscale=True,
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Test Confusion Matrix" + (" (Normalized)" if normalize else " (Raw Counts)"),
        xaxis_title="Predicted Class",
        yaxis_title="True Ground Truth",
        yaxis_autorange="reversed",
        height=420,
    )
    return fig


def plot_per_class_metrics_plotly(class_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Precision", x=class_df["class"], y=class_df["precision"], marker_color=TEAL))
    fig.add_trace(go.Bar(name="Recall", x=class_df["class"], y=class_df["recall"], marker_color=BLUE))
    fig.add_trace(go.Bar(name="F1-Score", x=class_df["class"], y=class_df["f1"], marker_color=AMBER))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="group",
        title="Per-Class Performance Metrics (Precision / Recall / F1)",
        yaxis=dict(range=[0.8, 1.02], gridcolor=GRID_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
    )
    return fig


def plot_confidence_distribution_plotly(pred_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    correct = pred_df[pred_df["correct"] == 1]["conf"]
    wrong = pred_df[pred_df["correct"] == 0]["conf"]

    fig.add_trace(go.Histogram(x=correct, name="Correct Predictions", marker_color=TEAL, opacity=0.75, nbinsx=35))
    fig.add_trace(go.Histogram(x=wrong, name="Misclassifications", marker_color=CORAL, opacity=0.85, nbinsx=35))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode="overlay",
        title="Model Prediction Confidence (Correct vs. Errors)",
        xaxis_title="Confidence Probability",
        yaxis_title="Slice Count",
        height=380,
    )
    return fig


def plot_roc_curves_plotly(pred_df: pd.DataFrame, target_class: Optional[str] = None) -> go.Figure:
    fig = go.Figure()
    classes = ["nCT", "pCT", "NiCT"]
    colors = {"nCT": TEAL, "pCT": CORAL, "NiCT": AMBER}

    for c in classes:
        if target_class and target_class != "All" and target_class != c:
            continue
        y_true_binary = (pred_df["true_label"] == c).astype(int)
        y_prob = pred_df[f"p_{c}"]
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob)
        roc_auc = auc(fpr, tpr)

        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{c} (AUC = {roc_auc:.4f})",
                line=dict(color=colors[c], width=2.5),
            )
        )

    fig.add_shape(type="line", line=dict(dash="dash", color=TEXT_MUTED), x0=0, x1=1, y0=0, y1=1)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="One-vs-Rest ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=380,
    )
    return fig


def plot_pr_curves_plotly(pred_df: pd.DataFrame, target_class: Optional[str] = None) -> go.Figure:
    fig = go.Figure()
    classes = ["nCT", "pCT", "NiCT"]
    colors = {"nCT": TEAL, "pCT": CORAL, "NiCT": AMBER}

    for c in classes:
        if target_class and target_class != "All" and target_class != c:
            continue
        y_true_binary = (pred_df["true_label"] == c).astype(int)
        y_prob = pred_df[f"p_{c}"]
        precision, recall, _ = precision_recall_curve(y_true_binary, y_prob)
        pr_auc = auc(recall, precision)

        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"{c} (PR-AUC = {pr_auc:.4f})",
                line=dict(color=colors[c], width=2.5),
            )
        )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Precision-Recall Curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=380,
    )
    return fig


def plot_split_accuracy_plotly(comp_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=comp_df["split"],
            y=comp_df["accuracy"],
            marker_color=[BLUE, PURPLE, TEAL],
            text=[f"{acc:.2%}" for acc in comp_df["accuracy"]],
            textposition="auto",
        )
    )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="ResNet-18 Accuracy Across Splits",
        xaxis_title="Split Partition",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0.9, 1.01], gridcolor=GRID_COLOR),
        height=320,
    )
    return fig
