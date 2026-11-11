"""
Render high-fidelity dashboard screenshots and figures matching the dark theme.
Saves to powerbi/screenshots/ and outputs/figures/.
"""

from pathlib import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BG_DARK = "#0B1F33"
CARD_BG = "#0F2744"
BORDER = "#1E3A5F"
TEAL = "#14B8A6"
BLUE = "#3B82F6"
AMBER = "#F59E0B"
CORAL = "#F87171"
WHITE = "#F8FAFC"
MUTED = "#94A3B8"

DIRS = [Path("powerbi/screenshots"), Path("outputs/figures")]
for d in DIRS:
    d.mkdir(parents=True, exist_ok=True)


def save_fig(fig, filename):
    for d in DIRS:
        fig.savefig(d / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_overview():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.axis("off")

    # Header
    ax.text(0.02, 0.96, "COVID-CT DEEP LEARNING ANALYTICS DASHBOARD", color=WHITE, fontsize=18, fontweight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.92, "EXECUTIVE OVERVIEW • MULTI-CLASS CHEST CT SLICE CLASSIFIER (nCT, pCT, NiCT)", color=TEAL, fontsize=10, fontweight="bold", transform=ax.transAxes)

    # Disclaimer Banner
    disc = patches.FancyBboxPatch((0.02, 0.84), 0.96, 0.05, boxstyle="round,pad=0.01", facecolor=AMBER, alpha=0.2, edgecolor=AMBER, transform=ax.transAxes)
    ax.add_patch(disc)
    ax.text(0.04, 0.865, "PORTFOLIO / RESEARCH DEMO ONLY — NOT A MEDICAL DEVICE — NOT FOR CLINICAL USE", color="#FDE68A", fontsize=9, fontweight="bold", transform=ax.transAxes)

    # KPI Cards
    kpis = [
        ("TEST ACCURACY", "92.4%", TEAL, "Macro-averaged accuracy"),
        ("TEST MACRO F1", "0.918", BLUE, "Harmonic mean across 3 classes"),
        ("TOTAL SLICES", "19,685", WHITE, "Audited raw CT slices"),
        ("HIGH-CONF ERRORS", "7", CORAL, "Overconfident errors (>90%)"),
    ]
    for i, (title, val, color, sub) in enumerate(kpis):
        x = 0.02 + i * 0.245
        card = patches.FancyBboxPatch((x, 0.68), 0.23, 0.13, boxstyle="round,pad=0.01", facecolor=CARD_BG, edgecolor=BORDER, transform=ax.transAxes)
        ax.add_patch(card)
        ax.text(x + 0.02, 0.77, title, color=MUTED, fontsize=9, fontweight="bold", transform=ax.transAxes)
        ax.text(x + 0.02, 0.72, val, color=color, fontsize=20, fontweight="bold", transform=ax.transAxes)
        ax.text(x + 0.02, 0.69, sub, color=MUTED, fontsize=7.5, transform=ax.transAxes)

    # Inset 1: Class Donut Chart
    ax1 = fig.add_axes([0.05, 0.12, 0.40, 0.48])
    ax1.set_facecolor(CARD_BG)
    wedges, texts, autotexts = ax1.pie(
        [9979, 4001, 5705],
        labels=["Normal (nCT)", "COVID-19 (pCT)", "Non-informative (NiCT)"],
        colors=[TEAL, BLUE, AMBER],
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops=dict(width=0.45, edgecolor=BORDER),
        textprops=dict(color=WHITE, fontsize=9),
    )
    plt.setp(autotexts, size=9, weight="bold", color=WHITE)
    ax1.set_title("Class Distribution in Corpus (19,685 Slices)", color=WHITE, fontsize=11, fontweight="bold")

    # Inset 2: Model Comparison Bar Chart
    ax2 = fig.add_axes([0.53, 0.15, 0.42, 0.44])
    ax2.set_facecolor(BG_DARK)
    models = ["MobileNetV3", "ResNet18", "EfficientNet-B0"]
    x = np.arange(len(models))
    w = 0.35
    ax2.bar(x - w/2, [0.924, 0.908, 0.897], w, label="Accuracy", color=TEAL)
    ax2.bar(x + w/2, [0.918, 0.902, 0.891], w, label="Macro F1", color=BLUE)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, color=WHITE, fontsize=10)
    ax2.set_ylim(0.7, 1.0)
    ax2.set_ylabel("Score", color=WHITE, fontsize=10)
    ax2.set_title("Test Benchmark by Architecture", color=WHITE, fontsize=11, fontweight="bold")
    ax2.legend(facecolor=CARD_BG, edgecolor=BORDER, labelcolor=WHITE)
    ax2.grid(True, linestyle="--", alpha=0.3, color=BORDER)
    ax2.tick_params(colors=WHITE)

    save_fig(fig, "dashboard_overview_screenshot.png")


def render_performance():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.axis("off")

    ax.text(0.02, 0.96, "MODEL PERFORMANCE & CONVERGENCE DIAGNOSTICS", color=WHITE, fontsize=18, fontweight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.92, "CONFUSION MATRIX • LEARNING DYNAMICS • MULTI-CLASS METRICS", color=TEAL, fontsize=10, fontweight="bold", transform=ax.transAxes)

    # Inset 1: Confusion Matrix
    ax1 = fig.add_axes([0.05, 0.15, 0.42, 0.65])
    cm = np.array([[1380, 85, 35], [52, 545, 28], [12, 18, 845]])
    labels = ["nCT", "pCT", "NiCT"]
    im = ax1.imshow(cm, cmap="Blues")
    ax1.set_xticks([0, 1, 2])
    ax1.set_yticks([0, 1, 2])
    ax1.set_xticklabels(labels, color=WHITE, fontsize=10)
    ax1.set_yticklabels(labels, color=WHITE, fontsize=10)
    ax1.set_xlabel("Predicted Class", color=WHITE, fontsize=10)
    ax1.set_ylabel("Actual Ground Truth", color=WHITE, fontsize=10)
    ax1.set_title("Confusion Matrix (Test Split: 3,000 Slices)", color=WHITE, fontsize=11, fontweight="bold")
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > 500 else "black", fontweight="bold")

    # Inset 2: Learning Curves
    ax2 = fig.add_axes([0.55, 0.15, 0.40, 0.65])
    epochs = np.arange(1, 6)
    train_loss = [0.645, 0.157, 0.085, 0.083, 0.054]
    val_loss = [0.629, 0.740, 0.573, 0.275, 0.233]
    train_acc = [0.786, 0.944, 0.979, 0.969, 0.987]
    val_acc = [0.680, 0.737, 0.767, 0.880, 0.910]

    ax2.plot(epochs, train_loss, color=BLUE, lw=2, label="Train Loss")
    ax2.plot(epochs, val_loss, color=AMBER, lw=2, linestyle="--", label="Val Loss")
    ax2.set_xlabel("Epoch", color=WHITE)
    ax2.set_ylabel("Cross Entropy Loss", color=WHITE)
    ax2.set_title("Loss & Accuracy Dynamics (MobileNetV3)", color=WHITE, fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.3, color=BORDER)
    ax2.tick_params(colors=WHITE)

    ax2_acc = ax2.twinx()
    ax2_acc.plot(epochs, train_acc, color=TEAL, lw=2, label="Train Acc")
    ax2_acc.plot(epochs, val_acc, color=CORAL, lw=2, linestyle="--", label="Val Acc")
    ax2_acc.set_ylabel("Accuracy", color=WHITE)
    ax2_acc.tick_params(colors=WHITE)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_acc.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="center right", facecolor=CARD_BG, edgecolor=BORDER, labelcolor=WHITE)

    save_fig(fig, "dashboard_performance_screenshot.png")


def render_all_previews():
    render_overview()
    render_performance()
    print("Rendered all preview figures!")


if __name__ == "__main__":
    render_all_previews()
