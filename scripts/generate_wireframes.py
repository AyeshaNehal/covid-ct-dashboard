"""
Generate high-resolution PNG layout wireframes (1280x720 canvas) for Power BI dashboard pages.
Provides visual architectural blueprints for manual .pbix reconstruction.
"""

from pathlib import Path
import matplotlib.patches as patches
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("powerbi/docs/layout_wireframes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dark theme palette
BG_DARK = "#0B1F33"
CARD_BG = "#0F2744"
BORDER_COLOR = "#1E3A5F"
TEAL = "#14B8A6"
BLUE = "#3B82F6"
AMBER = "#F59E0B"
CORAL = "#F87171"
TEXT_WHITE = "#F8FAFC"
TEXT_MUTED = "#94A3B8"


def draw_canvas(title: str, subtitle: str, active_page: str):
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    fig.patch.set_facecolor(BG_DARK)
    ax.set_facecolor(BG_DARK)
    ax.set_xlim(0, 1280)
    ax.set_ylim(720, 0)  # Inverted Y for screen coordinates (0,0 at top-left)
    ax.axis("off")

    # Header Bar
    header = patches.Rectangle((0, 0), 1280, 60, facecolor="#071524", edgecolor=BORDER_COLOR, linewidth=1)
    ax.add_patch(header)
    ax.text(20, 26, title, color=TEXT_WHITE, fontsize=14, fontweight="bold", va="center")
    ax.text(20, 48, subtitle, color=AMBER, fontsize=8, fontweight="bold", va="center")

    # Nav Tabs
    pages = ["Overview", "Performance", "Errors", "Insights", "About"]
    for i, p in enumerate(pages):
        x = 750 + i * 102
        is_active = (p.lower() in active_page.lower())
        color = TEAL if is_active else CARD_BG
        edge = TEAL if is_active else BORDER_COLOR
        txt_color = TEXT_WHITE if is_active else TEXT_MUTED
        btn = patches.FancyBboxPatch(
            (x, 15), 94, 30, boxstyle="round,pad=2", facecolor=color, edgecolor=edge, linewidth=1
        )
        ax.add_patch(btn)
        ax.text(x + 47, 30, p, color=txt_color, fontsize=8, fontweight="bold", ha="center", va="center")

    return fig, ax


def draw_card(ax, x, y, w, h, title, value, accent_color=TEAL):
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=3", facecolor=CARD_BG, edgecolor=BORDER_COLOR, linewidth=1
    )
    ax.add_patch(rect)
    ax.text(x + 15, y + 25, title, color=TEXT_MUTED, fontsize=9, fontweight="bold", va="center")
    ax.text(x + 15, y + 58, value, color=accent_color, fontsize=18, fontweight="bold", va="center")


def draw_box(ax, x, y, w, h, label, sublabel=""):
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=3", facecolor=CARD_BG, edgecolor=BORDER_COLOR, linewidth=1
    )
    ax.add_patch(rect)
    ax.text(x + 15, y + 22, label, color=TEXT_WHITE, fontsize=10, fontweight="bold", va="center")
    if sublabel:
        ax.text(x + 15, y + 40, sublabel, color=TEXT_MUTED, fontsize=8, va="center")


def generate_wireframes():
    # 1. Overview
    fig, ax = draw_canvas(
        "COVID-CT DEEP LEARNING ANALYTICS",
        "PORTFOLIO / RESEARCH DEMO • NOT A MEDICAL DEVICE",
        "Overview",
    )
    draw_box(ax, 20, 70, 1240, 38, "KEY INSIGHT", "MobileNetV3 achieves leading macro-F1 with leakage-free patient grouping.")
    draw_box(ax, 20, 118, 220, 580, "SLICERS PANEL", "[Model Dropdown] | [Split: test/val/train]")
    draw_card(ax, 260, 118, 230, 85, "TEST ACCURACY", "92.4%", TEAL)
    draw_card(ax, 510, 118, 230, 85, "MACRO F1-SCORE", "0.918", BLUE)
    draw_card(ax, 760, 118, 230, 85, "TOTAL TEST SLICES", "480", TEXT_WHITE)
    draw_card(ax, 1010, 118, 250, 85, "HIGH-CONFIDENCE ERRORS", "7", CORAL)
    draw_box(ax, 260, 215, 480, 265, "DONUT CHART: CLASS BALANCE", "nCT (Normal) vs pCT (COVID-19) vs NiCT (Non-informative)")
    draw_box(ax, 760, 215, 500, 265, "CLUSTERED COLUMN: ACCURACY BY SPLIT", "Train vs Val vs Test across Model Backbones")
    draw_box(ax, 260, 492, 1000, 206, "HORIZONTAL BAR: ARCHITECTURE COMPARISON", "MobileNetV3 vs ResNet18 vs EfficientNet-B0 (F1, Accuracy, Params)")
    fig.savefig(OUTPUT_DIR / "p1_executive_overview.png", bbox_inches="tight")
    plt.close()

    # 2. Performance
    fig, ax = draw_canvas("MODEL PERFORMANCE & METRIC DIAGNOSTICS", "CONFUSION MATRIX • LEARNING DYNAMICS • ROC", "Performance")
    draw_box(ax, 20, 70, 1240, 38, "KEY INSIGHT", "Zero false negatives on NiCT slices; mild boundary confusion between early pCT & nCT.")
    draw_box(ax, 20, 118, 440, 320, "HEATMAP: CONFUSION MATRIX", "Rows: Ground Truth | Columns: Predicted (nCT, pCT, NiCT)")
    draw_box(ax, 480, 118, 780, 320, "DUAL LINE CHART: TRAINING & VALIDATION CURVES", "Loss & Accuracy Progression vs 25 Epochs")
    draw_box(ax, 20, 450, 620, 248, "TABLE: CLASS PERFORMANCE BREAKDOWN", "Precision, Recall, F1-Score, and Support per Class")
    draw_box(ax, 660, 450, 600, 248, "MULTI-CLASS ROC & PR SUMMARY", "One-vs-Rest AUC Trajectories (nCT, pCT, NiCT)")
    fig.savefig(OUTPUT_DIR / "p2_model_performance.png", bbox_inches="tight")
    plt.close()

    # 3. Error Analysis
    fig, ax = draw_canvas("SYSTEMATIC ERROR ANALYSIS & UNCERTAINTY", "MISCLASSIFICATION TAXONOMY • CONFIDENCE PROFILES", "Errors")
    draw_box(ax, 20, 70, 1240, 38, "KEY INSIGHT", "Top errors feature severe motion artifact and peripheral low-contrast consolidations.")
    draw_card(ax, 20, 118, 290, 85, "TOTAL MISCLASSIFICATIONS", "36", CORAL)
    draw_card(ax, 330, 118, 290, 85, "TEST ERROR RATE", "7.6%", AMBER)
    draw_card(ax, 640, 118, 290, 85, "AVG CONFIDENCE (WRONG)", "74.2%", TEAL)
    draw_card(ax, 950, 118, 310, 85, "HIGH-CONFIDENCE ERRORS (>90%)", "7", CORAL)
    draw_box(ax, 20, 215, 440, 483, "BAR CHART: ERROR TRANSITION PAIRS", "pCT -> nCT | nCT -> pCT | NiCT -> pCT counts")
    draw_box(ax, 480, 215, 780, 230, "DISTRIBUTION: CONFIDENCE BINS (CORRECT VS ERROR)", "<50%, 50-60%, 60-70%, 70-80%, 80-90%, 90-100%")
    draw_box(ax, 480, 457, 780, 241, "DRILL-THROUGH TABLE: MISCLASSIFIED SLICES", "image_id, true_label, pred_label, conf, path, mean_brightness")
    fig.savefig(OUTPUT_DIR / "p3_error_analysis.png", bbox_inches="tight")
    plt.close()

    # 4. Dataset Insights
    fig, ax = draw_canvas("EXPLORATORY DATA ANALYSIS & HEALTH CHECK", "SLICE RESOLUTION • BRIGHTNESS DYNAMICS • LEAKAGE PREVENTION", "Insights")
    draw_box(ax, 20, 70, 1240, 38, "KEY INSIGHT", "Standardized 224x224 input slices; no corrupt images or byte duplicates detected.")
    draw_box(ax, 20, 118, 380, 270, "DONUT CHART: CLASS SHARES", "Proportion of nCT, pCT, NiCT in corpus")
    draw_box(ax, 420, 118, 420, 270, "COLUMN CHART: SLICES PER GROUP", "Frequency of slices per patient / proxy volume")
    draw_box(ax, 860, 118, 400, 270, "HEALTH CHECK CARDS", "Corrupt Images: 0 | Duplicates: 0 | Total Groups: N")
    draw_box(ax, 20, 400, 610, 298, "HISTOGRAM: MEAN BRIGHTNESS BY CLASS", "Intensity distributions across normal vs pathology")
    draw_box(ax, 650, 400, 610, 298, "STACKED BAR: SPLIT PARTITION COMPOSITION", "70% Train / 15% Val / 15% Test strictly segregated")
    fig.savefig(OUTPUT_DIR / "p4_dataset_insights.png", bbox_inches="tight")
    plt.close()

    # 5. About & Limitations
    fig, ax = draw_canvas("METHODOLOGY, CITATIONS & CLINICAL DISCLAIMERS", "ETHICAL AI • ARCHITECTURAL TRANSPARENCY • RIGOROUS LIMITS", "About")
    draw_box(ax, 20, 70, 610, 305, "PROJECT OVERVIEW & CITATIONS", "Abu Zahid Bin Aziz (CC BY 4.0), iCTCF (Ning et al., 2020), MobileNetV3 (Howard et al., 2019)")
    draw_box(ax, 650, 70, 610, 305, "SPLITTING STRATEGY & LEAKAGE AVOIDANCE", "GroupShuffleSplit enforces mutual exclusivity across patient volumes to avoid optimistic bias.")
    draw_box(ax, 20, 390, 610, 308, "CRITICAL METHODOLOGICAL LIMITATIONS", "2D slice limitation without 3D volumetric context; single regional epicenter cohort in Wuhan.")
    draw_box(ax, 650, 390, 610, 308, "CLINICAL REGULATORY DISCLAIMER", "NON-MEDICAL DEVICE: For educational demonstration and machine learning research only.")
    fig.savefig(OUTPUT_DIR / "p5_about_limitations.png", bbox_inches="tight")
    plt.close()

    print("Successfully generated all 5 wireframes in powerbi/docs/layout_wireframes/")


if __name__ == "__main__":
    generate_wireframes()
