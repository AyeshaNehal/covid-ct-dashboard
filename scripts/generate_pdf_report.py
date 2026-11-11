"""
Generate professional PDF report for ResNet-18 model evaluation.
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

PDF_OUTPUT = Path("reports/model_report.pdf")
FIGURES_DIR = Path("outputs/figures")

def build_pdf():
    PDF_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(PDF_OUTPUT),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0B1F33"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0E7490"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
        spaceAfter=6,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#92400E"),
        backColor=colors.HexColor("#FEF3C7"),
        borderColor=colors.HexColor("#F59E0B"),
        borderWidth=1,
        borderPadding=8,
        spaceAfter=12,
        borderRadius=4,
    )

    story = []

    # Title & Header
    story.append(Paragraph("COVID-CT Classifier: ResNet-18 Technical Evaluation", title_style))
    story.append(Paragraph("Portfolio Demonstration & Machine Learning Diagnostics Report | Date: November 2026", subtitle_style))

    # Regulatory Disclaimer
    story.append(Paragraph(
        "<b>REGULATORY DISCLAIMER — NOT A MEDICAL DEVICE:</b> This document and associated software are strictly for engineering research, educational portfolio, and explainability benchmarking. It has NOT undergone FDA, CE, or NMPA clinical clearance and must NEVER be used for clinical decision-making or patient diagnosis.",
        disclaimer_style,
    ))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary & Benchmark Results", h1_style))
    story.append(Paragraph(
        "We evaluated a deep residual network (<b>ResNet-18</b>, 11.2M parameters) on <b>19,685 original chest CT cross-sectional slices</b> originating from the open-access iCTCF Wuhan Cohort. All slices were grouped into 25-slice proxy volume sequence blocks to mitigate adjacent slice leakage. Training ran for 12.87 minutes wall-clock on an AMD Ryzen AI 7 350 CPU, achieving <b>98.27% accuracy</b> on the held-out test split (3,000 slices).",
        body_style,
    ))

    # Benchmark KPI Table
    kpi_data = [
        ["Metric", "Score", "Evaluation Context"],
        ["Overall Test Accuracy", "98.27%", "2,948 / 3,000 correct test slices"],
        ["Macro F1-Score", "0.9806", "Harmonic mean across nCT, pCT, NiCT"],
        ["COVID-19 Positive Recall", "99.36%", "621 of 625 positive slices detected (4 false negatives)"],
        ["Macro Precision", "0.9775", "nCT: 0.9939, pCT: 0.9525, NiCT: 0.9862"],
        ["Wall-Clock Train Time", "12.87 min", "CPU execution on 13,735 full training slices"],
        ["Total Test Misclassifications", "52 slices", "1.73% overall test error rate"],
    ]
    t_kpi = Table(kpi_data, colWidths=[160, 90, 290])
    t_kpi.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B1F33")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 10))

    # Per-Class Table
    story.append(Paragraph("2. Per-Class Test Performance (3,000 Held-Out Slices)", h1_style))
    class_data = [
        ["Diagnostic Class", "Support", "Precision", "Recall (Sensitivity)", "F1-Score", "Failure Specifics"],
        ["Normal (nCT)", "1,500", "0.9939", "0.9813", "0.9876", "23 predicted as pCT, 5 as NiCT"],
        ["COVID-19 Positive (pCT)", "625", "0.9525", "0.9936", "0.9726", "2 predicted as nCT, 2 as NiCT (4 missed)"],
        ["Non-Informative (NiCT)", "875", "0.9862", "0.9771", "0.9816", "13 predicted as pCT, 7 as nCT"],
    ]
    t_class = Table(class_data, colWidths=[120, 50, 65, 95, 60, 150])
    t_class.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#14B8A6")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0B1F33")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    story.append(t_class)
    story.append(Spacer(1, 10))

    # Key Figures
    cm_fig = FIGURES_DIR / "confusion_matrix_resnet18.png"
    if cm_fig.exists():
        story.append(Paragraph("3. Confusion Matrix (Raw Counts & Normalized)", h1_style))
        story.append(RLImage(str(cm_fig), width=520, height=220))
        story.append(Spacer(1, 10))

    gradcam_fig = FIGURES_DIR / "gradcam_analysis.png"
    if gradcam_fig.exists():
        story.append(KeepTogether([
            Paragraph("4. Grad-CAM Salience Explainability (Layer 4 Bottleneck)", h1_style),
            Paragraph("Grad-CAM overlays show that model activations localize to peripheral ground-glass opacities in COVID-19 positive scans, while errors trigger on bronchial branching or diaphragmatic dome contours.", body_style),
            RLImage(str(gradcam_fig), width=520, height=260),
            Spacer(1, 10),
        ]))

    # Limitations & Discussion
    story.append(Paragraph("5. Critical Clinical & Methodological Limitations", h1_style))
    story.append(Paragraph(
        "• <b>Proxy-Grouped Splitting Caveat:</b> While contiguous sequence grouping eliminates adjacent slice overlap, unstripped patient metadata is missing from the Kaggle dataset. True patient-level overlap remains possible.<br/>"
        "• <b>Single 2D Slice Context:</b> Radiologists diagnose COVID-19 using 3D volumetric series. 2D evaluation lacks volumetric lesion measurement and longitudinal slice continuity.<br/>"
        "• <b>Scanner & Population Bias:</b> The cohort reflects Wuhan 2020 protocols and has not been externally tested across Western multi-scanner cohorts.",
        body_style,
    ))

    doc.build(story)
    print(f"PDF successfully built at {PDF_OUTPUT}!")

if __name__ == "__main__":
    build_pdf()
