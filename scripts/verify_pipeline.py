"""
Automated quality check script.
Verifies all project artifacts, data exports, models, and dashboard assets.
"""

from pathlib import Path
import sys

def verify_all():
    print("=== COVID-CT Dashboard Deliverables Audit ===")
    errors = []

    # 1. Directory Structure
    required_dirs = [
        "data/raw",
        "data/splits",
        "notebooks",
        "src",
        "outputs/checkpoints",
        "outputs/figures",
        "outputs/logs",
        "powerbi/data",
        "powerbi/theme",
        "powerbi/docs",
        "powerbi/screenshots",
        "app/pages",
        "app/components",
        "scripts",
        "tests",
    ]
    for d in required_dirs:
        p = Path(d)
        if not p.exists():
            errors.append(f"Missing directory: {d}")
        else:
            print(f"  [OK] Directory: {d}")

    # 2. Key Files
    required_files = [
        "requirements.txt",
        ".gitignore",
        "LICENSE",
        "README.md",
        "Makefile",
        "pytest.ini",
        "powerbi/theme/covid_ct_theme.json",
        "powerbi/docs/measures.dax",
        "powerbi/docs/data_dictionary.md",
        "powerbi/docs/BUILD_GUIDE.md",
        "powerbi/data/dataset_summary.csv",
        "data/splits/train.csv",
        "data/splits/val.csv",
        "data/splits/test.csv",
        "notebooks/01_eda.ipynb",
        "notebooks/02_training_walkthrough.ipynb",
        "notebooks/03_error_analysis.ipynb",
        "src/__init__.py",
        "src/config.py",
        "src/dataset.py",
        "src/splits.py",
        "src/model.py",
        "src/train.py",
        "src/evaluate.py",
        "src/gradcam.py",
        "src/utils.py",
        "app/main.py",
        "app/pages/1_Overview.py",
        "app/pages/2_Model_Performance.py",
        "app/pages/3_Error_Analysis.py",
        "app/pages/4_Dataset_Insights.py",
        "app/pages/5_Classify_Slice.py",
        "app/pages/6_About.py",
    ]
    for f in required_files:
        p = Path(f)
        if not p.exists():
            errors.append(f"Missing file: {f}")
        else:
            print(f"  [OK] File: {f}")

    # 3. Wireframes
    wireframes = [
        "powerbi/docs/layout_wireframes/p1_executive_overview.png",
        "powerbi/docs/layout_wireframes/p2_model_performance.png",
        "powerbi/docs/layout_wireframes/p3_error_analysis.png",
        "powerbi/docs/layout_wireframes/p4_dataset_insights.png",
        "powerbi/docs/layout_wireframes/p5_about_limitations.png",
    ]
    for w in wireframes:
        if not Path(w).exists():
            errors.append(f"Missing wireframe: {w}")
        else:
            print(f"  [OK] Wireframe: {w}")

    if errors:
        print("\nERRORS DETECTED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\nAll required core files and directories successfully verified!")


if __name__ == "__main__":
    verify_all()
