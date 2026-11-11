#!/usr/bin/env bash
# scripts/run_all.sh
# End-to-end execution of covid-ct-dashboard pipeline
#
# DISCLAIMER: Portfolio/Educational Project Only • NOT A MEDICAL DEVICE

set -euo pipefail

echo "=========================================================="
echo " COVID-CT Analytics Pipeline: End-to-End Orchestrator"
echo "=========================================================="

# Check venv
if [ -d ".venv/Scripts" ]; then
    PYTHON=".venv/Scripts/python"
    PYTEST=".venv/Scripts/pytest"
    STREAMLIT=".venv/Scripts/streamlit"
else
    PYTHON="python"
    PYTEST="pytest"
    STREAMLIT="streamlit"
fi

echo "[1/6] Running Exploratory Data Analysis (EDA)..."
$PYTHON scripts/run_eda.py

echo "[2/6] Generating Leakage-Aware Splits (70/15/15)..."
$PYTHON -m src.splits

echo "[3/6] Running Unit Tests (Pre-training verification)..."
$PYTEST tests/test_model.py tests/test_splits.py tests/test_dataset.py -v

echo "[4/6] Training Architectures (MobileNetV3, ResNet18, EfficientNet-B0)..."
$PYTHON -m src.train --all-models

echo "[5/6] Comprehensive Evaluation & Power BI Exports..."
$PYTHON -m src.evaluate

echo "[6/6] Generating Wireframes..."
$PYTHON scripts/generate_wireframes.py

echo "=== Pipeline Completed Successfully! ==="
echo "Launch dashboard: make dashboard"
