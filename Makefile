# Makefile for covid-ct-dashboard
# Disclaimer: Portfolio/Educational Project Only • NOT A MEDICAL DEVICE

VENV := .venv
PYTHON := $(VENV)/Scripts/python
PIP := $(VENV)/Scripts/pip
STREAMLIT := $(VENV)/Scripts/streamlit
PYTEST := $(VENV)/Scripts/pytest

.PHONY: help setup data splits train evaluate dashboard test clean wireframes

help:
	@echo "Available commands:"
	@echo "  make setup      : Create virtual environment and install dependencies"
	@echo "  make data       : Run EDA and verify data directory"
	@echo "  make splits     : Generate leakage-aware train/val/test splits"
	@echo "  make train      : Train models (MobileNetV3, ResNet18, EfficientNet-B0)"
	@echo "  make evaluate   : Run full evaluation and export Power BI tables"
	@echo "  make dashboard  : Launch Streamlit analytics dashboard"
	@echo "  make test       : Run test suite via pytest"
	@echo "  make wireframes : Render Power BI layout wireframe PNGs"

setup:
	py -3.11 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

data:
	$(PYTHON) scripts/run_eda.py

splits:
	$(PYTHON) -m src.splits

train:
	$(PYTHON) -m src.train --all-models

evaluate:
	$(PYTHON) -m src.evaluate

dashboard:
	$(STREAMLIT) run app/main.py

test:
	$(PYTEST) tests/ -v

wireframes:
	$(PYTHON) scripts/generate_wireframes.py

clean:
	rm -rf __pycache__ .pytest_cache outputs/logs/*.log
