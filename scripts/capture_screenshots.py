"""
Capture full-page, pixel-perfect screenshots of all Streamlit dashboard pages.
Saves to outputs/figures/dashboard/ and powerbi/screenshots/.
"""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8501"
OUTPUT_DIRS = [
    Path("outputs/figures/dashboard"),
    Path("powerbi/screenshots"),
]

for d in OUTPUT_DIRS:
    d.mkdir(parents=True, exist_ok=True)

PAGES = [
    ("main", "", "dashboard_hero_main.png"),
    ("overview", "Overview", "01_executive_overview.png"),
    ("model_perf", "Model_Performance", "02_model_performance.png"),
    ("error_analysis", "Error_Analysis", "03_error_analysis.png"),
    ("dataset_insights", "Dataset_Insights", "04_dataset_insights.png"),
    ("classify_slice", "Classify_Slice", "05_classify_slice.png"),
    ("reports", "Reports", "06_reports_assets.png"),
    ("about", "About", "07_about_limitations.png"),
]

def capture_all():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        for key, path_suffix, filename in PAGES:
            target_url = f"{BASE_URL}/{path_suffix}" if path_suffix else BASE_URL
            print(f"Navigating to {target_url}...")
            try:
                page.goto(target_url, wait_until="networkidle", timeout=25000)
                time.sleep(3.5)  # Wait for charts to fully render

                for out_dir in OUTPUT_DIRS:
                    dest = out_dir / filename
                    page.screenshot(path=str(dest), full_page=True)
                    print(f"  -> Saved {dest}")
            except Exception as e:
                print(f"  -> Failed to capture {target_url}: {e}")

        browser.close()
    print("All screenshots successfully captured!")

if __name__ == "__main__":
    capture_all()
