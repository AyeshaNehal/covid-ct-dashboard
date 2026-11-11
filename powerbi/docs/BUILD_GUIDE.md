# Power BI Desktop Step-by-Step Build Guide: COVID-CT Analytics Dashboard

**Project Name:** `covid-ct-dashboard`  
**Canvas Resolution:** 1280 x 720 (16:9 Standard Widescreen)  
**Theme:** `powerbi/theme/covid_ct_theme.json`  
**Data Directory:** `powerbi/data/`  
**Disclaimer:** *Portfolio/Educational Project Only • NOT a medical device • NOT for clinical diagnostic use.*

---

## Part 1: Initial Setup & Theme Import

1. **Open Power BI Desktop**.
2. Go to **View** ribbon tab > **Themes** dropdown > **Browse for themes...**
3. Select and open: `powerbi/theme/covid_ct_theme.json`.
4. Verify default canvas background is Dark Navy (`#0B1F33`) and accent is Teal (`#14B8A6`).
5. In **File** > **Options and settings** > **Options** > **Current File** > **Report settings**:
   - Enable "Allow users to drillthrough in visuals".
   - Enable "Modern visual tooltips".

---

## Part 2: Data Loading & Power Query Setup

1. Click **Get Data** > **Text/CSV**.
2. Load the following 5 CSV files from `powerbi/data/`:
   - `predictions.csv`
   - `training_history.csv`
   - `model_comparison.csv`
   - `class_metrics.csv`
   - `dataset_summary.csv`
3. In the Power Query Editor (**Transform Data**):
   - **predictions**:
     - Ensure `conf`, `p_nCT`, `p_pCT`, `p_NiCT`, `mean_brightness` are set to **Decimal Number**.
     - Ensure `correct`, `width`, `height` are set to **Whole Number**.
     - Ensure `image_id`, `path`, `split`, `model`, `true_label`, `pred_label`, `error_type` are set to **Text**.
     - Add Custom Column: `Outcome` = `if [correct] = 1 then "Correct" else "Misclassified"` (Type: Text).
     - Add Custom Column: `Confidence Bin` =
       ```powerquery
       if [conf] < 0.5 then "<50%"
       else if [conf] < 0.6 then "50-60%"
       else if [conf] < 0.7 then "60-70%"
       else if [conf] < 0.8 then "70-80%"
       else if [conf] < 0.9 then "80-90%"
       else "90-100%"
       ```
   - **training_history**:
     - `train_loss`, `train_acc`, `val_loss`, `val_acc`, `lr` -> **Decimal Number**.
     - `epoch` -> **Whole Number**.
   - **model_comparison**:
     - `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `train_time_min` -> **Decimal Number**.
     - `params` -> **Whole Number**.
   - **class_metrics**:
     - `precision`, `recall`, `f1` -> **Decimal Number**.
     - `support` -> **Whole Number**.
   - Click **Close & Apply**.

---

## Part 3: Data Modeling & Star Schema

1. Switch to **Model View**.
2. Create calculated dimension table for Confidence Bins (Modeling tab > **New Table**):
   ```dax
   ConfidenceBins = 
   DATATABLE(
       "Confidence Bin", STRING,
       "SortOrder", INTEGER,
       {
           {"<50%", 1},
           {"50-60%", 2},
           {"60-70%", 3},
           {"70-80%", 4},
           {"80-90%", 5},
           {"90-100%", 6}
       }
   )
   ```
   - Select column `Confidence Bin`, click **Sort by column** > choose `SortOrder`.
3. Create Relationship:
   - `ConfidenceBins`[`Confidence Bin`] (1) to `predictions`[`Confidence Bin`] (*) [Single cross-filter direction].
4. Load DAX Measures:
   - Create a dedicated measure table: **Home** > **Enter Data** > Name it `_Measures`.
   - Copy all measures from `powerbi/docs/measures.dax` into `_Measures`.

---

## Part 4: Layout & Visual Specifications (1280 x 720)

### Universal Page Header (Present on all pages)
- **Top Header Bar**: X=0, Y=0, Width=1280, Height=60. Background: `#071524`.
- **Title Text**: "COVID-CT DEEP LEARNING ANALYTICS DASHBOARD" (Segoe UI Bold 18pt, `#F8FAFC`, X=20, Y=12).
- **Subtitle / Disclaimer Tag**: "RESEARCH & PORTFOLIO DEMO • NOT A MEDICAL DEVICE" (Segoe UI 9pt Bold, `#F59E0B`, X=20, Y=36).
- **Navigation Buttons**: (Overview, Models, Errors, Dataset, About) horizontally aligned on right (X=780 to X=1260, Y=12, Height=36).

---

### Page 1: Executive Overview (`P1_Overview`)
- **Key Insight Card**: X=20, Y=70, Width=1240, Height=40. Text:
  *"MobileNetV3 achieves highest test macro-F1 while maintaining 4.2M parameters. Leakage-aware splitting prevents patient cross-contamination."*
- **Slicers Panel**: X=20, Y=120, Width=220, Height=570.
  - Slicer 1: `predictions[model]` (Dropdown, Single select).
  - Slicer 2: `predictions[split]` (Default: "test").
- **KPI Cards Row**: Y=120, Height=95.
  - Card 1: `[Accuracy]` (Format: 0.0%) -> X=260, Width=230. Accent: Teal `#14B8A6`.
  - Card 2: `[Macro F1]` or `[Class F1]` -> X=510, Width=230. Accent: Blue `#3B82F6`.
  - Card 3: `[Total Images]` (Format: #,##0) -> X=760, Width=230. Accent: White `#F8FAFC`.
  - Card 4: `[High-Confidence Errors]` -> X=1010, Width=250. Accent: Red `#F87171`.
- **Visual 1 (Donut Chart - Class Balance)**: X=260, Y=230, Width=480, Height=260.
  - Legend: `predictions[true_label]`, Values: `[Total Images]`.
- **Visual 2 (Clustered Column Chart - Accuracy by Split)**: X=760, Y=230, Width=500, Height=260.
  - X-Axis: `predictions[split]`, Y-Axis: `[Accuracy]`, Legend: `predictions[model]`.
- **Visual 3 (Bar Chart - Model Architecture Comparison)**: X=260, Y=505, Width=1000, Height=185.
  - Y-Axis: `model_comparison[model]`, X-Axis: `[Accuracy]` & `[macro_f1]`.

---

### Page 2: Model Performance (`P2_Model_Performance`)
- **Key Insight**: X=20, Y=70, Width=1240, Height=40.
  *"Non-informative (NiCT) slices show >98% sensitivity; minimal confusion occurs between normal (nCT) and mild COVID-19 (pCT)."*
- **Visual 1 (Matrix - Confusion Matrix)**: X=20, Y=120, Width=440, Height=320.
  - Rows: `predictions[true_label]`
  - Columns: `predictions[pred_label]`
  - Values: `[Total Images]`
  - Conditional formatting: Background color scale (Low `#0B1F33` -> High `#14B8A6`).
- **Visual 2 (Line Chart - Training & Validation Curves)**: X=480, Y=120, Width=780, Height=320.
  - Source: `training_history` table.
  - X-Axis: `epoch`, Y-Axis: `train_loss`, `val_loss`, `train_acc`, `val_acc`.
- **Visual 3 (Table - Class Metrics Breakdown)**: X=20, Y=455, Width=620, Height=235.
  - Columns: `class`, `precision`, `recall`, `f1`, `support`.
- **Visual 4 (Scatter / Line - ROC & PR Summary)**: X=660, Y=455, Width=600, Height=235.
  - Comparison of TPR vs FPR and Precision vs Recall.

---

### Page 3: Error Analysis (`P3_Error_Analysis`)
- **Key Insight**: X=20, Y=70, Width=1240, Height=40.
  *"High-confidence errors are predominantly driven by motion artifact or low-contrast peripheral ground-glass opacities."*
- **KPI Row**: Y=120, Height=85.
  - `[Error Count]` (X=20, Width=290, Coral `#F87171`)
  - `[Error Rate]` (X=330, Width=290, Amber `#F59E0B`)
  - `[Avg Confidence (Wrong)]` (X=640, Width=290, Teal `#14B8A6`)
  - `[High-Confidence Errors]` (X=950, Width=310, Coral `#F87171`)
- **Visual 1 (Bar Chart - Misclassification Transitions)**: X=20, Y=220, Width=440, Height=470.
  - Y-Axis: `predictions[error_type]` (Filter out "None"), X-Axis: `[Total Images]`. Color: Coral `#F87171`.
- **Visual 2 (Column Chart - Confidence Distribution: Correct vs Error)**: X=480, Y=220, Width=780, Height=220.
  - X-Axis: `ConfidenceBins[Confidence Bin]`, Y-Axis: `[Total Images]`, Legend: `predictions[Outcome]`.
- **Visual 3 (Table - Error Slices Deep Dive)**: X=480, Y=455, Width=780, Height=235.
  - Columns: `image_id`, `true_label`, `pred_label`, `conf`, `path`, `mean_brightness`.
  - Sorted by `conf` descending. Drill-through enabled to Image Detail page.

---

### Page 4: Dataset Insights (`P4_Dataset_Insights`)
- **Key Insight**: X=20, Y=70, Width=1240, Height=40.
  *"Raw slice dimensions are standardized at 224x224; brightness distribution shows distinct bimodal separation for NiCT."*
- **Visual 1 (Donut Chart - Pathology Class Shares)**: X=20, Y=120, Width=380, Height=270.
- **Visual 2 (Column Chart - Slices per Patient/Group Distribution)**: X=420, Y=120, Width=420, Height=270.
- **Visual 3 (Card Cluster - EDA Health Check)**: X=860, Y=120, Width=400, Height=270.
  - Corrupt Images: `0`
  - Exact Duplicates: `0`
  - Total Patient Groups: `#`
- **Visual 4 (Histogram - Mean Brightness by Class)**: X=20, Y=405, Width=610, Height=285.
- **Visual 5 (Stacked Bar - Split Partition Composition)**: X=650, Y=405, Width=610, Height=285.

---

### Page 5: About, Methodology & Limitations (`P5_About`)
- **Text Box 1 (Project Overview & Credits)**: X=20, Y=70, Width=610, Height=300.
  - Dataset: Abu Zahid Bin Aziz (CC BY 4.0), originating from iCTCF (Ning et al., 2020).
  - Models: MobileNetV3 (Howard et al., 2019), ResNet-18 (He et al., 2015), EfficientNet-B0 (Tan & Le, 2019).
- **Text Box 2 (Splitting & Leakage Prevention)**: X=650, Y=70, Width=610, Height=300.
  - Explains GroupShuffleSplit (70/15/15) by patient/sequence clustering to avoid slice-level data leakage.
- **Text Box 3 (Critical Limitations)**: X=20, Y=390, Width=610, Height=290.
  - Single 2D slice classification rather than 3D volumetric context.
  - Cohort source limited to two hospitals in Wuhan during early 2020.
  - Absence of external geographic and scanner validation sets.
- **Callout Banner (Medical Device Disclaimer)**: X=650, Y=390, Width=610, Height=290.
  - High-visibility amber border with the strict clinical disclaimer text.

---

## Part 5: Wireframe Layout References
Rendered PNG layout guides matching these visual coordinates are stored in:
`powerbi/docs/layout_wireframes/`
- `p1_executive_overview.png`
- `p2_model_performance.png`
- `p3_error_analysis.png`
- `p4_dataset_insights.png`
- `p5_about_limitations.png`
