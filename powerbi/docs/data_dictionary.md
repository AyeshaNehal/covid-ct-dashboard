# COVID-CT Dashboard - Power BI Data Dictionary

**Project Name:** `covid-ct-dashboard`  
**Disclaimer:** *This dataset and analytics dashboard are for educational and portfolio demonstration only. It is NOT a medical device and must NOT be used for clinical decision making or diagnosis.*

---

## 1. Table: `predictions.csv`
Primary granular fact table storing individual slice predictions across splits and model architectures.

| Column Name | Data Type | Description | Allowed Values / Range |
| :--- | :--- | :--- | :--- |
| `image_id` | Text | Unique identifier of the slice (derived from filename stem) | E.g. `patient10_slice4`, `nCT_012` |
| `path` | Text | Relative filesystem path to the CT slice image | E.g. `data/raw/nCT/slice_001.png` |
| `split` | Text | Dataset partition to prevent leakage | `train`, `val`, `test` |
| `model` | Text | Neural network architecture evaluated | `mobilenet_v3_large`, `resnet18`, `efficientnet_b0` |
| `true_label` | Text | Ground truth diagnostic category | `nCT`, `pCT`, `NiCT` |
| `pred_label` | Text | Model's top predicted diagnostic category | `nCT`, `pCT`, `NiCT` |
| `conf` | Decimal (Float) | Confidence / maximum softmax probability | `0.0000` to `1.0000` |
| `p_nCT` | Decimal (Float) | Predicted probability for Normal CT (`nCT`) | `0.0000` to `1.0000` |
| `p_pCT` | Decimal (Float) | Predicted probability for COVID-19 Positive (`pCT`) | `0.0000` to `1.0000` |
| `p_NiCT` | Decimal (Float) | Predicted probability for Non-informative CT (`NiCT`) | `0.0000` to `1.0000` |
| `correct` | Whole Number | Binary correctness indicator (1 = True, 0 = False) | `1`, `0` |
| `error_type` | Text | Misclassification transition category | `None`, `nCT->pCT`, `pCT->nCT`, etc. |
| `width` | Whole Number | Image slice width in pixels | E.g. `224`, `512` |
| `height` | Whole Number | Image slice height in pixels | E.g. `224`, `512` |
| `mean_brightness`| Decimal (Float) | Normalized average grayscale intensity | `0.0000` to `1.0000` |

---

## 2. Table: `training_history.csv`
Per-epoch training and validation loss and accuracy progression.

| Column Name | Data Type | Description | Allowed Values / Range |
| :--- | :--- | :--- | :--- |
| `model` | Text | Neural network architecture | `mobilenet_v3_large`, `resnet18`, `efficientnet_b0` |
| `epoch` | Whole Number | Training epoch number (1-indexed) | `1` to `25` |
| `train_loss` | Decimal (Float) | Cross-entropy training loss | `>= 0.0000` |
| `train_acc` | Decimal (Float) | Training partition accuracy | `0.0000` to `1.0000` |
| `val_loss` | Decimal (Float) | Validation partition cross-entropy loss | `>= 0.0000` |
| `val_acc` | Decimal (Float) | Validation partition accuracy | `0.0000` to `1.0000` |
| `lr` | Decimal (Float) | Learning rate applied during epoch | E.g. `0.0001` |

---

## 3. Table: `model_comparison.csv`
Benchmark summary metrics across models and dataset splits.

| Column Name | Data Type | Description | Allowed Values / Range |
| :--- | :--- | :--- | :--- |
| `model` | Text | Model architecture name | `mobilenet_v3_large`, `resnet18`, `efficientnet_b0` |
| `split` | Text | Dataset partition | `train`, `val`, `test` |
| `accuracy` | Decimal (Float) | Overall classification accuracy | `0.0000` to `1.0000` |
| `macro_precision`| Decimal (Float) | Unweighted macro-average precision | `0.0000` to `1.0000` |
| `macro_recall` | Decimal (Float) | Unweighted macro-average recall | `0.0000` to `1.0000` |
| `macro_f1` | Decimal (Float) | Unweighted macro-average F1-score | `0.0000` to `1.0000` |
| `params` | Whole Number | Total parameter count of network | E.g. `4,213,971` |
| `train_time_min` | Decimal (Float) | Wall-clock training duration in minutes | `>= 0.0` |

---

## 4. Table: `class_metrics.csv`
Class-stratified evaluation metrics for detailed pathology analysis.

| Column Name | Data Type | Description | Allowed Values / Range |
| :--- | :--- | :--- | :--- |
| `model` | Text | Model architecture name | `mobilenet_v3_large`, `resnet18`, `efficientnet_b0` |
| `split` | Text | Dataset partition | `train`, `val`, `test` |
| `class` | Text | Target diagnostic class | `nCT`, `pCT`, `NiCT` |
| `precision` | Decimal (Float) | Class precision (Positive Predictive Value) | `0.0000` to `1.0000` |
| `recall` | Decimal (Float) | Class recall (Sensitivity) | `0.0000` to `1.0000` |
| `f1` | Decimal (Float) | Harmonic mean of precision and recall | `0.0000` to `1.0000` |
| `support` | Whole Number | True positive ground truth count | `>= 0` |

---

## 5. Table: `dataset_summary.csv`
Exploratory Data Analysis (EDA) summary of the underlying raw slice archive.

| Column Name | Data Type | Description | Allowed Values / Range |
| :--- | :--- | :--- | :--- |
| `class` | Text | Pathology label | `nCT`, `pCT`, `NiCT`, `Total` |
| `total_slices` | Whole Number | Count of images in the class | `>= 0` |
| `unique_groups` | Whole Number | Count of distinct patient / proxy clusters | `>= 0` |
| `mean_width` | Decimal (Float) | Mean pixel width | E.g. `224.0` |
| `mean_height` | Decimal (Float) | Mean pixel height | E.g. `224.0` |
| `mean_brightness`| Decimal (Float) | Mean pixel intensity [0 - 1] | `0.0000` to `1.0000` |
| `corrupt_images` | Whole Number | Count of corrupt or unreadable files | `>= 0` |
| `exact_duplicates`| Whole Number | Count of byte-exact duplicate images | `>= 0` |
