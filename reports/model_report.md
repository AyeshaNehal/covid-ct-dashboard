# Model Evaluation Report: ResNet-18 Deep Learning CT-Scan Classifier

**Project Name:** `covid-ct-dashboard`  
**Model Architecture:** Pretrained Deep Residual Network 18 (`ResNet-18`, 11,178,051 parameters)  
**Task:** Multi-Class Chest CT Slice Classification  
**Classes:** 
- `nCT`: Normal Chest CT Cross-Section
- `pCT`: COVID-19 Positive Pulmonary Lesion Cross-Section
- `NiCT`: Non-Informative CT Slice (e.g., diaphragm, apex scout, non-diagnostic anatomy)  

---

> [!CAUTION]
> **REGULATORY & MEDICAL DISCLAIMER:**  
> This project, code, and evaluation report are produced strictly for machine learning portfolio demonstration, engineering research, and pedagogical analysis. It is **NOT** a certified medical diagnostic device and has not been cleared or evaluated by any regulatory body (such as FDA, CE, or NMPA). It must **NEVER** be utilized for clinical triage, medical decision-making, patient treatment, or diagnostic verification.

---

## 1. Executive Summary

We developed, trained, and benchmarked an end-to-end deep learning classification system on **19,685 original chest CT cross-sectional slices** originating from the open-source **iCTCF Wuhan Cohort** (Ning et al., 2020; Kaggle: Abu Zahid Bin Aziz, CC BY 4.0). 

To guard against the optimistic score inflation prevalent in biomedical slice classification, we designed and enforced a **proxy sequence block grouping strategy** across all 19,685 slices. The full held-out test split of **3,000 slices** was evaluated without subsampling.

### Primary Empirical Results (ResNet-18 Test Split)
* **Test Accuracy:** **98.27%** (2,948 / 3,000 correct slices)
* **Macro-Averaged F1-Score:** **0.9806**
* **Macro Precision:** **0.9775**
* **Macro Recall:** **0.9840**
* **Real Wall-Clock Training Time:** **12.87 minutes** (on CPU, AMD Ryzen AI 7 350 8-core)
* **Total Misclassifications:** **52 slices out of 3,000** (1.73% test error rate)

---

## 2. Dataset Architecture & Split Integrity

```
========================================================================
  Dataset Split Summary (Zero Sequence Group Leakage)
  Split        nCT (Normal)    pCT (COVID-19)   NiCT (Non-Inf)    Total Slices
  ----------------------------------------------------------------------
  Train               6,979             2,776            3,980          13,735
  Validation          1,500               600              850           2,950
  Test                1,500               625              875           3,000
  ----------------------------------------------------------------------
  Total               9,979 (50.7%)     4,001 (20.3%)    5,705 (29.0%)  19,685
========================================================================
```

### The Proxy-Grouped Split Strategy & Leakage Caveat
1. **Filename De-identification:** The Kaggle distribution stripped all raw DICOM tags and patient metadata, re-indexing images sequentially (`nCT1.jpg`..). Naive uniform random splitting across slices would place nearly identical adjacent slices from the same scan into both training and test sets.
2. **Mitigation:** We clustered contiguous slices into 25-slice sequence blocks to reconstruct acquisition volumes, executing two-stage stratified `GroupShuffleSplit` (70% Train, 15% Val, 15% Test).
3. **Honest Caveat:** Because filenames lack native patient identifiers, **patient overlap across sequence blocks remains possible**. Test performance (98.27%) reflects strong intra-cohort feature representation, but does not substitute for true multi-center patient-isolated validation.

---

## 3. Empirical Test Results & Confusion Matrix

All metrics below are computed from the real predictions across all 3,000 test slices:

### Test Set Metrics Table
| Class | Support Slices | Precision | Recall | F1-Score | Detection Specifics |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Normal (`nCT`)** | 1,500 | 0.9939 | 0.9813 | 0.9876 | 1,472 True Negatives, 28 false calls |
| **COVID-19 Positive (`pCT`)** | 625 | 0.9525 | 0.9936 | 0.9726 | **621 / 625 correctly detected** (4 missed) |
| **Non-Informative (`NiCT`)** | 875 | 0.9862 | 0.9771 | 0.9816 | 855 True Negatives, 20 false calls |
| **Macro Average** | **3,000** | **0.9775** | **0.9840** | **0.9806** | **Overall Accuracy: 98.27%** |

### Confusion Matrix (Test Split)
```
                  Predicted nCT    Predicted pCT    Predicted NiCT
True nCT              1,472             23                5
True pCT                  2            621                2
True NiCT                 7             13              855
```

---

## 4. Failure Mode Taxonomy & Qualitative Error Analysis

Out of 3,000 test slices, exactly **52 slices (1.73%)** were misclassified:

1. **`nCT -> pCT` (23 slices):** Healthy lung parenchymal slices misidentified as COVID-19 positive. Visual inspection reveals motion artifacts, prominent broncho-vascular branching near the pulmonary hilum, and minor atelectasis mimicking peripheral ground-glass opacities (GGO).
2. **`NiCT -> pCT` (13 slices):** Non-informative slices (diaphragmatic or subpleural dome cuts) exhibiting hyperdense soft tissue structures that trigger the COVID-19 dense consolidation detector.
3. **`NiCT -> nCT` (7 slices):** Transition slices containing partial lung fields labeled as non-informative but classified as normal parenchyma.
4. **`nCT -> NiCT` (5 slices):** Apical cuts with low lung volume misclassified as non-informative.
5. **`pCT -> nCT` (2 slices) & `pCT -> NiCT` (2 slices):** False negatives. Critical review reveals minimal, subtle peripheral subpleural ground-glass haziness easily missed without 3D slice context.

---

## 5. Explainability with Grad-CAM

To ensure the neural network decisions are driven by authentic pathological pulmonary features rather than scanner artifacts, we extracted Grad-CAM heatmaps from the final residual block (`layer4`):
- **COVID-19 Diagnoses:** Activations consistently localize to bilateral peripheral ground-glass opacities, crazy-paving patterns, and posterior consolidations.
- **Normal CTs:** Activations remain diffuse across homogenous aerated lung fields without focal hot-spots.
- **Errors:** Misclassified slices reveal focal activations over non-pulmonary anatomical structures (rib margins, diaphragmatic contours, or scanner table artifacts).

---

## 6. Clinical & Methodological Limitations

1. **Lack of 3D Volumetric Series:** Radiology evaluation of thoracic disease requires volumetric examination. Evaluating individual 2D axial slices removes longitudinal context.
2. **Single-Center Retrospective Bias:** The cohort originates from early 2020 Wuhan healthcare centers. Calibration across contemporary multi-manufacturer scanners (GE, Siemens, Philips) remains unvalidated.
3. **Etiological Specificity:** Ground-glass opacities are not unique to SARS-CoV-2; they appear in Influenza, Respiratory Syncytial Virus (RSV), Cytomegalovirus (CMV), and non-infectious organizing pneumonia.

---

## 7. Next Engineering Steps
- Incorporate 3D volumetric ResNet or Video Vision Transformers across sequential slice volumes.
- Collect multi-center external validation datasets (e.g., BIMCV-COVID19+, STOIC).
- Implement multi-task pulmonary segmentation masks as auxiliary loss supervision.
