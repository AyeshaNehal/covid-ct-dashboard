# Dataset Card: Kaggle Preprocessed CT Scans for COVID-19 (iCTCF Cohort)

## 1. Dataset Summary
- **Dataset Name:** Preprocessed CT Scans for COVID-19
- **Source Publisher:** Abu Zahid Bin Aziz (Kaggle), licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Primary Clinical Origin:** iCTCF Consortium (Ning et al., 2020), Union Hospital & Tongji Medical College, Huazhong University of Science and Technology, Wuhan, Hubei, China.
- **Total Slices:** 19,685 original JPEG CT images.
- **Classes:**
  - `nCT` (Normal Chest CT): 9,979 images (50.7%)
  - `pCT` (COVID-19 Positive Chest CT): 4,001 images (20.3%)
  - `NiCT` (Non-informative CT Scan): 5,705 images (29.0%)

---

## 2. Data Splits & Partitioning Strategy

| Partition | Total Images | Normal (`nCT`) | COVID-19 (`pCT`) | Non-Informative (`NiCT`) | Proportion |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 13,735 | 6,979 | 2,776 | 3,980 | 69.8% |
| **Validation** | 2,950 | 1,500 | 600 | 850 | 15.0% |
| **Test (Held-Out)** | 3,000 | 1,500 | 625 | 875 | 15.2% |
| **Total** | **19,685** | **9,979** | **4,001** | **5,705** | **100.0%** |

- **Grouping Method:** Sequence block clustering (25 contiguous slices per proxy volume group).
- **Splitting Algorithm:** Two-stage stratified `GroupShuffleSplit` (Seed 42).
- **Group Isolation:** Zero sequence group intersection across splits (`Train ∩ Val ∩ Test = ∅`).

---

## 3. Data Preprocessing & Augmentations
- **Source Format:** 8-bit grayscale stored as 3-channel JPEG files.
- **Image Geometry:** Width = 512 px, Height = 512 px across all slices.
- **Mean Pixel Brightness:** 0.286 across cohort.
- **Integrity Check:** 0 corrupt files; 55 byte-level exact duplicates identified in `NiCT` slices.
- **Training Augmentations:** `RandomResizedCrop(224)`, `RandomHorizontalFlip(p=0.5)`, `RandomRotation(10°)`, `ColorJitter(brightness=0.15, contrast=0.15)`.
- **Evaluation Transforms:** `Resize(256)`, `CenterCrop(224)`, ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

---

## 4. Known Limitations & Biases
1. **De-identification & Patient ID Stripping:** Kaggle re-indexed filenames sequentially without embedding original DICOM tags. True patient identifiers cannot be reconstructed.
2. **Scanner & Acquisition Protocol Uniformity:** Images reflect protocols and scanners from Wuhan healthcare facilities during early 2020.
3. **Class Definition Ambiguity (`NiCT`):** The non-informative class consists of slices with minimal lung parenchyma (e.g. neck, upper abdomen, scout scans). Boundary slices between `nCT` and `NiCT` exhibit inherent label subjectivity.

---

## 5. Regulatory Disclaimer
This dataset and its derived splits are intended exclusively for academic, machine learning research, and educational purposes. **NOT FOR MEDICAL USE.**
