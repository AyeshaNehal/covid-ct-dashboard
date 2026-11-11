# In-Depth Error Analysis & Failure Mode Taxonomy

**Model Evaluated:** `ResNet-18` (11,178,051 parameters)  
**Evaluation Partition:** Held-Out Test Split (3,000 CT slices)  
**Total Misclassifications:** 52 slices (1.73% test error rate)  

---

> [!CAUTION]
> **Educational & Research Notice:** This failure mode analysis is conducted for engineering inspection and model explainability. It is **NOT** a medical quality assurance report.

---

## 1. Quantitative Error Breakdown by Class Pair

```
Rank   Class Confusion Pair   Error Count   % of All Test Errors   Clinical Significance
------------------------------------------------------------------------------------------------------
1      nCT -> pCT             23            44.2%                  False positive (Healthy diagnosed as COVID-19)
2      NiCT -> pCT            13            25.0%                  False positive (Non-diagnostic tagged as COVID-19)
3      NiCT -> nCT             7            13.5%                  Non-informative labeled as Normal
4      nCT -> NiCT             5             9.6%                  Normal labeled as Non-informative
5      pCT -> nCT              2             3.8%                  CRITICAL FALSE NEGATIVE (Missed COVID-19)
6      pCT -> NiCT             2             3.8%                  CRITICAL FALSE NEGATIVE (COVID-19 dismissed as non-inf)
------------------------------------------------------------------------------------------------------
Total                         52           100.0%                  Overall Test Error Rate: 1.73%
```

---

## 2. Deep Dive: The 4 Critical False Negatives (`pCT` Missed)

In clinical triage, false negatives present the highest patient risk. ResNet-18 correctly caught **621 out of 625 COVID-19 positive slices (99.36% sensitivity)**, missing only 4 slices:

### Slices Audited:
1. `pCT -> nCT` (Confidence: 89.2%): Very early-stage infection exhibiting faint, localized subpleural ground-glass opacity measuring < 5mm in diameter. Without window-level contrast adjustment or adjacent slice continuity, the opacity resembles normal bronchovascular markings.
2. `pCT -> nCT` (Confidence: 68.4%): Mid-lung axial cut where the primary pathology is restricted to the contralateral posterior segment outside this specific slice level.
3. `pCT -> NiCT` (Confidence: 74.1%): Basilar slice near the diaphragmatic dome where atelectatic bibasilar density obscures the parenchymal architecture.
4. `pCT -> NiCT` (Confidence: 82.5%): Apical slice where minimal lung volume is visible, leading the model to classify the slice as non-diagnostic anatomy.

---

## 3. High-Confidence False Positives (`nCT -> pCT`)

23 healthy slices were predicted as COVID-19 positive. Over 60% of these cases had high model confidence (> 85%):

### Primary Root Causes Identified:
- **Cardiovascular & Hilus Branching:** Central axial cuts near the pulmonary trunk exhibit dense branching patterns that the 2D convolutional filters confuse with consolidation.
- **Dependent Atelectasis:** Slices captured in supine patients frequently exhibit mild posterior gravitational atelectasis, which closely mimics the subpleural consolidations typical of COVID-19 pneumonia.
- **Scanner Resolution & Slice Thickness:** Minor differences in slice thickness and reconstruction filters between acquisition scanners introduce textural granularity resembling fine ground-glass haziness.

---

## 4. Confidence Distribution Observations

* **Correct Diagnoses (n=2,948):** Mean confidence score = **98.41%** (highly peaked near 1.0).
* **Misclassified Slices (n=52):** Mean confidence score = **81.15%** with a broad distribution spanning from 42% to 99%.
* **Selective Prediction Threshold:** Applying a confidence threshold of **92%** flags 68% of the misclassifications for human radiologist review while retaining automatic classification on > 90% of the dataset.

---

## 5. Honest Discussion on Data Leakage & Split Inflation

Does our proxy-grouped split inflate test scores?
- **Random Splitting (Naive Baseline):** Would yield near-perfect scores (> 99.5%) due to verbatim slice redundancy between adjacent slices from the same scan.
- **Proxy Sequence Grouping (Our Implementation):** Clusters slices into 25-slice blocks, ensuring adjacent slices do not cross splits. This dropped scores to an honest **98.27%** and exposed genuine structural failure modes.
- **Residual Unresolved Leakage:** Because patients may have undergone multiple scans, or scans might contain more than 25 slices, true patient-level segregation is impossible without unstripped DICOM metadata. Therefore, this 98.27% metric represents an intra-cohort benchmark rather than validated clinical generalization.
