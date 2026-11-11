# Model Card: ResNet-18 COVID-CT Classifier

## 1. Model Details
- **Architecture Name:** Deep Residual Network 18 (`ResNet-18`, He et al., 2016)
- **Framework:** PyTorch 2.15.0+cpu, Torchvision 0.29.2
- **Parameters:** 11,178,051 total (100% trainable)
- **Pretrained Weights:** ImageNet-1K (`ResNet18_Weights.DEFAULT`)
- **Classification Head:** Linear layer projecting 512 embedding features to 3 output logits.
- **License:** MIT License

---

## 2. Intended Use & Deployment Scope
- **Intended Use:** Machine learning portfolio project, educational demonstrations, and computer vision explainability research.
- **Out-of-Scope / Prohibited Use:** Clinical triage, patient diagnosis, radiological second opinion, or deployment as a medical device. **NOT A MEDICAL DEVICE.**

---

## 3. Training Details
- **Hardware Platform:** AMD Ryzen AI 7 350 w/ Radeon 860M (8 Physical Cores, 16 Logical Processors), CPU execution.
- **Training Duration:** **12.87 minutes wall-clock** on full 13,735 training images.
- **Optimization Strategy:** AdamW (`lr=1e-4`, `weight_decay=1e-4`).
- **Learning Rate Schedule:** Cosine Annealing (`T_max=12`, `eta_min=1e-6`).
- **Loss Function:** Categorical Cross-Entropy Loss.
- **Batch Size:** 64.

---

## 4. Evaluated Performance Metrics (All 3,000 Test Slices)

| Metric | Empirical Score | Notes |
| :--- | :---: | :--- |
| **Overall Test Accuracy** | **98.27%** | 2,948 / 3,000 correct slices |
| **Macro F1-Score** | **0.9806** | Harmonic mean across all 3 classes |
| **Macro Precision** | **0.9775** | `nCT`: 0.9939, `pCT`: 0.9525, `NiCT`: 0.9862 |
| **Macro Recall (Sensitivity)** | **0.9840** | `nCT`: 0.9813, `pCT`: 0.9936, `NiCT`: 0.9771 |
| **COVID-19 Positive Recall** | **99.36%** | Missed only 4 out of 625 positive test slices |
| **Test Error Count** | **52 slices** | 1.73% overall test error rate |

---

## 5. Explainability Architecture
- **Method:** Gradient-weighted Class Activation Mapping (Grad-CAM, Selvaraju et al., 2017).
- **Target Extraction Layer:** `model.layer4[-1]` (final convolutional bottleneck).
- **Validation:** Heatmaps verify that activations localize specifically to peripheral ground-glass opacities and pulmonary consolidations rather than background table or text artifacts.

---

## 6. Critical Warnings & Ethical Notes
- **False Negative Risk:** 4 COVID-19 positive slices were missed (faint GGO, single-slice limitation).
- **False Positive Risk:** 23 healthy slices were predicted as positive due to hilus vascular congestion and dependent atelectasis.
- **Disclaimer:** This model must not be deployed in clinical environments without regulatory clearance, prospective clinical trials, and multi-center validation.
