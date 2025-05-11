# 🧠 DSViT: Densely Scaled Vision Transformer for Brain Tumor Detection

This repository contains the full implementation of **DSViT**, a transformer-based detection architecture for brain tumor localization and classification using the FigShare dataset.

---

## 🚀 Key Highlights

- 🔁 **Densely Connected Transformer Blocks**
- 📐 **Multi-scale Token Embedding**
- 🧠 **Neuro-Attentive Pre-Encoder**
- 🧪 **Efficient Training on Medium Medical Datasets**
- 📦 **Bounding Box + Class Label Output**

---

## 📁 Project Structure
BRAINTUMOR/
├── DATASET/
│ ├── FILES/ # .mat files (MRI slices)
│ └── cvind.mat # Fold split file
│
├── DSVIT/
│ ├── dsvit/
│ │ ├── model.py # Main + ablation variants
│ │ ├── dataset.py # Custom PyTorch dataset
│ │ ├── preprocessor.py # CLAHE, Skull Strip, etc.
│ │ ├── hooks.py # For attention visualizations
│ │ └── utils.py # Confusion matrix, ROC, etc.
│ ├── model-weight/ # Saved model weights
│ └── notebooks/
│ ├── 1_EDA.IPYNB
│ ├── 2_3_VISUALIZE_ADVANCED_PREPROCESSING.IPYNB
│ ├── 4_FEATURE_EXTRACTION.IPYNB
│ ├── 5_DATA_GENERATOR.IPYNB
│ ├── 6_TRAIN.IPYNB
│ ├── TEST.IPYNB # Mock training + test on same data
│ ├── 7_TEST.IPYNB
│ ├── 8_INTERMEDIATE_FEATURES.IPYNB
│ ├── 9_EVALUATION_RESULTS.IPYNB
│ └── 10_ABLATION_STUDY.IPYNB
---
## 🔧 Setup

Install dependencies:

```bash
pip install -r requirements.txt
```
Use Python 3.9+ and activate your virtual environment before running notebooks.

