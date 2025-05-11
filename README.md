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

```
BRAINTUMOR/
├── DATASET/
│   ├── FILES/                        # .mat files (MRI slices)
│   └── cvind.mat                     # Fold split file
│
├── DSVIT/
│   ├── dsvit/
│   │   ├── model.py                 # Main + ablation variants
│   │   ├── dataset.py               # Custom PyTorch dataset
│   │   ├── preprocessor.py          # CLAHE, Skull Strip, etc.
│   │   ├── hooks.py                 # For attention visualizations
│   │   └── utils.py                 # Confusion matrix, ROC, etc.
│   ├── model-weight/                # Saved model weights
│   └── notebooks/
│       ├── 1_EDA.IPYNB
│       ├── 2_3_VISUALIZE_ADVANCED_PREPROCESSING.IPYNB
│       ├── 4_FEATURE_EXTRACTION.IPYNB
│       ├── 5_DATA_GENERATOR.IPYNB
│       ├── 6_TRAIN.IPYNB
│       ├── TEST.IPYNB               # Mock training + test on same data
│       ├── 7_TEST.IPYNB
│       ├── 8_INTERMEDIATE_FEATURES.IPYNB
│       ├── 9_EVALUATION_RESULTS.IPYNB
│       └── 10_ABLATION_STUDY.IPYNB
```

---

## 🔧 Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Use Python 3.9+ and activate your virtual environment before running notebooks.

---

## 📊 Training & Inference

To train DSViT:

```python
# In 6_TRAIN.IPYNB
model = DSViTDetector()
...
```

Test on mock data:

```python
# In TEST.IPYNB
bbox_pred, class_logits = model(image)
```

---

## 📈 Evaluation Metrics

Use the following notebooks:

- `9_EVALUATION_RESULTS.IPYNB`: Confusion matrix, F1, ROC-AUC
- `8_INTERMEDIATE_FEATURES.IPYNB`: Visualize token attention maps
- `10_ABLATION_STUDY.IPYNB`: Compare ablation variants

---

## 🖼️ Sample Visualization

- Raw + Annotated MRI slices
- Predicted bounding boxes + tumor types
- Attention maps from intermediate layers
