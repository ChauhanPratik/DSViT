import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, roc_curve, auc
)
import numpy as np
import torch

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def print_classification_report(y_true, y_pred, class_names):
    """
    Prints classification report for the classes present in the test data.
    
    Automatically matches class_names to labels found in y_true.
    """
    present_labels = sorted(set(y_true))
    present_names = [class_names[i] for i in present_labels]

    report = classification_report(
        y_true, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    )

    print("Classification Report:\n")
    print(report)

def plot_roc(y_true, y_probs, n_classes, class_names):
    plt.figure(figsize=(7, 5))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve((np.array(y_true) == i).astype(int), y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
