import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay
import pandas as pd

def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    """
    Plots the Receiver Operating Characteristic (ROC) curve.

    Parameters:
    - y_true: True labels
    - y_proba: Predicted probabilities for the positive class.
    """

    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title("ROC Curve")
    plt.show()


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    """
    Plots the Precision-Recall (PR) curve.

    Parameters:
    - y_true: True labels
    - y_proba: Predicted probabilities for the positive class.
    """

    PrecisionRecallDisplay.from_predictions(y_true, y_proba)
    plt.title("Precision-Recall Curve")
    plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plots a confusion matrix as a heatmap.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    """

    cm = confusion_matrix(y_true, y_pred)
    labels = ["Negative", "Positive"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="g", xticklabels=labels, yticklabels=labels)

    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()
