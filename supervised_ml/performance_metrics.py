import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, confusion_matrix, log_loss
)
from typing import Dict, Tuple

def evaluate_classification_model(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_proba: np.ndarray
) -> Dict[str, float]:
    """
    Computes classification performance metrics.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - y_proba: Predicted probabilities (for AUC metrics)

    Returns:
    - Dictionary containing accuracy, F1-score, ROC AUC, PR AUC, precision, recall, and log loss.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_score": f1_score(y_true, y_pred),
        "ROC_AUC": roc_auc_score(y_true, y_proba),
        "PR_AUC": average_precision_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba)
    }

    print("\nClassification Performance:")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")

    return metrics


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Computes and returns a confusion matrix as a pandas DataFrame.

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels

    Returns:
    - Pandas DataFrame representing the confusion matrix.
    """

    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm, columns=["Predicted Negative", "Predicted Positive"], 
                           index=["Actual Negative", "Actual Positive"])
