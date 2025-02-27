import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from typing import Tuple, Dict

def analyze_model_predictions(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_df: pd.DataFrame,
    user_id_col: str,
    target_col: str,
    threshold: float = 0.5,
    bins: int = 10,
    color: str = 'lavender'
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Analyze model predictions by computing classification metrics and plotting a histogram of predicted scores.

    Parameters:
    - model: Trained XGBoost classifier.
    - X_test: Feature matrix for the test set.
    - y_test: True labels for the test set.
    - test_df: Original test dataset containing user identifiers and target values.
    - user_id_col: Column name identifying users (or samples).
    - target_col: Column name for the actual target variable.
    - threshold: Decision threshold for classification.
    - bins: Number of bins for histogram.
    - color: Color of histogram bars.

    Returns:
    - predictions_df: DataFrame with user_id, true label, and predicted probabilities.
    - confusion_counts: Dictionary containing TP, FP, TN, FN counts.
    """
    
    # Get classification outcomes
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Create a DataFrame for comparison
    predictions_df = pd.DataFrame({
        user_id_col: test_df[user_id_col],
        target_col: test_df[target_col],
        'predicted_risk': y_pred_proba
    })
    
    # Compute confusion matrix values
    y_pred = (y_pred_proba >= threshold).astype(int)
    tp = ((y_pred == 1) & (y_test == 1)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    tn = ((y_pred == 0) & (y_test == 0)).sum()
    
    confusion_counts = {"true_positive": tp, "false_negative": fn, "false_positive": fp, "true_negative": tn}

    # Plot histogram of predicted probabilities
    counts, bin_edges, patches = plt.hist(predictions_df['predicted_risk'], bins=bins, color=color, edgecolor='black')
    total_count = predictions_df['predicted_risk'].size

    # Calculate and annotate percentages for each bin
    percentages = (counts / total_count) * 100
    for percentage, bin_edge, patch in zip(percentages, bin_edges, patches):
        bin_center = patch.get_x() + patch.get_width() / 2.0
        height = patch.get_height()
        plt.text(bin_center, height, f'{percentage:.1f}%', ha='center', va='bottom')

    plt.xlabel('Predicted Risk Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Model-Predicted Risk Scores')
    plt.show()

    return predictions_df, confusion_counts
