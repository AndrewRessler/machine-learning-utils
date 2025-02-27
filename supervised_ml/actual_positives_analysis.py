import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

def analyze_actual_positives(
    predictions_df: pd.DataFrame,
    target_col: str,
    risk_col: str = "predicted_risk",
    threshold: float = 0.5,
    bins: int = 10,
    color: str = 'lavender'
) -> Tuple[float, pd.DataFrame]:
    """
    Analyzes cases where the predicted risk actually occurred, visualizes the distribution, 
    and calculates the percentage of positive cases correctly predicted as high risk.

    Parameters:
    - predictions_df: DataFrame containing model predictions, actual labels, and risk scores.
    - target_col: Column name representing the actual observed outcome (1 = event occurred, 0 = did not occur).
    - risk_col: Column name containing predicted risk scores.
    - threshold: Decision threshold for classifying high-risk predictions.
    - bins: Number of bins for the histogram.
    - color: Color of histogram bars.

    Returns:
    - correctly_predicted_fraction: Fraction of actual positive cases that were classified above the risk threshold.
    - actual_positives_df: Filtered DataFrame of actual positive cases with their predicted risk scores.
    """
    
    # Filter dataset to only include cases where the event actually occurred
    actual_positives_df = predictions_df[predictions_df[target_col].notna() & (predictions_df[target_col] == 1)]
    
    if actual_positives_df.empty:
        print("Warning: No actual positive cases found in the dataset.")
        return 0.0, actual_positives_df

    # Plot histogram of predicted risk scores for actual positives
    counts, bin_edges, patches = plt.hist(actual_positives_df[risk_col], bins=bins, color=color, edgecolor='black')
    total_count = actual_positives_df[risk_col].size

    # Calculate and annotate percentages for each bin
    percentages = (counts / total_count) * 100
    for percentage, bin_edge, patch in zip(percentages, bin_edges, patches):
        bin_center = patch.get_x() + patch.get_width() / 2.0
        height = patch.get_height()
        plt.text(bin_center, height, f'{percentage:.1f}%', ha='center', va='bottom')

    plt.xlabel("Predicted Risk Score")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Predicted Risk Scores for Actual Positive Cases ({target_col})")
    plt.show()

    # Calculate fraction of actual positives classified above the risk threshold
    correctly_predicted_fraction = (actual_positives_df[risk_col] > threshold).mean()
    print(f"Fraction of actual positive cases predicted as high risk (> {threshold}): {correctly_predicted_fraction:.2%}")

    return correctly_predicted_fraction, actual_positives_df
