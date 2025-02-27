import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from typing import Tuple, List

def compute_shap_values(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    feature_names: List[str],
    sample_size: int = 100,
    random_state: int = 42
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Computes SHAP values for a given tree-based model and ranks features by importance.

    Parameters:
    - model: Trained XGBoost classifier.
    - X_test: Feature matrix for the test set.
    - feature_names: List of feature names.
    - sample_size: Number of samples to use for SHAP value calculation (default: 100).
    - random_state: Random seed for sampling.

    Returns:
    - shap_values_sampled: SHAP values for the sampled dataset.
    - shap_summary_df: DataFrame containing feature names and their mean absolute SHAP values, sorted in descending order.
    """
    
    # Sample test data to speed up SHAP computation
    if sample_size and sample_size < len(X_test):
        sampled_X_test = X_test.sample(n=sample_size, random_state=random_state)
    else:
        sampled_X_test = X_test

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values_sampled = explainer.shap_values(sampled_X_test)

    # Compute mean absolute SHAP values per feature
    mean_shap_values = np.abs(shap_values_sampled).mean(axis=0)

    # Rank features by importance
    sorted_indices = np.argsort(mean_shap_values)[::-1]
    shap_summary_df = pd.DataFrame({
        "Feature": np.array(feature_names)[sorted_indices],
        "Mean Absolute SHAP Value": mean_shap_values[sorted_indices]
    })

    # Print sorted feature importance
    print("Feature importance based on mean absolute SHAP values:")
    print(shap_summary_df.to_string(index=False))

    return shap_values_sampled, shap_summary_df
