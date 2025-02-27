import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_and_sort_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes performance metrics for visualization.

    Parameters:
    - results_df: DataFrame containing model evaluation metrics.

    Returns:
    - Normalized DataFrame.
    """
    
    metrics = ['log_loss', 'pr_auc', 'roc_auc', 'f1_score']
    
    # Normalize the metrics
    scaler = MinMaxScaler()
    results_df[metrics] = scaler.fit_transform(results_df[metrics])

    return results_df.sort_values(by="roc_auc", ascending=False)
