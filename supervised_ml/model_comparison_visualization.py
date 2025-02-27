import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_model_comparison(results_df: pd.DataFrame, sort_by: str = "roc_auc") -> None:
    """
    Plots bar charts comparing models based on evaluation metrics.

    Parameters:
    - results_df: DataFrame containing model evaluation metrics.
    - sort_by: Metric to sort models by before plotting.
    """

    metrics = ['log_loss', 'pr_auc', 'roc_auc', 'f1_score']
    results_df = results_df.sort_values(by=sort_by, ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        sns.barplot(x='model', y=metric, data=results_df, ax=axes[idx])
        axes[idx].set_title(f'Model Comparison for {metric}')
        axes[idx].set_xlabel('Model')
        axes[idx].set_ylabel(metric)
        axes[idx].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

def plot_performance_heatmap(results_df: pd.DataFrame) -> None:
    """
    Plots a heatmap of model performance metrics.

    Parameters:
    - results_df: Normalized DataFrame of model evaluation metrics.
    """
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(results_df.set_index('model')[['log_loss', 'pr_auc', 'roc_auc', 'f1_score']], annot=True, cmap='viridis')
    plt.title('Model Performance Heatmap (Normalized)')
    plt.show()
