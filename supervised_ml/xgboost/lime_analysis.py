import numpy as np
import pandas as pd
import lime.lime_tabular
from collections import defaultdict
from typing import Tuple, Dict, List, Any
import xgboost as xgb

def create_lime_explainer(
    X_train: pd.DataFrame,
    class_names: List[str] = ["Class 0", "Class 1"]
) -> lime.lime_tabular.LimeTabularExplainer:
    """
    Creates a LIME explainer for an XGBoost model.

    Parameters:
    - X_train: Training feature matrix.
    - class_names: Names for the predicted classes.

    Returns:
    - explainer: Configured LIME tabular explainer.
    """
    
    # Fill missing values with column means
    X_train_filled = X_train.fillna(X_train.mean())

    # Identify categorical features (objects or low-unique-value numerical features)
    categorical_features = [
        i for i, col in enumerate(X_train.columns)
        if X_train[col].dtype == "object" or len(X_train[col].unique()) < 10
    ]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_filled.to_numpy(),
        feature_names=X_train.columns.tolist(),
        class_names=class_names,
        categorical_features=categorical_features,
        mode="classification",
        discretize_continuous=True
    )

    return explainer


def explain_instance_lime(
    explainer: lime.lime_tabular.LimeTabularExplainer,
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    instance_index: int
) -> Any:
    """
    Generates a LIME explanation for a specific instance.

    Parameters:
    - explainer: Pre-configured LIME explainer.
    - model: Trained XGBoost classifier.
    - X_test: Test feature matrix.
    - instance_index: Index of the instance to explain.

    Returns:
    - LIME explanation object.
    """
    
    try:
        instance = X_test.loc[instance_index].to_numpy()
        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict_proba
        )
        return exp
    except Exception as e:
        print(f"Failed to generate explanation for instance {instance_index}: {e}")
        return None


def batch_lime_analysis(
    explainer: lime.lime_tabular.LimeTabularExplainer,
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    sample_size: int = 1000,
    random_seed: int = 40
) -> Dict[str, Dict[str, float]]:
    """
    Runs LIME across a sample of data points to summarize feature importance.

    Parameters:
    - explainer: Pre-configured LIME explainer.
    - model: Trained XGBoost classifier.
    - X_test: Test feature matrix.
    - sample_size: Number of instances to analyze (default: 1000).
    - random_seed: Random seed for reproducibility.

    Returns:
    - feature_influence_summary: Dictionary with summarized LIME feature impact.
    """
    
    np.random.seed(random_seed)
    
    # Ensure we don't sample more than available instances
    sample_size = min(sample_size, len(X_test))
    sample_indices = np.random.choice(X_test.index, sample_size, replace=False)

    valid_explanations = []
    for idx in sample_indices:
        try:
            instance = X_test.loc[idx].to_numpy()
            exp = explainer.explain_instance(instance, model.predict_proba)
            valid_explanations.append((idx, exp))
        except Exception as e:
            print(f"Error processing instance {idx}: {e}")

    # Summarizing feature influence across sampled instances
    feature_influence_summary = defaultdict(lambda: defaultdict(float))
    
    for idx, exp in valid_explanations:
        for feature, weight in exp.as_list():
            influence_type = "positive" if weight > 0 else "negative"
            feature_influence_summary[feature][influence_type] += abs(weight)

    # Compute overall influence ranking
    overall_influence_ranking = sorted(
        [(feature, sum(weights.values())) for feature, weights in feature_influence_summary.items()],
        key=lambda x: x[1], reverse=True
    )

    print("Top influential feature ranges across sampled instances:")
    for feature, influence in overall_influence_ranking[:40]:  # Display top 40 influential features
        print(f"{feature}: {influence:.4f}")

    return feature_influence_summary
