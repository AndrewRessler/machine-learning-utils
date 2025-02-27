import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from typing import Dict, Any, Tuple

def cross_validate_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict[str, Any],
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[float, Dict[str, float]]:
    """
    Performs Stratified K-Fold cross-validation on an XGBoost model.

    Parameters:
    - X: Feature matrix.
    - y: Target vector.
    - params: Dictionary of hyperparameters for XGBoost.
    - n_splits: Number of cross-validation folds (default: 5).
    - random_state: Random seed for reproducibility.

    Returns:
    - mean_score: Average AUC score across folds.
    - metric_results: Dictionary with accuracy, F1, AUC, and PR AUC scores.
    """
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    scores = {"accuracy": [], "F1": [], "ROC_AUC": [], "PR_AUC": []}

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train XGBoost model
        model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train)

        # Get predictions & probabilities
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Compute metrics
        scores["accuracy"].append(accuracy_score(y_test, y_pred))
        scores["F1"].append(f1_score(y_test, y_pred))
        scores["ROC_AUC"].append(roc_auc_score(y_test, y_proba))
        scores["PR_AUC"].append(average_precision_score(y_test, y_proba))

    # Compute mean scores across folds
    mean_scores = {metric: np.mean(values) for metric, values in scores.items()}

    print("Cross-validation results:")
    for metric, score in mean_scores.items():
        print(f"{metric}: {score:.4f}")

    return mean_scores["ROC_AUC"], mean_scores
