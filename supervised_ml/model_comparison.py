import time
import numpy as np
import pandas as pd
import gc
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    log_loss, f1_score, roc_auc_score, precision_recall_curve, auc
)
from collections import defaultdict
from typing import Dict, Any, List
from catboost import Pool

def train_and_evaluate_models(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    early_stopping_rounds: int = 25
) -> pd.DataFrame:
    """
    Trains and evaluates multiple models, handling early stopping and errors.

    Parameters:
    - models: Dictionary of model names and instances.
    - X_train, y_train: Training dataset.
    - X_val, y_val: Validation dataset for early stopping.
    - X_test, y_test: Test dataset.
    - early_stopping_rounds: Number of rounds for early stopping.

    Returns:
    - DataFrame containing model evaluation metrics.
    """

    results = []
    for name, model in models.items():
        print(f"Training {name}...")

        y_pred, y_prob = None, None
        start_time = time.time()
        
        try:
            if name in ["LightGBM", "XGBoost", "CatBoost"]:
                # Specialized training for boosting models
                if name == "LightGBM":
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False
                    )
                elif name == "CatBoost":
                    train_pool = Pool(X_train, y_train)
                    eval_pool = Pool(X_val, y_val)
                    model.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=early_stopping_rounds, use_best_model=True, verbose=False)
                elif name == "XGBoost":
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=False
                    )
            else:
                model.fit(X_train, y_train)

            elapsed_time = time.time() - start_time
            print(f"Training {name} completed in {elapsed_time:.2f} seconds.")

            # Get predictions
            y_pred = model.predict(X_test)
            y_prob = (
                model.decision_function(X_test) if hasattr(model, "decision_function")
                else model.predict_proba(X_test)[:, 1]
            )

            # Calculate metrics
            ll = log_loss(y_test, y_prob)
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall, precision)
            roc_auc = roc_auc_score(y_test, y_prob)
            f1 = f1_score(y_test, y_pred)

            results.append({
                'model': name,
                'log_loss': ll,
                'pr_auc': pr_auc,
                'roc_auc': roc_auc,
                'f1_score': f1,
                'training_time': elapsed_time
            })
        
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"Error with {name}: {e}")
            results.append({
                'model': name,
                'log_loss': 1,
                'pr_auc': 0,
                'roc_auc': 0,
                'f1_score': 0,
                'training_time': elapsed_time
            })
        
        # Memory cleanup
        del y_pred, y_prob
        gc.collect()

    return pd.DataFrame(results)
