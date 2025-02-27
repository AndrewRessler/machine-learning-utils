from xgboost import XGBClassifier
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

def f1_crossval_stratified_objective(params, X_train, y_train, n_splits=5):
    """
    Objective function for optimizing F1-score using Stratified K-Fold cross-validation.

    Args:
        params (dict): Hyperparameters to be tested.
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        n_splits (int): Number of folds in Stratified K-Fold cross-validation.

    Returns:
        float: Negative mean F1-score across folds (for minimization).
    """
    params = {k: int(v) if isinstance(v, float) else v for k, v in params.items()}
    
    clf = XGBClassifier(**params)
    f1_scorer = make_scorer(f1_score)
    strat_kfold = StratifiedKFold(n_splits=n_splits)
    scores = cross_val_score(clf, X_train, y_train, cv=strat_kfold, scoring=f1_scorer)
    
    return -scores.mean()

def f1_objective(params, X_train, y_train, X_validation, y_validation):
    """
    Objective function for maximizing F1-score on validation data.

    Args:
        params (dict): Hyperparameters to be tested.
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_validation (array-like): Validation features.
        y_validation (array-like): Validation labels.

    Returns:
        float: Negative F1-score (for minimization).
    """
    params = {k: int(v) if isinstance(v, float) else v for k, v in params.items()}
    
    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_validation)
    return -f1_score(y_validation, y_pred, zero_division=0)

def f_beta_objective(params, X_train, y_train, X_validation, y_validation, beta=0.5):
    """
    Objective function for optimizing an F-beta score, which balances precision and recall.

    Args:
        params (dict): Hyperparameters to be tested.
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_validation (array-like): Validation features.
        y_validation (array-like): Validation labels.
        beta (float): Weighting factor for precision vs recall in F-beta score.

    Returns:
        float: Negative F-beta score (for minimization).
    """
    params = {k: int(v) if isinstance(v, float) else v for k, v in params.items()}

    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train)
    
    y_proba = clf.predict_proba(X_validation)[:, 1]
    f_beta_score_value = fbeta_score(y_validation, y_proba >= 0.5, beta=beta, zero_division=0)

    return -f_beta_score_value

def custom_objective(params, X_train, y_train, X_validation, y_validation):
    """
    Custom objective function optimizing a weighted precision-recall score.

    Args:
        params (dict): Hyperparameters to be tested.
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_validation (array-like): Validation features.
        y_validation (array-like): Validation labels.

    Returns:
        float: Negative weighted precision-recall score (for minimization).
    """
    params = {k: int(v) if isinstance(v, float) else v for k, v in params.items()}

    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train)
    
    y_proba = clf.predict_proba(X_validation)[:, 1]
    
    thresholds = [0.9, 0.75, 0.5]
    weights = [3, 3, 2]  # Weight higher thresholds more
    precisions = [precision_score(y_validation, y_proba >= threshold, zero_division=0) for threshold in thresholds]
    weighted_precision = sum(w * p for w, p in zip(weights, precisions)) / sum(weights)
    
    overall_recall = recall_score(y_validation, y_proba >= 0.5)
    
    final_score = 0.6 * weighted_precision + 0.4 * overall_recall
    
    return -final_score

def multi_target_custom_objective(params, X_train, y_train, X_validation, y_validation):
    """
    Custom multi-target objective function incorporating thresholded precision scoring.

    Args:
        params (dict): Hyperparameters to be tested.
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        X_validation (array-like): Validation features.
        y_validation (array-like): Validation labels.

    Returns:
        float: Negative weighted proportional precision score (for minimization).
    """
    params = {k: int(v) if isinstance(v, float) else v for k, v in params.items()}

    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train)
    probabilities = clf.predict_proba(X_validation)[:, 1]
    
    thresholds = [0.9, 0.75, 0.5]
    target_proportions = [0.05, 0.15, 0.30]
    weights = [5, 3, 1]  

    precisions = [precision_score(y_validation, probabilities >= threshold, zero_division=0) for threshold in thresholds]

    proportional_scores = []
    for idx, (threshold, target, weight) in enumerate(zip(thresholds, target_proportions, weights)):
        current_proportion = (probabilities >= threshold).mean()
        proportion_deviation = abs(current_proportion - target)
        proportional_score = weight * (precisions[idx] - 10 * proportion_deviation)  
        proportional_scores.append(proportional_score)

    final_score = sum(proportional_scores) / sum(weights)
    
    return -final_score
