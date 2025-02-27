from xgboost import XGBClassifier
from xgb_objective_functions import optimize_hyperparams

def train_xgboost(X_train, y_train, X_val, y_val, best_hp=None):
    """ Train XGBoost model with best hyperparameters found. """
    
    if best_hp is None:
        best_hp = optimize_hyperparams(X_train, y_train, X_val, y_val)

    xgb_clf = XGBClassifier(**best_hp, n_estimators=10_000, early_stopping_rounds=25)
    
    xgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True
    )
    
    return xgb_clf
