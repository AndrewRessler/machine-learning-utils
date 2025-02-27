import numpy as np
from hyperopt import fmin, tpe, Trials, hp
from functools import partial
from xgb_objective_functions import f1_objective
from xgboost import XGBClassifier

# Define hyperparameter search space
space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(0.3)),
    'max_depth': hp.quniform('max_depth', 3, 20, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 20, 1),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'gamma': hp.uniform('gamma', 0, 20),
    'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-5), np.log(100)),
    'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-5), np.log(100)),
    'scale_pos_weight': hp.uniform('scale_pos_weight', 1, 100), 
    'max_delta_step': hp.quniform('max_delta_step', 0, 10, 1),  
    'max_leaves': hp.quniform('max_leaves', 0, 5000, 1),  
}

# Wrap objective function for optimization
def optimize_hyperparams(X_train, y_train, X_val, y_val, max_evals=100):
    trials = Trials()
    partial_objective = partial(f1_objective, X_train=X_train, y_train=y_train, X_validation=X_val, y_validation=y_val)
    
    best_hp = fmin(
        fn=partial_objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials
    )

    # Ensure integer-based hyperparameters are properly cast
    best_hp['max_depth'] = int(best_hp['max_depth'])
    best_hp['min_child_weight'] = int(best_hp['min_child_weight'])
    best_hp['max_delta_step'] = int(best_hp['max_delta_step'])
    best_hp['max_leaves'] = int(best_hp['max_leaves'])
    
    return best_hp
