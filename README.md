# machine-learning-utils

This repository contains reusable machine learning utilities for training, tuning, evaluating, and explaining models. The structure is organized to separate **Supervised ML**, **Unsupervised ML**, **LLMs**, and **Reinforcement Learning**, with XGBoost-specific tools in a dedicated subdirectory.

---

## Repository Structure

### **Supervised Learning (`supervised_ml/`)**
Contains general utilities for classification and regression. XGBoost-specific tools are in `supervised_ml/xgboost/`.

#### **General Supervised ML**
- `performance_metrics.py` – Computes accuracy, F1-score, AUC, PR-AUC, MCC, log loss, etc.
- `visualization.py` – Plots ROC, PR curves, confusion matrices.
- `actual_positives_analysis.py` – Analyzes cases where predicted risk actually happened.
- `model_comparison.py` – Trains and evaluates multiple ML models.
- `model_results.py` – Normalizes and sorts evaluation results.

#### **XGBoost-Specific (`supervised_ml/xgboost/`)**
- `xgboost_tuning.py` – Hyperparameter tuning for XGBoost.
- `xgb_objective_functions.py` – Custom loss functions for tuning.
- `xgb_training.py` – Training script for XGBoost models.
- `hyperparam_logging.py` – Saves and loads best hyperparameters.
- `shap_analysis.py` – SHAP-based feature explainability.
- `lime_analysis.py` – LIME-based interpretability.
- `cross_validation.py` – Stratified K-Fold cross-validation for XGBoost.

### **Unsupervised Learning (`unsupervised_ml/`)**
(Currently a placeholder for future clustering, dimensionality reduction, and anomaly detection.)
- `kmeans.py` – K-Means clustering (future).
- `pca.py` – Principal Component Analysis (future).
- `autoencoders.py` – Autoencoders for anomaly detection (future).

### **Large Language Models (`llms/`)**
(Placeholder for future LLM fine-tuning and evaluation.)
- `fine_tuning.py` – Transformer-based fine-tuning (future).
- `prompt_engineering.py` – Optimizing prompts (future).
- `evaluation.py` – BLEU, ROUGE, and Perplexity scores (future).

### **Reinforcement Learning (`reinforcement_learning/`)**
(Placeholder for future reinforcement learning tools.)
- `q_learning.py` – Q-Learning (future).
- `ppo.py` – Proximal Policy Optimization (future).

### **Utilities (`utils/`)**
- in progress

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/ml-utils.git
cd ml-utils
pip install -r requirements.txt
```bash
