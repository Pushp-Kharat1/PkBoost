import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import subprocess
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data(path):
    df = pd.read_csv(path)
    y = df.pop('Class')
    return df.values, y.values

def evaluate_model(model, X, y, model_type='lgb'):
    if len(np.unique(y)) < 2:
        return 0.0, 0.5, 0.0
    
    if model_type == 'lgb':
        preds = model.predict(X, num_iteration=model.best_iteration)
    elif model_type == 'xgb':
        dmat = xgb.DMatrix(X)
        preds = model.predict(dmat, iteration_range=(0, model.best_iteration + 1))
    
    pr_auc = average_precision_score(y, preds)
    roc_auc = roc_auc_score(y, preds)
    
    # Use optimal threshold based on precision-recall curve instead of 0.5
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y, preds)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    y_pred = (preds >= optimal_threshold).astype(int)
    f1 = f1_score(y, y_pred, zero_division=0)
    return pr_auc, roc_auc, f1

print("\n=== DRIFT COMPARISON: LightGBM vs XGBoost ===\n")
print("Test 1: Single Large Evaluation (All samples at once)\n")

# Load data
data_path = Path('data')
print("Loading data...")
X_train, y_train = load_data(data_path / 'synthetic_financial_train.csv')
X_val, y_val = load_data(data_path / 'synthetic_financial_val.csv')
X_test, y_test = load_data(data_path / 'synthetic_financial_test.csv')
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}\n")

# Train LightGBM
print("Training LightGBM...")
train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
lgb_model = lgb.train(
    {"objective": "binary", "metric": ["auc"], "verbosity": -1},
    train_set, num_boost_round=2000, valid_sets=[val_set],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
)

# Train XGBoost
print("Training XGBoost...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
xgb_model = xgb.train(
    {"objective": "binary:logistic", "eval_metric": ["auc"], "verbosity": 0},
    dtrain, num_boost_round=2000, evals=[(dval, 'val')],
    early_stopping_rounds=100, verbose_eval=False
)

# Phase 1: Normal data (use test set WITHOUT drift)
print("\n=== PHASE 1: NORMAL DATA ===")
print("Evaluating on test set (no drift)...")

lgb_pr, lgb_roc, lgb_f1 = evaluate_model(lgb_model, X_test, y_test, 'lgb')
xgb_pr, xgb_roc, xgb_f1 = evaluate_model(xgb_model, X_test, y_test, 'xgb')

print(f"Fraud rate: {y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.2f}%)")
print(f"LightGBM - PR-AUC: {lgb_pr:.4f}, ROC-AUC: {lgb_roc:.4f}, F1: {lgb_f1:.4f}")
print(f"XGBoost  - PR-AUC: {xgb_pr:.4f}, ROC-AUC: {xgb_roc:.4f}, F1: {xgb_f1:.4f}")

baseline_lgb, baseline_xgb = lgb_pr, xgb_pr


# Phase 2: Apply drift to SAME test set
print("\n=== PHASE 2: APPLYING DRIFT ===")
drift_features = [5, 10, 15, 20, 25]
X_drift = X_test.copy()
for feat_idx in drift_features:
    if feat_idx < X_drift.shape[1]:
        X_drift[:, feat_idx] = -X_drift[:, feat_idx] + 10.0

print("Evaluating on drifted test set (same data, different distribution)...")
lgb_pr_drift, _, _ = evaluate_model(lgb_model, X_drift, y_test, 'lgb')
xgb_pr_drift, _, _ = evaluate_model(xgb_model, X_drift, y_test, 'xgb')

print(f"LightGBM - PR-AUC: {lgb_pr_drift:.4f}")
print(f"XGBoost  - PR-AUC: {xgb_pr_drift:.4f}")

lgb_drift = [lgb_pr_drift]
xgb_drift = [xgb_pr_drift]

# Analysis
print("\n=== PERFORMANCE ANALYSIS ===\n")

print(f"LightGBM:")
print(f"  Baseline: {baseline_lgb:.4f}")
print(f"  Drift: {lgb_drift[0]:.4f}")
print(f"  Degradation: {((baseline_lgb - lgb_drift[0])/baseline_lgb*100):.1f}%")

print(f"\nXGBoost:")
print(f"  Baseline: {baseline_xgb:.4f}")
print(f"  Drift: {xgb_drift[0]:.4f}")
print(f"  Degradation: {((baseline_xgb - xgb_drift[0])/baseline_xgb*100):.1f}%")


# Bar chart comparison
models = ['LightGBM', 'XGBoost']
baseline_scores = [baseline_lgb, baseline_xgb]
drift_scores = [lgb_drift[0], xgb_drift[0]]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 7))
ax.bar(x - width/2, baseline_scores, width, label='Baseline (Normal)', color=['blue', 'orange'], alpha=0.7)
ax.bar(x + width/2, drift_scores, width, label='After Drift', color=['blue', 'orange'], alpha=0.4)

ax.set_ylabel('PR-AUC', fontsize=12)
ax.set_title('Model Performance: Baseline vs Drift', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('drift_comparison_complete.png', dpi=300)
plt.close()

print("\n=== Plot saved ===")
print("  - drift_comparison_complete.png\n")
