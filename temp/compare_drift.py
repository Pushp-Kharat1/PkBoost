
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import warnings
warnings.filterwarnings('ignore')

# Load baseline results
baseline = {}
with open('temp/baseline_results.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        baseline[parts[0]] = {'pr_auc': float(parts[3])}

# Load models and drifted test data
train = pd.read_csv('temp/train.csv')
val = pd.read_csv('temp/val.csv')
test_drift = pd.read_csv('temp/test_drift.csv')

X_train = train.drop('Class', axis=1)
y_train = train['Class']
X_val = val.drop('Class', axis=1)
y_val = val['Class']
X_test_drift = test_drift.drop('Class', axis=1)
y_test_drift = test_drift['Class']

# Retrain models (they were already trained)
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

print("┌─────────────────────────────────────────────────────────────┐")
print("│ XGBoost (Under Drift)                                       │")
print("└─────────────────────────────────────────────────────────────┘")

start = time.time()
xgb_drift_probs = xgb_model.predict_proba(X_test_drift)[:, 1]
xgb_drift_pred_time = time.time() - start

xgb_drift_pr_auc = average_precision_score(y_test_drift, xgb_drift_probs)
xgb_drift_roc_auc = roc_auc_score(y_test_drift, xgb_drift_probs)
xgb_degradation = abs((baseline['XGBoost']['pr_auc'] - xgb_drift_pr_auc) / baseline['XGBoost']['pr_auc'] * 100)

print(f"  Prediction time: {xgb_drift_pred_time:.4f}s")
print(f"  PR-AUC:          {xgb_drift_pr_auc:.4f}")
print(f"  ROC-AUC:         {xgb_drift_roc_auc:.4f}")
print(f"  Degradation:     {xgb_degradation:.2f}%\n")

print("┌─────────────────────────────────────────────────────────────┐")
print("│ LightGBM (Under Drift)                                      │")
print("└─────────────────────────────────────────────────────────────┘")

start = time.time()
lgb_drift_probs = lgb_model.predict_proba(X_test_drift)[:, 1]
lgb_drift_pred_time = time.time() - start

lgb_drift_pr_auc = average_precision_score(y_test_drift, lgb_drift_probs)
lgb_drift_roc_auc = roc_auc_score(y_test_drift, lgb_drift_probs)
lgb_degradation = abs((baseline['LightGBM']['pr_auc'] - lgb_drift_pr_auc) / baseline['LightGBM']['pr_auc'] * 100)

print(f"  Prediction time: {lgb_drift_pred_time:.4f}s")
print(f"  PR-AUC:          {lgb_drift_pr_auc:.4f}")
print(f"  ROC-AUC:         {lgb_drift_roc_auc:.4f}")
print(f"  Degradation:     {lgb_degradation:.2f}%\n")

# Save drift results
with open('temp/drift_results.txt', 'w') as f:
    f.write(f"XGBoost,{xgb_drift_pred_time:.4f},{xgb_drift_pr_auc:.4f},{xgb_drift_roc_auc:.4f},{xgb_degradation:.2f}\n")
    f.write(f"LightGBM,{lgb_drift_pred_time:.4f},{lgb_drift_pr_auc:.4f},{lgb_drift_roc_auc:.4f},{lgb_degradation:.2f}\n")
