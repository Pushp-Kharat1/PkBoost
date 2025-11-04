import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import warnings
warnings.filterwarnings('ignore')

print("\n=== XGBoost & LightGBM Comparison ===\n")

# Load data
train = pd.read_csv('temp/train.csv')
val = pd.read_csv('temp/val.csv')
test = pd.read_csv('temp/test.csv')
test_drift = pd.read_csv('temp/test_drift.csv')

X_train = train.drop('Class', axis=1)
y_train = train['Class']
X_val = val.drop('Class', axis=1)
y_val = val['Class']
X_test = test.drop('Class', axis=1)
y_test = test['Class']
X_test_drift = test_drift.drop('Class', axis=1)
y_test_drift = test_drift['Class']

# XGBoost Baseline
print("[XGBoost - Baseline]")
start = time.time()
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_train_time = time.time() - start

start = time.time()
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_pred_time = time.time() - start

xgb_pr_auc = average_precision_score(y_test, xgb_probs)
xgb_roc_auc = roc_auc_score(y_test, xgb_probs)

print(f"  Training:   {xgb_train_time:.2f}s")
print(f"  Prediction: {xgb_pred_time:.4f}s")
print(f"  PR-AUC:     {xgb_pr_auc:.4f}")
print(f"  ROC-AUC:    {xgb_roc_auc:.4f}\n")

# XGBoost Under Drift
print("[XGBoost - Under Drift]")
start = time.time()
xgb_drift_probs = xgb_model.predict_proba(X_test_drift)[:, 1]
xgb_drift_pred_time = time.time() - start

xgb_drift_pr_auc = average_precision_score(y_test_drift, xgb_drift_probs)
xgb_drift_roc_auc = roc_auc_score(y_test_drift, xgb_drift_probs)
xgb_degradation = abs((xgb_pr_auc - xgb_drift_pr_auc) / xgb_pr_auc * 100)

print(f"  Prediction: {xgb_drift_pred_time:.4f}s")
print(f"  PR-AUC:     {xgb_drift_pr_auc:.4f}")
print(f"  ROC-AUC:    {xgb_drift_roc_auc:.4f}")
print(f"  Degradation: {xgb_degradation:.2f}%\n")

# LightGBM Baseline
print("[LightGBM - Baseline]")
start = time.time()
lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
lgb_train_time = time.time() - start

start = time.time()
lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
lgb_pred_time = time.time() - start

lgb_pr_auc = average_precision_score(y_test, lgb_probs)
lgb_roc_auc = roc_auc_score(y_test, lgb_probs)

print(f"  Training:   {lgb_train_time:.2f}s")
print(f"  Prediction: {lgb_pred_time:.4f}s")
print(f"  PR-AUC:     {lgb_pr_auc:.4f}")
print(f"  ROC-AUC:    {lgb_roc_auc:.4f}\n")

# LightGBM Under Drift
print("[LightGBM - Under Drift]")
start = time.time()
lgb_drift_probs = lgb_model.predict_proba(X_test_drift)[:, 1]
lgb_drift_pred_time = time.time() - start

lgb_drift_pr_auc = average_precision_score(y_test_drift, lgb_drift_probs)
lgb_drift_roc_auc = roc_auc_score(y_test_drift, lgb_drift_probs)
lgb_degradation = abs((lgb_pr_auc - lgb_drift_pr_auc) / lgb_pr_auc * 100)

print(f"  Prediction: {lgb_drift_pred_time:.4f}s")
print(f"  PR-AUC:     {lgb_drift_pr_auc:.4f}")
print(f"  ROC-AUC:    {lgb_drift_roc_auc:.4f}")
print(f"  Degradation: {lgb_degradation:.2f}%\n")

# Save results
with open('temp/results.csv', 'w') as f:
    f.write("Model,Baseline_PR_AUC,Drift_PR_AUC,Degradation,Train_Time\n")
    f.write(f"XGBoost,{xgb_pr_auc:.4f},{xgb_drift_pr_auc:.4f},{xgb_degradation:.2f},{xgb_train_time:.2f}\n")
    f.write(f"LightGBM,{lgb_pr_auc:.4f},{lgb_drift_pr_auc:.4f},{lgb_degradation:.2f},{lgb_train_time:.2f}\n")

print("=== Results saved to temp/results.csv ===")
