
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('temp/train.csv')
val = pd.read_csv('temp/val.csv')
test = pd.read_csv('temp/test.csv')

X_train = train.drop('Class', axis=1)
y_train = train['Class']
X_val = val.drop('Class', axis=1)
y_val = val['Class']
X_test = test.drop('Class', axis=1)
y_test = test['Class']

print("┌─────────────────────────────────────────────────────────────┐")
print("│ XGBoost (Default Parameters)                                │")
print("└─────────────────────────────────────────────────────────────┘")

start = time.time()
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_train_time = time.time() - start

start = time.time()
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_pred_time = time.time() - start

xgb_pr_auc = average_precision_score(y_test, xgb_probs)
xgb_roc_auc = roc_auc_score(y_test, xgb_probs)

print(f"  Training time:   {xgb_train_time:.2f}s")
print(f"  Prediction time: {xgb_pred_time:.4f}s")
print(f"  PR-AUC:          {xgb_pr_auc:.4f}")
print(f"  ROC-AUC:         {xgb_roc_auc:.4f}\n")

print("┌─────────────────────────────────────────────────────────────┐")
print("│ LightGBM (Default Parameters)                               │")
print("└─────────────────────────────────────────────────────────────┘")

start = time.time()
lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
lgb_train_time = time.time() - start

start = time.time()
lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
lgb_pred_time = time.time() - start

lgb_pr_auc = average_precision_score(y_test, lgb_probs)
lgb_roc_auc = roc_auc_score(y_test, lgb_probs)

print(f"  Training time:   {lgb_train_time:.2f}s")
print(f"  Prediction time: {lgb_pred_time:.4f}s")
print(f"  PR-AUC:          {lgb_pr_auc:.4f}")
print(f"  ROC-AUC:         {lgb_roc_auc:.4f}\n")

# Save results
with open('temp/baseline_results.txt', 'w') as f:
    f.write(f"XGBoost,{xgb_train_time:.2f},{xgb_pred_time:.4f},{xgb_pr_auc:.4f},{xgb_roc_auc:.4f}\n")
    f.write(f"LightGBM,{lgb_train_time:.2f},{lgb_pred_time:.4f},{lgb_pr_auc:.4f},{lgb_roc_auc:.4f}\n")
