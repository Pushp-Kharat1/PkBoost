"""
Example: PKBoost on Credit Card Fraud Dataset
Standard training without drift detection
"""
import numpy as np
import pandas as pd
from pkboost import PKBoostClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

print("=== PKBoost Credit Card Fraud Detection ===\n")

# Load credit card fraud dataset
print("Loading data...")
train_df = pd.read_csv("data/creditcard_train.csv")
val_df = pd.read_csv("data/creditcard_val.csv")
test_df = pd.read_csv("data/creditcard_test.csv")

# Separate features and labels
X_train = train_df.drop('Class', axis=1).values
y_train = train_df['Class'].values
X_val = val_df.drop('Class', axis=1).values
y_val = val_df['Class'].values
X_test = test_df.drop('Class', axis=1).values
y_test = test_df['Class'].values

print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features, {(y_train.sum() / len(y_train) * 100):.2f}% fraud")
print(f"Val: {X_val.shape[0]} samples, {(y_val.sum() / len(y_val) * 100):.2f}% fraud")
print(f"Test: {X_test.shape[0]} samples, {(y_test.sum() / len(y_test) * 100):.2f}% fraud\n")

# Convert to contiguous arrays
X_train = np.ascontiguousarray(X_train, dtype=np.float64)
y_train = np.ascontiguousarray(y_train, dtype=np.float64)
X_val = np.ascontiguousarray(X_val, dtype=np.float64)
y_val = np.ascontiguousarray(y_val, dtype=np.float64)
X_test = np.ascontiguousarray(X_test, dtype=np.float64)
y_test = np.ascontiguousarray(y_test, dtype=np.float64)

# Train with auto-tuning
print("Training PKBoost with auto-tuning...")
model = PKBoostClassifier.auto()
model.fit(X_train, y_train, x_val=X_val, y_val=y_val, verbose=True)

# Predict on test set
print("\n=== Test Set Evaluation ===")
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test, threshold=0.5)

# Metrics
pr_auc = average_precision_score(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Confusion matrix
tp = ((y_pred == 1) & (y_test == 1)).sum()
fp = ((y_pred == 1) & (y_test == 0)).sum()
tn = ((y_pred == 0) & (y_test == 0)).sum()
fn = ((y_pred == 0) & (y_test == 1)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nPR-AUC: {pr_auc:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"\nConfusion Matrix:")
print(f"  TP: {tp:5d}  FP: {fp:5d}")
print(f"  FN: {fn:5d}  TN: {tn:5d}")
print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Feature importance
importance = model.get_feature_importance()
top_features = importance.argsort()[-10:][::-1]
print(f"\nTop 10 Important Features: {top_features}")
print(f"Number of trees: {model.get_n_trees()}")
