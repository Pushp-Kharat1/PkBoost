"""
Example usage of PKBoost Python bindings
"""
import numpy as np
from pkboost import PKBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# Generate imbalanced dataset
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    weights=[0.50, 0.50],
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to contiguous arrays
y_train = np.ascontiguousarray(y_train, dtype=np.float64)
y_test = np.ascontiguousarray(y_test, dtype=np.float64)
X_train = np.ascontiguousarray(X_train, dtype=np.float64)
X_test = np.ascontiguousarray(X_test, dtype=np.float64)

# Auto-tuned model
print("Training PKBoost with auto-tuning...")
model = PKBoostClassifier.auto()
model.fit(X_train, y_train, x_val=X_test[:500], y_val=y_test[:500], verbose=True)

# Predict
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test, threshold=0.5)
importance = model.get_feature_importance()

# Evaluate
roc_auc = roc_auc_score(y_test, y_pred_proba)
pr_auc = average_precision_score(y_test, y_pred_proba)

print(f"\nResults:")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"Model fitted: {model.is_fitted}")
print(f"Top 5 important features: {importance.argsort()[-5:][::-1]}")
print(f"Predictions: {y_pred[:10]}")
