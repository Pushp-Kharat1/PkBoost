"""
Example: PKBoost Adaptive with Drift Detection on Credit Card Fraud
Demonstrates real-time drift monitoring and automatic metamorphosis
"""
import numpy as np
import pandas as pd
from pkboost import PKBoostAdaptive
from sklearn.metrics import roc_auc_score, average_precision_score

print("=== PKBoost Adaptive Drift Detection Demo ===")
print("Dataset: Credit Card Fraud (0.2% fraud rate)\n")

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

print(f"Train: {X_train.shape[0]} samples, {(y_train.sum() / len(y_train) * 100):.2f}% fraud")
print(f"Val: {X_val.shape[0]} samples, {(y_val.sum() / len(y_val) * 100):.2f}% fraud")
print(f"Test: {X_test.shape[0]} samples, {(y_test.sum() / len(y_test) * 100):.2f}% fraud\n")

# Convert to contiguous arrays
y_train = np.ascontiguousarray(y_train, dtype=np.float64)
y_val = np.ascontiguousarray(y_val, dtype=np.float64)
y_test = np.ascontiguousarray(y_test, dtype=np.float64)
X_train = np.ascontiguousarray(X_train, dtype=np.float64)
X_val = np.ascontiguousarray(X_val, dtype=np.float64)
X_test = np.ascontiguousarray(X_test, dtype=np.float64)

# Initialize adaptive model
model = PKBoostAdaptive()

# Initial training
print("Step 1: Initial Training")
model.fit_initial(X_train, y_train, x_val=X_val[:5000], y_val=y_val[:5000], verbose=True)

# Baseline evaluation
y_pred = model.predict_proba(X_val)
baseline_pr_auc = average_precision_score(y_val, y_pred)
baseline_roc_auc = roc_auc_score(y_val, y_pred)

print(f"\n=== Baseline Performance ===")
print(f"PR-AUC: {baseline_pr_auc:.4f}")
print(f"ROC-AUC: {baseline_roc_auc:.4f}")
print(f"State: {model.get_state()}")
print(f"Vulnerability Score: {model.get_vulnerability_score():.4f}")
print(f"Metamorphoses: {model.get_metamorphosis_count()}")

# Simulate streaming data with concept drift
print("\n\nStep 2: Simulating Streaming Data with Drift")
print("=" * 60)

np.random.seed(123)
n_batches = 10
batch_size = 1000

# Split test set into batches and add noise to simulate drift
test_indices = np.arange(len(X_test))
np.random.shuffle(test_indices)

for batch_idx in range(n_batches):
    # Get batch from test set
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(X_test))
    batch_indices = test_indices[start_idx:end_idx]
    
    X_batch = X_test[batch_indices].copy()
    y_batch = y_test[batch_indices].copy()
    
    # Add increasing noise to simulate drift
    noise_level = batch_idx * 0.1
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, X_batch.shape)
        X_batch = X_batch + noise
    
    X_batch = np.ascontiguousarray(X_batch, dtype=np.float64)
    y_batch = np.ascontiguousarray(y_batch, dtype=np.float64)
    
    # Observe batch (triggers drift detection)
    print(f"\n--- Batch {batch_idx + 1}/{n_batches} (noise level: {noise_level:.1f}) ---")
    model.observe_batch(X_batch, y_batch, verbose=True)
    
    # Check current state
    state = model.get_state()
    vuln_score = model.get_vulnerability_score()
    meta_count = model.get_metamorphosis_count()
    
    # Evaluate on validation set
    y_pred = model.predict_proba(X_val)
    current_pr_auc = average_precision_score(y_val, y_pred)
    current_roc_auc = roc_auc_score(y_val, y_pred)
    
    print(f"State: {state}")
    print(f"Vulnerability: {vuln_score:.4f}")
    print(f"Metamorphoses: {meta_count}")
    print(f"Val PR-AUC: {current_pr_auc:.4f} (baseline: {baseline_pr_auc:.4f})")
    print(f"Val ROC-AUC: {current_roc_auc:.4f} (baseline: {baseline_roc_auc:.4f})")
    
    degradation = (baseline_pr_auc - current_pr_auc) / baseline_pr_auc * 100
    print(f"Performance degradation: {degradation:.1f}%")

print("\n\n=== Final Summary ===")
print(f"Total metamorphoses triggered: {model.get_metamorphosis_count()}")
print(f"Final state: {model.get_state()}")
print(f"Final vulnerability score: {model.get_vulnerability_score():.4f}")

# Final evaluation on validation set
y_pred_final = model.predict_proba(X_val)
final_pr_auc = average_precision_score(y_val, y_pred_final)
final_roc_auc = roc_auc_score(y_val, y_pred_final)

print(f"\nPerformance comparison:")
print(f"  Baseline PR-AUC: {baseline_pr_auc:.4f}")
print(f"  Final PR-AUC:    {final_pr_auc:.4f}")
print(f"  Change:          {(final_pr_auc - baseline_pr_auc):.4f} ({((final_pr_auc - baseline_pr_auc) / baseline_pr_auc * 100):.1f}%)")
