"""
Example: PKBoost Adaptive with Drift Detection
Demonstrates real-time drift monitoring and automatic metamorphosis
"""
import numpy as np
from pkboost import PKBoostAdaptive
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

print("=== PKBoost Adaptive Drift Detection Demo ===\n")

# Generate initial imbalanced dataset
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    weights=[0.98, 0.02],  # 2% minority class
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

# Initialize adaptive model
model = PKBoostAdaptive()

# Initial training
print("Step 1: Initial Training")
model.fit_initial(X_train, y_train, x_val=X_test[:500], y_val=y_test[:500], verbose=True)

# Baseline evaluation
y_pred = model.predict_proba(X_test)
baseline_pr_auc = average_precision_score(y_test, y_pred)
baseline_roc_auc = roc_auc_score(y_test, y_pred)

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
batch_size = 500

for batch_idx in range(n_batches):
    # Generate new batch with increasing drift
    drift_factor = batch_idx * 0.15  # Gradual drift
    
    X_batch, y_batch = make_classification(
        n_samples=batch_size,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        weights=[0.98, 0.02],
        random_state=42 + batch_idx,
        shift=drift_factor,  # Introduce covariate shift
        flip_y=0.05 * batch_idx  # Introduce label noise
    )
    
    X_batch = np.ascontiguousarray(X_batch, dtype=np.float64)
    y_batch = np.ascontiguousarray(y_batch, dtype=np.float64)
    
    # Observe batch (triggers drift detection)
    print(f"\n--- Batch {batch_idx + 1}/{n_batches} ---")
    model.observe_batch(X_batch, y_batch, verbose=True)
    
    # Check current state
    state = model.get_state()
    vuln_score = model.get_vulnerability_score()
    meta_count = model.get_metamorphosis_count()
    
    # Evaluate on test set
    y_pred = model.predict_proba(X_test)
    current_pr_auc = average_precision_score(y_test, y_pred)
    current_roc_auc = roc_auc_score(y_test, y_pred)
    
    print(f"State: {state}")
    print(f"Vulnerability: {vuln_score:.4f}")
    print(f"Metamorphoses: {meta_count}")
    print(f"Test PR-AUC: {current_pr_auc:.4f} (baseline: {baseline_pr_auc:.4f})")
    print(f"Test ROC-AUC: {current_roc_auc:.4f} (baseline: {baseline_roc_auc:.4f})")
    
    degradation = (baseline_pr_auc - current_pr_auc) / baseline_pr_auc * 100
    print(f"Performance degradation: {degradation:.1f}%")

print("\n\n=== Final Summary ===")
print(f"Total metamorphoses triggered: {model.get_metamorphosis_count()}")
print(f"Final state: {model.get_state()}")
print(f"Final vulnerability score: {model.get_vulnerability_score():.4f}")

# Final evaluation
y_pred_final = model.predict_proba(X_test)
final_pr_auc = average_precision_score(y_test, y_pred_final)
final_roc_auc = roc_auc_score(y_test, y_pred_final)

print(f"\nPerformance comparison:")
print(f"  Baseline PR-AUC: {baseline_pr_auc:.4f}")
print(f"  Final PR-AUC:    {final_pr_auc:.4f}")
print(f"  Change:          {(final_pr_auc - baseline_pr_auc):.4f} ({((final_pr_auc - baseline_pr_auc) / baseline_pr_auc * 100):.1f}%)")