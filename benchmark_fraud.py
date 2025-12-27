"""
PKBoost Credit Card Fraud Detection Benchmark
Tests PKBoost on the Credit Card Fraud dataset.

Dataset files expected:
  - train.csv
  - val.csv  
  - test.csv
"""

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

# ============================================================
# CONFIGURATION - Update these paths as needed
# ============================================================
TRAIN_PATH = "data/creditcard_train.csv"
VAL_PATH = "data/creditcard_val.csv"
TEST_PATH = "data/creditcard_test.csv"
TARGET_COLUMN = "Class"  # Column name for the target variable
# ============================================================


def load_data(train_path, val_path, test_path, target_col):
    """Load pre-split train, validation, and test datasets."""
    print("=" * 60)
    print("Loading Credit Card Fraud Dataset")
    print("=" * 60)
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"✓ Train: {len(train_df):,} samples")
    print(f"✓ Val:   {len(val_df):,} samples")
    print(f"✓ Test:  {len(test_df):,} samples")
    
    # Extract features and target
    X_train = train_df.drop([target_col], axis=1).values
    y_train = train_df[target_col].values
    
    X_val = val_df.drop([target_col], axis=1).values
    y_val = val_df[target_col].values
    
    X_test = test_df.drop([target_col], axis=1).values
    y_test = test_df[target_col].values
    
    print(f"\nFeatures: {X_train.shape[1]}")
    print(f"Train fraud ratio: {y_train.mean()*100:.3f}%")
    print(f"Val fraud ratio:   {y_val.mean()*100:.3f}%")
    print(f"Test fraud ratio:  {y_test.mean()*100:.3f}%")
    
    # Convert to float64 contiguous arrays
    X_train = np.ascontiguousarray(X_train, dtype=np.float64)
    X_val = np.ascontiguousarray(X_val, dtype=np.float64)
    X_test = np.ascontiguousarray(X_test, dtype=np.float64)
    y_train = np.ascontiguousarray(y_train, dtype=np.float64)
    y_val = np.ascontiguousarray(y_val, dtype=np.float64)
    y_test = np.ascontiguousarray(y_test, dtype=np.float64)
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate comprehensive classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'pr_auc': average_precision_score(y_true, y_proba),
    }
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['tp'] = tp
    metrics['fp'] = fp
    metrics['tn'] = tn
    metrics['fn'] = fn
    
    return metrics


def print_metrics(name, metrics, train_time, predict_time):
    """Print metrics in a formatted table."""
    print(f"\n{name} Results:")
    print("-" * 40)
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Precision:       {metrics['precision']:.4f}")
    print(f"  Recall:          {metrics['recall']:.4f}")
    print(f"  F1 Score:        {metrics['f1']:.4f}")
    print(f"  ROC-AUC:         {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:          {metrics['pr_auc']:.4f}")
    print(f"  Training Time:   {train_time:.2f}s")
    print(f"  Prediction Time: {predict_time:.4f}s")
    print(f"\n  Confusion Matrix:")
    print(f"    TP: {metrics['tp']:,}  FN: {metrics['fn']:,}")
    print(f"    FP: {metrics['fp']:,}  TN: {metrics['tn']:,}")


def benchmark_pkboost_classifier(X_train, X_val, X_test, y_train, y_val, y_test):
    """Benchmark PKBoostClassifier on fraud detection."""
    print("\n" + "=" * 60)
    print("PKBoostClassifier Benchmark")
    print("=" * 60)
    
    from pkboost import PKBoostClassifier
    
    # Create classifier
    clf = PKBoostClassifier()
    
    # Train with validation set
    print("\nTraining...")
    start_time = time.time()
    clf.fit(X_train, y_train, x_val=X_val, y_val=y_val, verbose=True)
    train_time = time.time() - start_time
    print(f"\n✓ Training completed in {train_time:.2f} seconds")
    print(f"  Trees built: {clf.get_n_trees()}")
    
    # Predict on test set
    print("\nPredicting on test set...")
    start_time = time.time()
    y_proba = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test, threshold=0.5)
    predict_time = time.time() - start_time
    print(f"✓ Prediction completed in {predict_time:.4f} seconds")
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    print_metrics("PKBoostClassifier", metrics, train_time, predict_time)
    
    return clf, metrics, train_time


def benchmark_pkboost_adaptive(X_train, X_val, X_test, y_train, y_val, y_test):
    """Benchmark PKBoostAdaptive on fraud detection."""
    print("\n" + "=" * 60)
    print("PKBoostAdaptive Benchmark")
    print("=" * 60)
    
    from pkboost import PKBoostAdaptive
    
    # Create adaptive classifier
    aclf = PKBoostAdaptive()
    
    # Initial training
    print("\nInitial training...")
    start_time = time.time()
    aclf.fit_initial(X_train, y_train, x_val=X_val, y_val=y_val, verbose=True)
    train_time = time.time() - start_time
    print(f"\n✓ Training completed in {train_time:.2f} seconds")
    
    # Status
    print(f"  Vulnerability score: {aclf.get_vulnerability_score():.4f}")
    print(f"  System state: {aclf.get_state()}")
    
    # Predict on test set
    print("\nPredicting on test set...")
    start_time = time.time()
    y_proba = aclf.predict_proba(X_test)
    y_pred = aclf.predict(X_test, threshold=0.5)
    predict_time = time.time() - start_time
    print(f"✓ Prediction completed in {predict_time:.4f} seconds")
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    print_metrics("PKBoostAdaptive", metrics, train_time, predict_time)
    
    return aclf, metrics, train_time


def main():
    print("\n" + "=" * 60)
    print("PKBoost Credit Card Fraud Benchmark")
    print("=" * 60 + "\n")
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(
        TRAIN_PATH, VAL_PATH, TEST_PATH, TARGET_COLUMN
    )
    
    results = {}
    
    # PKBoostClassifier benchmark
    try:
        clf, metrics, train_time = benchmark_pkboost_classifier(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        results['PKBoostClassifier'] = (metrics, train_time)
    except Exception as e:
        print(f"✗ PKBoostClassifier failed: {e}")
    
    # PKBoostAdaptive benchmark
    try:
        aclf, metrics, train_time = benchmark_pkboost_adaptive(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        results['PKBoostAdaptive'] = (metrics, train_time)
    except Exception as e:
        print(f"✗ PKBoostAdaptive failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<25} {'ROC-AUC':>10} {'PR-AUC':>10} {'F1':>10} {'Time':>10}")
    print("-" * 65)
    for name, (m, t) in results.items():
        print(f"{name:<25} {m['roc_auc']:>10.4f} {m['pr_auc']:>10.4f} {m['f1']:>10.4f} {t:>9.2f}s")
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
