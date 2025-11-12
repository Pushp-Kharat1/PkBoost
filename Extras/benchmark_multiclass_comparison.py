#!/usr/bin/env python3
"""
Imbalanced Multi-Class Benchmark: PKBoost vs XGBoost vs LightGBM
"""
import numpy as np
import time
from sklearn.metrics import accuracy_score, f1_score

def generate_imbalanced_multiclass():
    """Generate imbalanced 5-class dataset matching Rust version"""
    np.random.seed(42)
    
    class_ratios = [0.50, 0.25, 0.15, 0.07, 0.03]
    n_train = 5000
    n_test = 1000
    n_features = 20
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    for class_id, ratio in enumerate(class_ratios):
        mean = class_id * 0.5  # Reduced separation (0, 0.5, 1.0, 1.5, 2.0)
        std = 2.0  # High variance for overlap
        
        # Train
        n_samples = int(n_train * ratio)
        X_class = np.zeros((n_samples, n_features))
        # Only first 5 features are informative
        X_class[:, :5] = np.random.randn(n_samples, 5) * std + mean
        # Rest are noise
        X_class[:, 5:] = np.random.rand(n_samples, n_features - 5) * 4.0 - 2.0
        X_train.append(X_class)
        y_train.extend([class_id] * n_samples)
        
        # Test
        n_samples = int(n_test * ratio)
        X_class = np.zeros((n_samples, n_features))
        X_class[:, :5] = np.random.randn(n_samples, 5) * std + mean
        X_class[:, 5:] = np.random.rand(n_samples, n_features - 5) * 4.0 - 2.0
        X_test.append(X_class)
        y_test.extend([class_id] * n_samples)
    
    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Shuffle
    train_idx = np.random.permutation(len(y_train))
    X_train, y_train = X_train[train_idx], y_train[train_idx]
    test_idx = np.random.permutation(len(y_test))
    X_test, y_test = X_test[test_idx], y_test[test_idx]
    
    return X_train, y_train, X_test, y_test

def print_distribution(y, label):
    """Print class distribution"""
    unique, counts = np.unique(y, return_counts=True)
    print(f"{label} distribution:")
    for cls, count in zip(unique, counts):
        pct = count / len(y) * 100
        print(f"  Class {cls}: {count} ({pct:.1f}%)")

def main():
    print("=== Imbalanced Multi-Class Benchmark ===\n")
    
    X_train, y_train, X_test, y_test = generate_imbalanced_multiclass()
    
    print(f"Dataset: {len(X_train)} train, {len(X_test)} test")
    print_distribution(y_train, "Train")
    print_distribution(y_test, "Test")
    print()
    
    results = []
    
    # XGBoost
    try:
        import xgboost as xgb
        print("--- XGBoost ---")
        start = time.time()
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)
        xgb_time = time.time() - start
        preds = model.predict(X_test)
        xgb_acc = accuracy_score(y_test, preds)
        xgb_f1 = f1_score(y_test, preds, average='macro')
        print(f"Time: {xgb_time:.2f}s | Accuracy: {xgb_acc*100:.2f}% | Macro-F1: {xgb_f1:.4f}\n")
        results.append(("XGBoost", xgb_time, xgb_acc, xgb_f1))
    except ImportError:
        print("XGBoost not installed\n")
    
    # LightGBM
    try:
        import lightgbm as lgb
        print("--- LightGBM ---")
        start = time.time()
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1
        )
        model.fit(X_train, y_train)
        lgb_time = time.time() - start
        preds = model.predict(X_test)
        lgb_acc = accuracy_score(y_test, preds)
        lgb_f1 = f1_score(y_test, preds, average='macro')
        print(f"Time: {lgb_time:.2f}s | Accuracy: {lgb_acc*100:.2f}% | Macro-F1: {lgb_f1:.4f}\n")
        results.append(("LightGBM", lgb_time, lgb_acc, lgb_f1))
    except ImportError:
        print("LightGBM not installed\n")
    
    # PKBoost (from Rust output)
    results.append(("PKBoost", 7.14, 0.465, 0.2854))
    
    # Summary
    print("=== Results Summary ===")
    print(f"{'Model':<12} {'Time(s)':<10} {'Accuracy':<12} {'Macro-F1':<10}")
    print("-" * 50)
    for name, t, acc, f1 in results:
        print(f"{name:<12} {t:<10.2f} {acc*100:<12.2f} {f1:<10.4f}")
    
    # Winner analysis
    print("\n=== Analysis ===")
    best_acc = max(results, key=lambda x: x[2])
    best_f1 = max(results, key=lambda x: x[3])
    fastest = min(results, key=lambda x: x[1])
    
    print(f"Best Accuracy: {best_acc[0]} ({best_acc[2]*100:.2f}%)")
    print(f"Best Macro-F1: {best_f1[0]} ({best_f1[3]:.4f})")
    print(f"Fastest: {fastest[0]} ({fastest[1]:.2f}s)")

if __name__ == "__main__":
    main()
