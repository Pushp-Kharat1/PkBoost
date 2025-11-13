#!/usr/bin/env python3
"""Compare PKBoost vs XGBoost vs LightGBM on Dry Bean dataset"""
import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score, f1_score, classification_report

def load_data():
    train = pd.read_csv("data/drybean_train.csv")
    test = pd.read_csv("data/drybean_test.csv")
    
    X_train = train.iloc[:, :-1].values
    y_train = train.iloc[:, -1].values
    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values
    
    return X_train, y_train, X_test, y_test

def main():
    print("=== Dry Bean Dataset Comparison ===\n")
    
    X_train, y_train, X_test, y_test = load_data()
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}, Classes: {len(np.unique(y_train))}\n")
    
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
        xgb_macro = f1_score(y_test, preds, average='macro')
        xgb_weighted = f1_score(y_test, preds, average='weighted')
        print(f"Time: {xgb_time:.2f}s | Accuracy: {xgb_acc*100:.2f}% | Macro-F1: {xgb_macro:.4f} | Weighted-F1: {xgb_weighted:.4f}\n")
        results.append(("XGBoost", xgb_time, xgb_acc, xgb_macro, xgb_weighted))
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
        lgb_macro = f1_score(y_test, preds, average='macro')
        lgb_weighted = f1_score(y_test, preds, average='weighted')
        print(f"Time: {lgb_time:.2f}s | Accuracy: {lgb_acc*100:.2f}% | Macro-F1: {lgb_macro:.4f} | Weighted-F1: {lgb_weighted:.4f}\n")
        results.append(("LightGBM", lgb_time, lgb_acc, lgb_macro, lgb_weighted))
    except ImportError:
        print("LightGBM not installed\n")
    
    # PKBoost (from Rust output)
    results.append(("PKBoost", 14.17, 0.9236, 0.9360, 0.9237))
    
    # Summary
    print("=== Results Summary ===")
    print(f"{'Model':<12} {'Time(s)':<10} {'Accuracy':<12} {'Macro-F1':<12} {'Weighted-F1':<12}")
    print("-" * 65)
    for name, t, acc, macro, weighted in results:
        print(f"{name:<12} {t:<10.2f} {acc*100:<12.2f} {macro:<12.4f} {weighted:<12.4f}")
    
    # Analysis
    print("\n=== Analysis ===")
    best_acc = max(results, key=lambda x: x[2])
    best_macro = max(results, key=lambda x: x[3])
    fastest = min(results, key=lambda x: x[1])
    
    print(f"Best Accuracy: {best_acc[0]} ({best_acc[2]*100:.2f}%)")
    print(f"Best Macro-F1: {best_macro[0]} ({best_macro[3]:.4f})")
    print(f"Fastest: {fastest[0]} ({fastest[1]:.2f}s)")

if __name__ == "__main__":
    main()
