#!/usr/bin/env python3
"""Compare drift resilience: PKBoost vs XGBoost vs LightGBM"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def load_data():
    train = pd.read_csv("data/drybean_train.csv")
    test = pd.read_csv("data/drybean_test.csv")
    
    X_train = train.iloc[:, :-1].values
    y_train = train.iloc[:, -1].values.astype(int)
    X_test = test.iloc[:, :-1].values
    y_test = test.iloc[:, -1].values.astype(int)
    
    return X_train, y_train, X_test, y_test

def inject_drift(X, intensity):
    """Inject Gaussian noise to 50% of features"""
    X_drift = X.copy()
    n_features = X.shape[1]
    n_drift_features = n_features // 2
    
    noise = np.random.randn(X.shape[0], n_drift_features) * intensity
    X_drift[:, :n_drift_features] += noise
    
    return X_drift

def main():
    print("=== Dry Bean Drift Comparison ===\n")
    
    X_train, y_train, X_test, y_test = load_data()
    
    drift_levels = [0.0, 0.5, 1.0, 2.0, 3.0]
    results = {}
    
    # XGBoost
    try:
        import xgboost as xgb
        print("Training XGBoost...")
        model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                   random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        
        xgb_results = []
        for drift in drift_levels:
            X_drift = inject_drift(X_test, drift)
            preds = model.predict(X_drift)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='macro')
            xgb_results.append((acc, f1))
        results['XGBoost'] = xgb_results
        print("[OK] XGBoost done")
    except ImportError:
        print("XGBoost not installed")
    
    # LightGBM
    try:
        import lightgbm as lgb
        print("Training LightGBM...")
        model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                    random_state=42, verbosity=-1)
        model.fit(X_train, y_train)
        
        lgb_results = []
        for drift in drift_levels:
            X_drift = inject_drift(X_test, drift)
            preds = model.predict(X_drift)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='macro')
            lgb_results.append((acc, f1))
        results['LightGBM'] = lgb_results
        print("[OK] LightGBM done")
    except ImportError:
        print("LightGBM not installed")
    
    # PKBoost (from Rust output)
    results['PKBoost'] = [
        (0.9254, 0.9383),  # 0.0
        (0.9236, 0.9364),  # 0.5
        (0.9243, 0.9381),  # 1.0
        (0.9247, 0.9377),  # 2.0
        (0.9214, 0.9342),  # 3.0
    ]
    
    # Display results
    print("\n=== Accuracy Under Drift ===")
    print(f"{'Drift':<10} ", end="")
    for model in results.keys():
        print(f"{model:<12} ", end="")
    print()
    print("-" * (10 + 12 * len(results)))
    
    for i, drift in enumerate(drift_levels):
        print(f"{drift:<10.1f} ", end="")
        for model in results.keys():
            acc = results[model][i][0]
            print(f"{acc*100:<12.2f} ", end="")
        print()
    
    print("\n=== Macro-F1 Under Drift ===")
    print(f"{'Drift':<10} ", end="")
    for model in results.keys():
        print(f"{model:<12} ", end="")
    print()
    print("-" * (10 + 12 * len(results)))
    
    for i, drift in enumerate(drift_levels):
        print(f"{drift:<10.1f} ", end="")
        for model in results.keys():
            f1 = results[model][i][1]
            print(f"{f1:<12.4f} ", end="")
        print()
    
    # Degradation analysis
    print("\n=== Accuracy Degradation (Baseline -> Drift=3.0) ===")
    for model, res in results.items():
        baseline = res[0][0]
        final = res[-1][0]
        degradation = (baseline - final) / baseline * 100
        print(f"{model:<12} {baseline*100:.2f}% → {final*100:.2f}% ({degradation:+.1f}%)")
    
    print("\n=== Winner: Most Drift-Resilient ===")
    best_model = min(results.items(), 
                     key=lambda x: abs(x[1][0][0] - x[1][-1][0]))
    print(f"{best_model[0]} (only {abs(best_model[1][0][0] - best_model[1][-1][0])*100:.1f}% degradation)")

if __name__ == "__main__":
    np.random.seed(42)
    main()
