#!/usr/bin/env python3
"""
Three-Way Comparison: PKBoost (Rust) vs XGBoost vs LightGBM
Combines Rust PKBoost results with Python XGBoost/LightGBM
"""

import pandas as pd
import numpy as np
import time
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path

# PKBoost results from Rust benchmark (from previous run)
PKBOOST_RESULTS = {
    'baseline_pr_auc': 0.8487,
    'baseline_roc_auc': 0.9706,
    'drift_pr_auc': 0.8422,
    'drift_roc_auc': 0.9664,
    'train_time': 6.85,
    'pred_time_ms': 60.1
}

def load_data():
    train = pd.read_csv('data/creditcard_train.csv')
    val = pd.read_csv('data/creditcard_val.csv')
    test = pd.read_csv('data/creditcard_test.csv')
    
    X_train = train.drop('Class', axis=1).values
    y_train = train['Class'].values
    X_val = val.drop('Class', axis=1).values
    y_val = val['Class'].values
    X_test = test.drop('Class', axis=1).values
    y_test = test['Class'].values
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def add_drift(X, n_features=10, noise_std=2.0):
    X_drift = X.copy()
    for i in range(min(n_features, X.shape[1])):
        X_drift[:, i] += np.random.randn(len(X)) * noise_std
    return X_drift

def benchmark_model(name, model, X_train, y_train, X_val, y_val, X_test, y_test, X_test_drift):
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    
    # Train
    start = time.time()
    if 'LightGBM' in name:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    else:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.time() - start
    
    # Baseline prediction
    start = time.time()
    y_pred = model.predict_proba(X_test)[:, 1]
    pred_time = time.time() - start
    
    pr_auc = average_precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    print(f"Training:   {train_time:.2f}s")
    print(f"Prediction: {pred_time*1000:.1f}ms ({len(X_test)/pred_time:.0f} samples/s)")
    print(f"PR-AUC:     {pr_auc:.4f}")
    print(f"ROC-AUC:    {roc_auc:.4f}")
    
    # Drift prediction
    y_pred_drift = model.predict_proba(X_test_drift)[:, 1]
    pr_auc_drift = average_precision_score(y_test, y_pred_drift)
    roc_auc_drift = roc_auc_score(y_test, y_pred_drift)
    degradation = abs((pr_auc - pr_auc_drift) / pr_auc * 100)
    
    print(f"\nUnder Drift:")
    print(f"  PR-AUC:      {pr_auc_drift:.4f}")
    print(f"  ROC-AUC:     {roc_auc_drift:.4f}")
    print(f"  Degradation: {degradation:.2f}%")
    
    return {
        'baseline_pr_auc': pr_auc,
        'baseline_roc_auc': roc_auc,
        'drift_pr_auc': pr_auc_drift,
        'drift_roc_auc': roc_auc_drift,
        'degradation': degradation,
        'train_time': train_time,
        'pred_time_ms': pred_time * 1000
    }

def main():
    print("="*80)
    print("THREE-WAY COMPARISON: PKBoost vs XGBoost vs LightGBM")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    X_test_drift = add_drift(X_test, n_features=10, noise_std=2.0)
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")
    print(f"  Drift: 10 features corrupted (noise_std=2.0)")
    
    # Benchmark XGBoost
    xgb_results = benchmark_model(
        "XGBoost",
        xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', tree_method='hist'),
        X_train, y_train, X_val, y_val, X_test, y_test, X_test_drift
    )
    
    # Benchmark LightGBM
    lgb_results = benchmark_model(
        "LightGBM",
        lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        X_train, y_train, X_val, y_val, X_test, y_test, X_test_drift
    )
    
    # Add PKBoost results
    pkboost_degradation = abs((PKBOOST_RESULTS['baseline_pr_auc'] - PKBOOST_RESULTS['drift_pr_auc']) / PKBOOST_RESULTS['baseline_pr_auc'] * 100)
    pkboost_results = {**PKBOOST_RESULTS, 'degradation': pkboost_degradation}
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Model':<12} {'Baseline':<10} {'Drift':<10} {'Degrad':<10} {'Train':<10} {'Pred':<10}")
    print(f"{'':12} {'PR-AUC':<10} {'PR-AUC':<10} {'(%)':<10} {'(s)':<10} {'(ms)':<10}")
    print("-"*80)
    
    results = {
        'PKBoost': pkboost_results,
        'XGBoost': xgb_results,
        'LightGBM': lgb_results
    }
    
    for name, r in results.items():
        print(f"{name:<12} {r['baseline_pr_auc']:<10.4f} {r['drift_pr_auc']:<10.4f} "
              f"{r['degradation']:<10.2f} {r['train_time']:<10.2f} {r['pred_time_ms']:<10.1f}")
    
    print("="*80)
    
    # Winner analysis
    print("\nWINNER ANALYSIS:")
    print("-"*80)
    
    best_baseline = max(results.items(), key=lambda x: x[1]['baseline_pr_auc'])
    best_drift = min(results.items(), key=lambda x: x[1]['degradation'])
    fastest_train = min(results.items(), key=lambda x: x[1]['train_time'])
    fastest_pred = min(results.items(), key=lambda x: x[1]['pred_time_ms'])
    
    print(f"Best Baseline Accuracy:  {best_baseline[0]:<12} ({best_baseline[1]['baseline_pr_auc']:.4f} PR-AUC)")
    print(f"Best Drift Resilience:   {best_drift[0]:<12} ({best_drift[1]['degradation']:.2f}% degradation)")
    print(f"Fastest Training:        {fastest_train[0]:<12} ({fastest_train[1]['train_time']:.2f}s)")
    print(f"Fastest Prediction:      {fastest_pred[0]:<12} ({fastest_pred[1]['pred_time_ms']:.1f}ms)")
    
    # Overall winner
    print("\n" + "="*80)
    if best_baseline[0] == best_drift[0]:
        print(f"OVERALL WINNER: {best_baseline[0]}")
        print(f"  - Best accuracy: {best_baseline[1]['baseline_pr_auc']:.4f} PR-AUC")
        print(f"  - Best resilience: {best_drift[1]['degradation']:.2f}% degradation")
        
        if best_baseline[0] == 'PKBoost':
            xgb_improvement = ((pkboost_results['baseline_pr_auc'] - xgb_results['baseline_pr_auc']) / xgb_results['baseline_pr_auc'] * 100)
            lgb_improvement = ((pkboost_results['baseline_pr_auc'] - lgb_results['baseline_pr_auc']) / lgb_results['baseline_pr_auc'] * 100)
            print(f"  - {xgb_improvement:+.1f}% better than XGBoost")
            print(f"  - {lgb_improvement:+.1f}% better than LightGBM")
    
    # Save results
    Path('results').mkdir(exist_ok=True)
    df = pd.DataFrame(results).T
    df.to_csv('results/threeway_final.csv')
    print(f"\nResults saved to: results/threeway_final.csv")
    print("="*80)

if __name__ == '__main__':
    main()
