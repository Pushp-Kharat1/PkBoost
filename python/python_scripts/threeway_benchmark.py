#!/usr/bin/env python3
"""
Three-Way Comparison: PKBoost vs XGBoost vs LightGBM
Tests baseline and drift scenarios on Credit Card Fraud dataset
"""

import pandas as pd
import numpy as np
import time
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Import models
try:
    import pkboost
    PKBOOST_AVAILABLE = True
except ImportError:
    PKBOOST_AVAILABLE = False
    print("Warning: pkboost not installed. Install with: pip install pkboost")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: lightgbm not installed. Install with: pip install lightgbm")

from sklearn.metrics import roc_auc_score, average_precision_score


def load_data(data_dir='data'):
    """Load Credit Card dataset"""
    print("\n[1/6] Loading Credit Card dataset...")
    train = pd.read_csv(f'{data_dir}/creditcard_train.csv')
    val = pd.read_csv(f'{data_dir}/creditcard_val.csv')
    test = pd.read_csv(f'{data_dir}/creditcard_test.csv')
    
    X_train = train.drop('Class', axis=1).values
    y_train = train['Class'].values
    X_val = val.drop('Class', axis=1).values
    y_val = val['Class'].values
    X_test = test.drop('Class', axis=1).values
    y_test = test['Class'].values
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def add_drift(X, n_features=10, noise_std=2.0):
    """Add covariate shift to features"""
    X_drift = X.copy()
    for i in range(min(n_features, X.shape[1])):
        X_drift[:, i] += np.random.randn(len(X)) * noise_std
    return X_drift


def benchmark_pkboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """Benchmark PKBoost"""
    print("\n" + "="*60)
    print("PKBoost v2.0.1 (with Progressive Precision)")
    print("="*60)
    
    # Training (convert to list for PKBoost)
    start = time.time()
    model = pkboost.PKBoostClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05
    )
    model.fit(X_train.tolist(), y_train.tolist())
    train_time = time.time() - start
    
    # Prediction (batch for speed)
    start = time.time()
    y_pred = np.array(model.predict_proba(X_test.tolist()))
    pred_time = time.time() - start
    
    pr_auc = average_precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    print(f"Training:   {train_time:.2f}s")
    print(f"Prediction: {pred_time*1000:.1f}ms ({len(X_test)/pred_time:.0f} samples/s)")
    print(f"PR-AUC:     {pr_auc:.4f}")
    print(f"ROC-AUC:    {roc_auc:.4f}")
    
    return model, pr_auc, roc_auc, train_time, pred_time


def benchmark_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """Benchmark XGBoost"""
    print("\n" + "="*60)
    print("XGBoost (Default Parameters)")
    print("="*60)
    
    # Training
    start = time.time()
    model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric='logloss',
        tree_method='hist'  # Faster
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    train_time = time.time() - start
    
    # Prediction (batch for speed)
    start = time.time()
    y_pred = model.predict_proba(X_test)[:, 1]
    pred_time = time.time() - start
    
    pr_auc = average_precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    print(f"Training:   {train_time:.2f}s")
    print(f"Prediction: {pred_time*1000:.1f}ms ({len(X_test)/pred_time:.0f} samples/s)")
    print(f"PR-AUC:     {pr_auc:.4f}")
    print(f"ROC-AUC:    {roc_auc:.4f}")
    
    return model, pr_auc, roc_auc, train_time, pred_time


def benchmark_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    """Benchmark LightGBM"""
    print("\n" + "="*60)
    print("LightGBM (Default Parameters)")
    print("="*60)
    
    # Training
    start = time.time()
    model = lgb.LGBMClassifier(
        n_estimators=100,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    train_time = time.time() - start
    
    # Prediction (batch for speed)
    start = time.time()
    y_pred = model.predict_proba(X_test)[:, 1]
    pred_time = time.time() - start
    
    pr_auc = average_precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    
    print(f"Training:   {train_time:.2f}s")
    print(f"Prediction: {pred_time*1000:.1f}ms ({len(X_test)/pred_time:.0f} samples/s)")
    print(f"PR-AUC:     {pr_auc:.4f}")
    print(f"ROC-AUC:    {roc_auc:.4f}")
    
    return model, pr_auc, roc_auc, train_time, pred_time


def test_drift(model, X_test, y_test, model_name, baseline_pr_auc, is_pkboost=False):
    """Test model under drift"""
    start = time.time()
    if is_pkboost:
        y_pred = np.array(model.predict_proba(X_test.tolist()))
    else:
        y_pred = model.predict_proba(X_test)[:, 1]
    pred_time = time.time() - start
    
    pr_auc = average_precision_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    degradation = abs((baseline_pr_auc - pr_auc) / baseline_pr_auc * 100)
    
    print(f"\n{model_name} (Under Drift):")
    print(f"  Prediction: {pred_time*1000:.1f}ms")
    print(f"  PR-AUC:     {pr_auc:.4f}")
    print(f"  ROC-AUC:    {roc_auc:.4f}")
    print(f"  Degradation: {degradation:.2f}%")
    
    return pr_auc, roc_auc, degradation


def print_summary_table(results):
    """Print comparison table"""
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Model':<15} {'Baseline':<12} {'Under Drift':<12} {'Degradation':<12} {'Train Time':<12}")
    print(f"{'':15} {'PR-AUC':<12} {'PR-AUC':<12} {'(%)':<12} {'(s)':<12}")
    print("-"*80)
    
    for model_name, data in results.items():
        print(f"{model_name:<15} {data['baseline_pr_auc']:<12.4f} {data['drift_pr_auc']:<12.4f} "
              f"{data['degradation']:<12.2f} {data['train_time']:<12.2f}")
    
    print("="*80)


def main():
    print("="*80)
    print("THREE-WAY COMPARISON: PKBoost vs XGBoost vs LightGBM")
    print("="*80)
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    results = {}
    
    # Baseline benchmarks
    print("\n" + "="*80)
    print("BASELINE (No Drift)")
    print("="*80)
    
    if PKBOOST_AVAILABLE:
        try:
            model_pk, pr_pk, roc_pk, train_pk, pred_pk = benchmark_pkboost(
                X_train, y_train, X_val, y_val, X_test, y_test
            )
            results['PKBoost'] = {
                'model': model_pk,
                'baseline_pr_auc': pr_pk,
                'baseline_roc_auc': roc_pk,
                'train_time': train_pk,
                'pred_time': pred_pk
            }
        except Exception as e:
            print(f"PKBoost error: {e}")
    
    if XGBOOST_AVAILABLE:
        model_xgb, pr_xgb, roc_xgb, train_xgb, pred_xgb = benchmark_xgboost(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        results['XGBoost'] = {
            'model': model_xgb,
            'baseline_pr_auc': pr_xgb,
            'baseline_roc_auc': roc_xgb,
            'train_time': train_xgb,
            'pred_time': pred_xgb
        }
    
    if LIGHTGBM_AVAILABLE:
        model_lgb, pr_lgb, roc_lgb, train_lgb, pred_lgb = benchmark_lightgbm(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        results['LightGBM'] = {
            'model': model_lgb,
            'baseline_pr_auc': pr_lgb,
            'baseline_roc_auc': roc_lgb,
            'train_time': train_lgb,
            'pred_time': pred_lgb
        }
    
    # Drift scenario
    print("\n" + "="*80)
    print("DRIFT SCENARIO (10 Features Corrupted, noise_std=2.0)")
    print("="*80)
    
    X_test_drift = add_drift(X_test, n_features=10, noise_std=2.0)
    print(f"Added drift to 10 features")
    
    for model_name, data in results.items():
        is_pkboost = (model_name == 'PKBoost')
        pr_drift, roc_drift, degradation = test_drift(
            data['model'], X_test_drift, y_test, model_name, data['baseline_pr_auc'], is_pkboost
        )
        data['drift_pr_auc'] = pr_drift
        data['drift_roc_auc'] = roc_drift
        data['degradation'] = degradation
    
    # Summary
    print_summary_table(results)
    
    # Save results
    df = pd.DataFrame({
        'Model': list(results.keys()),
        'Baseline_PR_AUC': [r['baseline_pr_auc'] for r in results.values()],
        'Drift_PR_AUC': [r['drift_pr_auc'] for r in results.values()],
        'Degradation_%': [r['degradation'] for r in results.values()],
        'Train_Time_s': [r['train_time'] for r in results.values()],
        'Pred_Time_ms': [r['pred_time']*1000 for r in results.values()]
    })
    
    Path('results').mkdir(exist_ok=True)
    df.to_csv('results/threeway_comparison.csv', index=False)
    print("\nResults saved to: results/threeway_comparison.csv")
    
    # Winner analysis
    print("\n" + "="*80)
    print("WINNER ANALYSIS")
    print("="*80)
    
    best_baseline = max(results.items(), key=lambda x: x[1]['baseline_pr_auc'])
    best_drift = min(results.items(), key=lambda x: x[1]['degradation'])
    
    print(f"\nBest Baseline Accuracy: {best_baseline[0]} ({best_baseline[1]['baseline_pr_auc']:.4f} PR-AUC)")
    print(f"Best Drift Resilience:  {best_drift[0]} ({best_drift[1]['degradation']:.2f}% degradation)")
    
    if best_baseline[0] == best_drift[0]:
        print(f"\nOVERALL WINNER: {best_baseline[0]}")
        print("  - Best accuracy on baseline")
        print("  - Best resilience under drift")


if __name__ == '__main__':
    main()
