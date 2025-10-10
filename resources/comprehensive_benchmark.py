# real_world_benchmark.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.datasets import fetch_openml
import xgboost as xgb
import subprocess
import time
import os

def load_sklearn_dataset(name, target_col=None, positive_class=1):
    """Load dataset from sklearn/OpenML"""
    print(f"\nLoading {name}...")
    
    if name == "breast-cancer":
        data = fetch_openml('breast-w', version=1, as_frame=True, parser='auto')
        X = data.data
        y = (data.target == '2').astype(int)  # malignant = 1
        
    elif name == "credit-g":
        data = fetch_openml('credit-g', version=1, as_frame=True, parser='auto')
        X = data.data
        y = (data.target == 'bad').astype(int)
        
    elif name == "phoneme":
        data = fetch_openml('phoneme', version=1, as_frame=True, parser='auto')
        X = data.data
        y = (data.target == '1').astype(int)
        
    elif name == "diabetes":
        data = fetch_openml('diabetes', version=1, as_frame=True, parser='auto')
        X = data.data
        y = (data.target == 'tested_positive').astype(int)
        
    elif name == "spambase":
        data = fetch_openml('spambase', version=1, as_frame=True, parser='auto')
        X = data.data
        y = data.target.astype(int)
    
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # Handle categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    # Drop rows with NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    print(f"  Shape: {X.shape}, Positive ratio: {y.mean():.3f}")
    
    return X.values, y.values

def prepare_and_save(X, y, dataset_name):
    """Split, scale, and save dataset"""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    os.makedirs('data', exist_ok=True)
    pd.DataFrame(X_train).assign(Class=y_train).to_csv('data/train_large.csv', index=False)
    pd.DataFrame(X_val).assign(Class=y_val).to_csv('data/val_large.csv', index=False)
    pd.DataFrame(X_test).assign(Class=y_test).to_csv('data/test_large.csv', index=False)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train XGBoost"""
    scale_pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'verbosity': 0,
    }
    
    start = time.time()
    model = xgb.train(params, dtrain, num_boost_round=500,
                     evals=[(dval, 'val')], early_stopping_rounds=50, verbose_eval=False)
    train_time = time.time() - start
    
    y_pred = model.predict(dtest)
    
    return {
        'pr_auc': average_precision_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred),
        'f1': f1_score(y_test, (y_pred >= 0.5).astype(int)),
        'time': train_time
    }

def train_pkboost():
    """Train PKBoost"""
    subprocess.run(["cargo", "build", "--release"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = subprocess.run(["cargo", "run", "--release", "--bin", "benchmark"],
                           capture_output=True, text=True)
    
    def extract(text, pattern):
        for line in text.split('\n'):
            if pattern in line:
                try:
                    return float(line.split(':')[-1].strip().replace('s', ''))
                except:
                    pass
        return None
    
    return {
        'pr_auc': extract(result.stdout, 'Test PR-AUC'),
        'roc_auc': extract(result.stdout, 'Test ROC-AUC'),
        'f1': extract(result.stdout, 'F1 Score'),
        'time': extract(result.stdout, 'Training time')
    }

# Datasets to test
DATASETS = [
    "breast-cancer",  # 699 samples, 9 features, ~35% positive
    "credit-g",       # 1000 samples, 20 features, ~30% positive
    "diabetes",       # 768 samples, 8 features, ~35% positive
    "phoneme",        # 5404 samples, 5 features, ~29% positive
    "spambase",       # 4601 samples, 57 features, ~39% positive
]

print("="*80)
print("REAL-WORLD BENCHMARK: PKBoost vs XGBoost")
print("="*80)

results = []

for dataset_name in DATASETS:
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset_name}")
    print(f"{'='*80}")
    
    try:
        # Load and prepare
        X, y = load_sklearn_dataset(dataset_name)
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_and_save(X, y, dataset_name)
        
        # Train both models
        print("\n--- XGBoost ---")
        xgb_res = train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test)
        print(f"  PR-AUC: {xgb_res['pr_auc']:.4f}, Time: {xgb_res['time']:.1f}s")
        
        print("\n--- PKBoost ---")
        pkb_res = train_pkboost()
        print(f"  PR-AUC: {pkb_res['pr_auc']:.4f}, Time: {pkb_res['time']:.1f}s")
        
        # Store results
        winner = 'PKBoost' if pkb_res['pr_auc'] > xgb_res['pr_auc'] else 'XGBoost'
        diff_pct = ((pkb_res['pr_auc'] - xgb_res['pr_auc']) / xgb_res['pr_auc']) * 100
        
        results.append({
            'dataset': dataset_name,
            'n_samples': len(y_train),
            'n_features': X_train.shape[1],
            'pos_ratio': y_train.mean(),
            'xgb_pr': xgb_res['pr_auc'],
            'pkb_pr': pkb_res['pr_auc'],
            'xgb_roc': xgb_res['roc_auc'],
            'pkb_roc': pkb_res['roc_auc'],
            'xgb_time': xgb_res['time'],
            'pkb_time': pkb_res['time'],
            'winner': winner,
            'diff_pct': diff_pct
        })
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

# Final analysis
print(f"\n\n{'='*80}")
print("FINAL RESULTS")
print(f"{'='*80}\n")

if results:
    df = pd.DataFrame(results)
    
    print("PER-DATASET BREAKDOWN:")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{row['dataset']:20s}: XGB={row['xgb_pr']:.4f}, PKB={row['pkb_pr']:.4f} "
              f"({row['diff_pct']:+.1f}%) - {row['winner']}")
    
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    pkb_wins = sum(df['pkb_pr'] > df['xgb_pr'])
    total = len(df)
    
    print(f"\nPKBoost wins: {pkb_wins}/{total} datasets ({pkb_wins/total*100:.1f}%)")
    print(f"\nPR-AUC:")
    print(f"  XGBoost: {df['xgb_pr'].mean():.4f} ± {df['xgb_pr'].std():.4f}")
    print(f"  PKBoost: {df['pkb_pr'].mean():.4f} ± {df['pkb_pr'].std():.4f}")
    print(f"  Average difference: {df['diff_pct'].mean():+.1f}%")
    
    print(f"\nSpeed:")
    print(f"  XGBoost: {df['xgb_time'].mean():.1f}s average")
    print(f"  PKBoost: {df['pkb_time'].mean():.1f}s average")
    print(f"  PKBoost is {(df['pkb_time'].mean() / df['xgb_time'].mean()):.1f}x slower")
    
    print(f"\n{'='*80}")
    print("VERDICT")
    print(f"{'='*80}")
    
    if pkb_wins >= 4:
        print("PKBoost consistently outperforms XGBoost on real-world data.")
        print("This is publication-worthy if margins are statistically significant.")
    elif pkb_wins >= 3:
        print("Mixed results. PKBoost wins majority but not dominant.")
        print("More investigation needed to identify when it excels.")
    elif pkb_wins >= 2:
        print("Results are split. No clear winner.")
        print("Performance appears dataset-dependent.")
    else:
        print("XGBoost dominates on real-world data.")
        print("PKBoost does not outperform the baseline.")
        print("\nRecommendation: Treat as learning project, not publishable research.")
    
    # Save results
    df.to_csv('real_world_benchmark_results.csv', index=False)
    print(f"\nResults saved to 'real_world_benchmark_results.csv'")
else:
    print("No successful benchmarks completed.")