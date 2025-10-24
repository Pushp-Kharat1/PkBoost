import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*UndefinedMetricWarning.*')

def load_data(path):
    df = pd.read_csv(path)
    y = df.pop('Class')
    return df.values, y.values

def evaluate_batch(model, X, y, model_type='lgb'):
    if model_type == 'lgb':
        preds = model.predict(X, num_iteration=model.best_iteration)
    else:  # xgb
        dmat = xgb.DMatrix(X)
        preds = model.predict(dmat, iteration_range=(0, model.best_iteration + 1))
    
    pr_auc = average_precision_score(y, preds)
    roc_auc = roc_auc_score(y, preds)
    y_pred = (preds >= 0.5).astype(int)
    f1 = f1_score(y, y_pred)
    return pr_auc, roc_auc, f1

def main():
    print("\n=== DRIFT COMPARISON: LightGBM vs XGBoost vs PKBoost ===\n")
    
    # Load data
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / 'data'
    
    print("Loading data...")
    X_train, y_train = load_data(data_path / 'synthetic_financial_train.csv')
    X_val, y_val = load_data(data_path / 'synthetic_financial_val.csv')
    X_test, y_test = load_data(data_path / 'synthetic_financial_test.csv')
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # === TRAIN LIGHTGBM ===
    print(" Training LightGBM ")
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    lgb_params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",
        "verbosity": -1
    }
    lgb_model = lgb.train(
        lgb_params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    print("LightGBM training complete")

    # === TRAIN XGBOOST ===
    print("\n=== Training XGBoost ===")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "verbosity": 0,
        "base_score": 0.5
    }
    xgb_model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=2000,
        evals=[(dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    print("XGBoost training complete")

    # === PHASE 1: NORMAL DATA ===
    print("\n=== PHASE 1: NORMAL DATA ===")
    batch_size = 5000  # keep benchmark batch size
    X_normal = X_test[:min(batch_size, len(X_test))]
    y_normal = y_test[:min(batch_size, len(X_test))]

    lgb_pr, lgb_roc, lgb_f1 = evaluate_batch(lgb_model, X_normal, y_normal, 'lgb')
    xgb_pr, xgb_roc, xgb_f1 = evaluate_batch(xgb_model, X_normal, y_normal, 'xgb')

    print(f"LightGBM - PR-AUC: {lgb_pr:.4f}, ROC-AUC: {lgb_roc:.4f}, F1: {lgb_f1:.4f}")
    print(f"XGBoost  - PR-AUC: {xgb_pr:.4f}, ROC-AUC: {xgb_roc:.4f}, F1: {xgb_f1:.4f}")

    baseline_lgb = lgb_pr
    baseline_xgb = xgb_pr

    # === PHASE 2: SUDDEN CATASTROPHIC DRIFT ===
    print("\n=== PHASE 2: SUDDEN CATASTROPHIC DRIFT ===")
    drift_features = [5, 10, 15, 20, 25]
    print(f"Applying drift to features: {drift_features}")

    X_drift = X_test.copy()
    for feat_idx in drift_features:
        if feat_idx < X_drift.shape[1]:
            X_drift[:, feat_idx] = -X_drift[:, feat_idx] + 10.0

    # === EVALUATE UNDER DRIFT ===
    print("\nEvaluating on drifted data...")
    lgb_drift_metrics = []
    xgb_drift_metrics = []

    for i in range(batch_size, len(X_drift), batch_size):
        end = min(i + batch_size, len(X_drift))
        if end - i < 100:
            continue  # skip too small batches

        X_batch = X_drift[i:end]
        y_batch = y_test[i:end]

        lgb_pr, lgb_roc, lgb_f1 = evaluate_batch(lgb_model, X_batch, y_batch, 'lgb')
        xgb_pr, xgb_roc, xgb_f1 = evaluate_batch(xgb_model, X_batch, y_batch, 'xgb')

        lgb_drift_metrics.append((lgb_pr, lgb_roc, lgb_f1))
        xgb_drift_metrics.append((xgb_pr, xgb_roc, xgb_f1))

        print(f"Batch {len(lgb_drift_metrics)} - LightGBM PR-AUC: {lgb_pr:.4f}, XGBoost PR-AUC: {xgb_pr:.4f}")

    # === ANALYSIS ===
    print("\n=== PERFORMANCE ANALYSIS ===")

    if len(lgb_drift_metrics) == 0 or len(xgb_drift_metrics) == 0:
        print("\n⚠️ No drift batches evaluated (test set too small for batch_size=5000).")
        print("   Suggestion: Use larger test data for benchmark runs.")
        print("   Baseline metrics remain valid for reference.\n")
        return

    lgb_drift_pr = [m[0] for m in lgb_drift_metrics]
    xgb_drift_pr = [m[0] for m in xgb_drift_metrics]

    lgb_avg = np.mean(lgb_drift_pr)
    xgb_avg = np.mean(xgb_drift_pr)

    print(f"\nLightGBM:")
    print(f"  Baseline PR-AUC: {baseline_lgb:.4f}")
    print(f"  Drift PR-AUC (avg): {lgb_avg:.4f}")
    print(f"  Degradation: {((baseline_lgb - lgb_avg) / baseline_lgb * 100):.1f}%")
    print(f"  Range: [{min(lgb_drift_pr):.4f}, {max(lgb_drift_pr):.4f}]")

    print(f"\nXGBoost:")
    print(f"  Baseline PR-AUC: {baseline_xgb:.4f}")
    print(f"  Drift PR-AUC (avg): {xgb_avg:.4f}")
    print(f"  Degradation: {((baseline_xgb - xgb_avg) / baseline_xgb * 100):.1f}%")
    print(f"  Range: [{min(xgb_drift_pr):.4f}, {max(xgb_drift_pr):.4f}]")

    # === PLOT RESULTS ===
    plt.figure(figsize=(12, 6))
    batches = list(range(1, len(lgb_drift_pr) + 1))

    plt.plot(batches, lgb_drift_pr, 'o-', label='LightGBM', linewidth=2)
    plt.plot(batches, xgb_drift_pr, 's-', label='XGBoost', linewidth=2)
    plt.axhline(y=baseline_lgb, color='blue', linestyle='--', alpha=0.5, label='LightGBM Baseline')
    plt.axhline(y=baseline_xgb, color='orange', linestyle='--', alpha=0.5, label='XGBoost Baseline')

    plt.xlabel('Batch Number (After Drift)', fontsize=12)
    plt.ylabel('PR-AUC', fontsize=12)
    plt.title('Model Performance Under Sudden Drift\n(No Adaptation)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('drift_comparison_static.png', dpi=300)
    print("\n=== Plot saved to drift_comparison_static.png ===")

    # === TRAIN PKBOOST (ADVERSARIAL LIVING BOOSTER) ===
    print("\n=== Training PKBoost (Adversarial Living Booster) ===")
    print("This may take 5-10 minutes on 3.8M samples...\n")
    import subprocess
    import os
    env = os.environ.copy()
    env['DRIFT_DATASET'] = 'synthetic_financial'
    result = subprocess.run(
        ['cargo', 'run', '--release', '--bin', 'test_drift'],
        env=env,
        cwd=str(project_root)
    )
    print("\nPKBoost training complete.")
    
    # Parse PKBoost results
    try:
        alb_csv_path = project_root / 'alb_metrics.csv'
        alb_df = pd.read_csv(alb_csv_path)
        normal_data = alb_df[alb_df['phase'] == 'normal']
        drift_data = alb_df[alb_df['phase'] == 'drift']
        
        pkboost_baseline = normal_data['pr_auc'].mean() if len(normal_data) > 0 else 0.0
        pkboost_drift_pr = drift_data['pr_auc'].tolist() if len(drift_data) > 0 else []
        
        if pkboost_baseline > 0 and len(pkboost_drift_pr) > 0:
            pkboost_avg = np.mean(pkboost_drift_pr)
            pkboost_degradation = ((pkboost_baseline - pkboost_avg) / pkboost_baseline * 100)
            
            print(f"\nPKBoost:")
            print(f"  Baseline PR-AUC: {pkboost_baseline:.4f}")
            print(f"  Drift PR-AUC (avg): {pkboost_avg:.4f}")
            print(f"  Degradation: {pkboost_degradation:.1f}%")
            print(f"  Range: [{min(pkboost_drift_pr):.4f}, {max(pkboost_drift_pr):.4f}]")
            
            # Update plot with PKBoost
            plt.figure(figsize=(12, 6))
            plt.plot(batches, lgb_drift_pr, 'o-', label='LightGBM', linewidth=2)
            plt.plot(batches, xgb_drift_pr, 's-', label='XGBoost', linewidth=2)
            pkb_batches = list(range(1, len(pkboost_drift_pr) + 1))
            plt.plot(pkb_batches, pkboost_drift_pr, '^-', label='PKBoost (Adaptive)', linewidth=2.5, markersize=7)
            plt.axhline(y=baseline_lgb, color='blue', linestyle='--', alpha=0.3)
            plt.axhline(y=baseline_xgb, color='orange', linestyle='--', alpha=0.3)
            plt.axhline(y=pkboost_baseline, color='green', linestyle='--', alpha=0.3)
            plt.xlabel('Batch Number', fontsize=12)
            plt.ylabel('PR-AUC', fontsize=12)
            plt.title('Drift Comparison: Static vs Adaptive Models', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('drift_comparison_with_pkboost.png', dpi=300)
            print("\n=== Plot saved to drift_comparison_with_pkboost.png ===")
            
            print("\n=== FINAL COMPARISON ===")
            print(f"LightGBM: {((baseline_lgb - lgb_avg) / baseline_lgb * 100):.1f}% degradation")
            print(f"XGBoost:  {((baseline_xgb - xgb_avg) / baseline_xgb * 100):.1f}% degradation")
            print(f"PKBoost:  {pkboost_degradation:.1f}% degradation")
            print(f"PKBoost:  {pkboost_degradation:.1f}% degradation")
    except Exception as e:
        print(f"\nWarning: Could not parse PKBoost results: {e}")

    print("\n=== END OF COMPARISON ===\n")

if __name__ == "__main__":
    main()