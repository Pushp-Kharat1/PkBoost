"""
Three-way comparison with PKBoost ADAPTIVE features
Tests P1 (Loss), P2 (Sensitivity), P3 (Scoring) + Metamorphosis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import subprocess
import time

def generate_data(n, start_idx, coef, intercept, noise_level):
    x_vals = np.arange(start_idx, start_idx + n) / 100.0
    X = np.column_stack([x_vals, x_vals * 2.0, np.sin(x_vals * 3.0)])
    noise = np.array([i % 10 for i in range(n)]) * noise_level
    y = coef * x_vals + intercept + noise
    return X, y

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def test_pkboost_adaptive():
    """Run PKBoost with adaptive features"""
    print("\n" + "="*60)
    print("TESTING PKBOOST (ADAPTIVE)")
    print("="*60)
    
    try:
        result = subprocess.run(
            ['cargo', 'run', '--release', '--bin', 'pkboost_drift_benchmark'],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        output = result.stdout
        lines = output.split('\n')
        
        results = {}
        train_time = 0.0
        train_rmse = 0.0
        metamorphoses = 0
        
        for line in lines:
            if line.startswith('TRAIN_TIME:'):
                train_time = float(line.split(':')[1])
            elif line.startswith('TRAIN_RMSE:'):
                train_rmse = float(line.split(':')[1])
            elif line.startswith('SCENARIO:'):
                parts = line.split(':')
                scenario = parts[1].replace('_', ' ')
                rmse_val = float(parts[3])
                results[scenario] = rmse_val
            elif line.startswith('METAMORPHOSES:'):
                metamorphoses = int(line.split(':')[1])
        
        print(f"Training complete in {train_time:.2f}s, Train RMSE: {train_rmse:.4f}")
        print(f"Metamorphoses triggered: {metamorphoses}")
        
        return results, train_time, train_rmse, metamorphoses
        
    except Exception as e:
        print(f"Error running PKBoost: {e}")
        return {}, 0.0, 0.0, 0

def main():
    print("\n" + "="*70)
    print("   ADAPTIVE COMPARISON: PKBoost vs XGBoost vs LightGBM")
    print("   PKBoost with P1 (Loss) + P2 (Sensitivity) + P3 (Scoring)")
    print("="*70 + "\n")
    
    # Generate training data
    print("Generating training data (5,000 samples)...")
    X_train, y_train = generate_data(5000, 0, 2.0, 5.0, 0.1)
    print(f"  Y range: [{y_train.min():.1f}, {y_train.max():.1f}]")
    
    # Generate test scenarios
    print("\nGenerating test scenarios...")
    scenarios = [
        ("1 Stable", 1000, 5000, 2.0, 5.0, 0.1, False, False),
        ("2 Sudden Drift", 1000, 6000, 4.0, 15.0, 0.1, False, False),
        ("3 Gradual Drift", 1000, 7000, 6.0, 15.0, 0.1, False, False),
        ("4 Outliers", 1000, 8000, 6.0, 15.0, 0.1, True, False),
        ("5 High Noise", 1000, 9000, 6.0, 15.0, 5.0, False, False),
        ("6 Temporal", 1000, 10000, 6.0, 15.0, 0.1, False, True),
        ("7 Reversal", 1000, 11000, -6.0, 100.0, 0.1, False, False),
    ]
    
    test_data = []
    for name, n, start_idx, coef, intercept, noise, add_outliers, add_temporal in scenarios:
        X, y = generate_data(n, start_idx, coef, intercept, noise)
        if add_outliers:
            outlier_mask = np.arange(len(y)) % 20 == 0
            y[outlier_mask] += 200.0
        if add_temporal:
            y += np.arange(len(y)) * 0.02
        test_data.append((name, X, y))
    
    print(f"Generated {len(test_data)} scenarios\n")
    
    # Train XGBoost (STATIC)
    print("="*60)
    print("TRAINING XGBOOST (STATIC - No Adaptation)")
    print("="*60)
    
    xgb_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42, tree_method='hist')
    start = time.time()
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_time = time.time() - start
    
    train_pred = xgb_model.predict(X_train)
    xgb_train_rmse = rmse(y_train, train_pred)
    print(f"Training complete in {xgb_time:.2f}s, Train RMSE: {xgb_train_rmse:.4f}")
    
    xgb_results = {}
    for name, X_test, y_test in test_data:
        pred = xgb_model.predict(X_test)
        xgb_results[name] = rmse(y_test, pred)
    
    # Train LightGBM (STATIC)
    print("\n" + "="*60)
    print("TRAINING LIGHTGBM (STATIC - No Adaptation)")
    print("="*60)
    
    lgb_model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
    start = time.time()
    lgb_model.fit(X_train, y_train)
    lgb_time = time.time() - start
    
    train_pred = lgb_model.predict(X_train)
    lgb_train_rmse = rmse(y_train, train_pred)
    print(f"Training complete in {lgb_time:.2f}s, Train RMSE: {lgb_train_rmse:.4f}")
    
    lgb_results = {}
    for name, X_test, y_test in test_data:
        pred = lgb_model.predict(X_test)
        lgb_results[name] = rmse(y_test, pred)
    
    # Train PKBoost (ADAPTIVE)
    pkb_results, pkb_time, pkb_train_rmse, metamorphoses = test_pkboost_adaptive()
    
    # Create comparison table
    print("\n" + "="*80)
    print("FINAL COMPARISON - RMSE")
    print("="*80)
    
    comparison = pd.DataFrame({
        'Scenario': [name for name, _, _ in test_data],
        'XGBoost': [xgb_results[name] for name, _, _ in test_data],
        'LightGBM': [lgb_results[name] for name, _, _ in test_data],
        'PKBoost': [pkb_results.get(name, 0.0) for name, _, _ in test_data]
    })
    
    # Calculate degradation
    baseline_xgb = comparison.iloc[0]['XGBoost']
    baseline_lgb = comparison.iloc[0]['LightGBM']
    baseline_pkb = comparison.iloc[0]['PKBoost']
    
    comparison['XGB_Deg%'] = ((comparison['XGBoost'] - baseline_xgb) / baseline_xgb * 100).round(1)
    comparison['LGBM_Deg%'] = ((comparison['LightGBM'] - baseline_lgb) / baseline_lgb * 100).round(1)
    comparison['PKB_Deg%'] = ((comparison['PKBoost'] - baseline_pkb) / baseline_pkb * 100).round(1)
    
    print("\n" + comparison.to_string(index=False))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nTraining Time:")
    print(f"  XGBoost:  {xgb_time:.2f}s")
    print(f"  LightGBM: {lgb_time:.2f}s")
    print(f"  PKBoost:  {pkb_time:.2f}s")
    
    print(f"\nTrain RMSE:")
    print(f"  XGBoost:  {xgb_train_rmse:.4f}")
    print(f"  LightGBM: {lgb_train_rmse:.4f}")
    print(f"  PKBoost:  {pkb_train_rmse:.4f}")
    
    print(f"\nAverage Test RMSE:")
    print(f"  XGBoost:  {comparison['XGBoost'].mean():.2f}")
    print(f"  LightGBM: {comparison['LightGBM'].mean():.2f}")
    print(f"  PKBoost:  {comparison['PKBoost'].mean():.2f}")
    
    print(f"\nAverage Degradation (excluding baseline):")
    print(f"  XGBoost:  {comparison.iloc[1:]['XGB_Deg%'].mean():.1f}%")
    print(f"  LightGBM: {comparison.iloc[1:]['LGBM_Deg%'].mean():.1f}%")
    print(f"  PKBoost:  {comparison.iloc[1:]['PKB_Deg%'].mean():.1f}%")
    
    print(f"\nWorst Case Degradation:")
    print(f"  XGBoost:  {comparison['XGB_Deg%'].max():.1f}%")
    print(f"  LightGBM: {comparison['LGBM_Deg%'].max():.1f}%")
    print(f"  PKBoost:  {comparison['PKB_Deg%'].max():.1f}%")
    
    print(f"\nPKBoost Metamorphoses: {metamorphoses}")
    
    # Winner analysis
    print("\n" + "="*80)
    print("WINNER BY SCENARIO")
    print("="*80)
    
    for idx, row in comparison.iterrows():
        scenario = row['Scenario']
        winner = row[['XGBoost', 'LightGBM', 'PKBoost']].idxmin()
        best_rmse = row[['XGBoost', 'LightGBM', 'PKBoost']].min()
        print(f"{scenario}: {winner} (RMSE: {best_rmse:.2f})")
    
    # Overall winner
    avg_rmse = {
        'XGBoost': comparison['XGBoost'].mean(),
        'LightGBM': comparison['LightGBM'].mean(),
        'PKBoost': comparison['PKBoost'].mean()
    }
    overall_winner = min(avg_rmse, key=avg_rmse.get)
    
    print(f"\nOVERALL WINNER: {overall_winner}")
    print(f"  Average RMSE: {avg_rmse[overall_winner]:.2f}")
    
    # Save results
    comparison.to_csv('adaptive_comparison_results.csv', index=False)
    print(f"\nResults saved to adaptive_comparison_results.csv")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    print("- XGBoost & LightGBM: STATIC models, no adaptation")
    print("- PKBoost: ADAPTIVE with P1+P2+P3 + Metamorphosis")
    print(f"- PKBoost triggered {metamorphoses} metamorphoses to adapt to drift")
    print("- Lower degradation = better drift handling")
    print("="*80)

if __name__ == "__main__":
    main()
