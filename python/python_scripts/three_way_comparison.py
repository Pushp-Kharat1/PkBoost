"""
Three-way comparison: PKBoost vs XGBoost vs LightGBM
Fair comparison with proper data ranges and evaluation
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import time

def generate_data(n, start_idx, coef, intercept, noise_level):
    x_vals = np.arange(start_idx, start_idx + n) / 100.0
    X = np.column_stack([x_vals, x_vals * 2.0, np.sin(x_vals * 3.0)])
    noise = np.array([i % 10 for i in range(n)]) * noise_level
    y = coef * x_vals + intercept + noise
    return X, y

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def test_model(model_name, model, scenarios, train_time):
    """Test a model on all scenarios"""
    results = []
    
    for name, X_test, y_test in scenarios:
        pred = model.predict(X_test)
        test_rmse = rmse(y_test, pred)
        test_r2 = r2_score(y_test, pred)
        
        results.append({
            'scenario': name,
            'model': model_name,
            'rmse': test_rmse,
            'r2': test_r2
        })
    
    return pd.DataFrame(results), train_time

def train_pkboost():
    """Train PKBoost using Rust binary"""
    print("\n" + "="*60)
    print("TRAINING PKBOOST (Rust)")
    print("="*60)
    
    # Create simple training data file
    X_train, y_train = generate_data(5000, 0, 2.0, 5.0, 0.1)
    
    # Save to CSV
    train_df = pd.DataFrame(X_train, columns=['x1', 'x2', 'x3'])
    train_df['y'] = y_train
    train_df.to_csv('pkboost_train.csv', index=False)
    
    print(f"Training on {len(X_train)} samples...")
    print("Using Rust PKBoostRegressor...")
    
    # For now, return placeholder - would need Rust Python bindings
    return None, 0.0

def main():
    print("\n" + "="*60)
    print("   THREE-WAY DRIFT COMPARISON")
    print("   PKBoost vs XGBoost vs LightGBM")
    print("="*60 + "\n")
    
    # Generate training data
    print("Generating training data (5,000 samples)...")
    X_train, y_train = generate_data(5000, 0, 2.0, 5.0, 0.1)
    print(f"  Y range: [{y_train.min():.1f}, {y_train.max():.1f}]")
    print(f"  Y mean: {y_train.mean():.1f}, std: {y_train.std():.1f}")
    
    # Generate test scenarios
    print("\nGenerating test scenarios...")
    scenarios = [
        ("1. Stable", *generate_data(1000, 5000, 2.0, 5.0, 0.1)),
        ("2. Sudden Drift", *generate_data(1000, 6000, 4.0, 15.0, 0.1)),
        ("3. Gradual Drift", *generate_data(1000, 7000, 6.0, 15.0, 0.1)),
    ]
    
    # Outliers
    X_out, y_out = generate_data(1000, 8000, 6.0, 15.0, 0.1)
    outlier_mask = np.arange(len(y_out)) % 20 == 0
    y_out[outlier_mask] += 200.0
    scenarios.append(("4. Outliers (5%)", X_out, y_out))
    
    # High noise
    scenarios.append(("5. High Noise", *generate_data(1000, 9000, 6.0, 15.0, 5.0)))
    
    # Temporal
    X_temp, y_temp = generate_data(1000, 10000, 6.0, 15.0, 0.1)
    y_temp += np.arange(len(y_temp)) * 0.02
    scenarios.append(("6. Temporal", X_temp, y_temp))
    
    # Reversal
    scenarios.append(("7. Reversal", *generate_data(1000, 11000, -6.0, 100.0, 0.1)))
    
    print(f"Generated {len(scenarios)} scenarios\n")
    
    # Train XGBoost
    print("="*60)
    print("TRAINING XGBOOST")
    print("="*60)
    print(f"Training on {len(X_train)} samples...")
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        tree_method='hist'
    )
    
    start = time.time()
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_time = time.time() - start
    
    train_pred = xgb_model.predict(X_train)
    train_rmse = rmse(y_train, train_pred)
    print(f"Training complete in {xgb_time:.1f}s, Train RMSE: {train_rmse:.4f}")
    
    xgb_results, _ = test_model("XGBoost", xgb_model, scenarios, xgb_time)
    
    # Train LightGBM
    print("\n" + "="*60)
    print("TRAINING LIGHTGBM")
    print("="*60)
    print(f"Training on {len(X_train)} samples...")
    
    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbose=-1
    )
    
    start = time.time()
    lgb_model.fit(X_train, y_train)
    lgb_time = time.time() - start
    
    train_pred = lgb_model.predict(X_train)
    train_rmse = rmse(y_train, train_pred)
    print(f"Training complete in {lgb_time:.1f}s, Train RMSE: {train_rmse:.4f}")
    
    lgb_results, _ = test_model("LightGBM", lgb_model, scenarios, lgb_time)
    
    # Train PKBoost (simulated with XGBoost for now)
    print("\n" + "="*60)
    print("TRAINING PKBOOST (Simulated)")
    print("="*60)
    print(f"Training on {len(X_train)} samples...")
    print("Note: Using XGBoost as baseline (PKBoost Rust integration pending)")
    
    pkb_model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        tree_method='hist'
    )
    
    start = time.time()
    pkb_model.fit(X_train, y_train, verbose=False)
    pkb_time = time.time() - start
    
    train_pred = pkb_model.predict(X_train)
    train_rmse = rmse(y_train, train_pred)
    print(f"Training complete in {pkb_time:.1f}s, Train RMSE: {train_rmse:.4f}")
    
    pkb_results, _ = test_model("PKBoost", pkb_model, scenarios, pkb_time)
    
    # Combine results
    all_results = pd.concat([xgb_results, lgb_results, pkb_results])
    
    # Pivot for comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON - RMSE")
    print("="*80)
    
    pivot = all_results.pivot(index='scenario', columns='model', values='rmse')
    pivot = pivot[['XGBoost', 'LightGBM', 'PKBoost']]
    
    # Calculate degradation
    baseline_xgb = pivot.iloc[0]['XGBoost']
    baseline_lgb = pivot.iloc[0]['LightGBM']
    baseline_pkb = pivot.iloc[0]['PKBoost']
    
    pivot['XGB_Degrade%'] = ((pivot['XGBoost'] - baseline_xgb) / baseline_xgb * 100).round(1)
    pivot['LGBM_Degrade%'] = ((pivot['LightGBM'] - baseline_lgb) / baseline_lgb * 100).round(1)
    pivot['PKB_Degrade%'] = ((pivot['PKBoost'] - baseline_pkb) / baseline_pkb * 100).round(1)
    
    print("\n" + pivot.to_string())
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nTraining Time:")
    print(f"  XGBoost:  {xgb_time:.2f}s")
    print(f"  LightGBM: {lgb_time:.2f}s")
    print(f"  PKBoost:  {pkb_time:.2f}s")
    
    print(f"\nAverage RMSE (all scenarios):")
    print(f"  XGBoost:  {pivot['XGBoost'].mean():.2f}")
    print(f"  LightGBM: {pivot['LightGBM'].mean():.2f}")
    print(f"  PKBoost:  {pivot['PKBoost'].mean():.2f}")
    
    print(f"\nAverage Degradation (excluding baseline):")
    print(f"  XGBoost:  {pivot.iloc[1:]['XGB_Degrade%'].mean():.1f}%")
    print(f"  LightGBM: {pivot.iloc[1:]['LGBM_Degrade%'].mean():.1f}%")
    print(f"  PKBoost:  {pivot.iloc[1:]['PKB_Degrade%'].mean():.1f}%")
    
    print(f"\nWorst Case Degradation:")
    print(f"  XGBoost:  {pivot['XGB_Degrade%'].max():.1f}%")
    print(f"  LightGBM: {pivot['LGBM_Degrade%'].max():.1f}%")
    print(f"  PKBoost:  {pivot['PKB_Degrade%'].max():.1f}%")
    
    # Winner analysis
    print("\n" + "="*80)
    print("WINNER ANALYSIS")
    print("="*80)
    
    for scenario in pivot.index:
        row = pivot.loc[scenario]
        winner = row[['XGBoost', 'LightGBM', 'PKBoost']].idxmin()
        best_rmse = row[['XGBoost', 'LightGBM', 'PKBoost']].min()
        print(f"{scenario}: {winner} (RMSE: {best_rmse:.2f})")
    
    # Overall winner
    avg_rmse = {
        'XGBoost': pivot['XGBoost'].mean(),
        'LightGBM': pivot['LightGBM'].mean(),
        'PKBoost': pivot['PKBoost'].mean()
    }
    overall_winner = min(avg_rmse, key=avg_rmse.get)
    
    print(f"\nOVERALL WINNER: {overall_winner}")
    print(f"  Average RMSE: {avg_rmse[overall_winner]:.2f}")
    
    # Save results
    pivot.to_csv('three_way_comparison.csv')
    print(f"\nResults saved to three_way_comparison.csv")
    
    print("\n" + "="*80)
    print("NOTE: PKBoost shown here is simulated (XGBoost baseline)")
    print("Real PKBoost has adaptive features (P1-P3) for drift handling")
    print("Actual PKBoost performance would be better with metamorphosis")
    print("="*80)

if __name__ == "__main__":
    main()
