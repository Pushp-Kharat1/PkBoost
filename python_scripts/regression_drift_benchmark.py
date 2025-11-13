#!/usr/bin/env python3
"""Regression drift benchmark - test PKBoost vs static models under distribution shift"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import subprocess
import os

def apply_drift(X, drift_type, severity=1.0):
    """Apply various drift types to regression data"""
    X_drift = X.copy()
    n_features = X.shape[1]
    
    if drift_type == "covariate_noise":
        noise = np.random.normal(0, severity * 0.5, X.shape)
        X_drift += noise
        
    elif drift_type == "feature_scaling":
        scale_features = np.random.choice(n_features, size=n_features//2, replace=False)
        X_drift[:, scale_features] *= (1 + severity)
        
    elif drift_type == "feature_shift":
        shift_features = np.random.choice(n_features, size=n_features//2, replace=False)
        X_drift[:, shift_features] += severity
        
    elif drift_type == "rotation":
        angle = severity * 0.1
        for i in range(0, n_features-1, 2):
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x1, x2 = X_drift[:, i].copy(), X_drift[:, i+1].copy()
            X_drift[:, i] = cos_a * x1 - sin_a * x2
            X_drift[:, i+1] = sin_a * x1 + cos_a * x2
            
    elif drift_type == "outliers":
        n_outliers = int(len(X) * severity * 0.1)
        outlier_idx = np.random.choice(len(X), n_outliers, replace=False)
        X_drift[outlier_idx] += np.random.normal(0, 5, (n_outliers, n_features))
        
    return X_drift

def prepare_drift_data():
    """Prepare California Housing with various drift scenarios"""
    print("Loading California Housing dataset...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    X = df.drop('MedHouseVal', axis=1).values
    y = df['MedHouseVal'].values
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Save baseline
    os.makedirs('data', exist_ok=True)
    
    scenarios = [
        ("baseline", X_test, 0),
        ("mild_noise", apply_drift(X_test, "covariate_noise", 0.5), 0.5),
        ("moderate_noise", apply_drift(X_test, "covariate_noise", 1.0), 1.0),
        ("severe_noise", apply_drift(X_test, "covariate_noise", 2.0), 2.0),
        ("feature_scaling", apply_drift(X_test, "feature_scaling", 1.0), 1.0),
        ("feature_shift", apply_drift(X_test, "feature_shift", 1.0), 1.0),
        ("rotation", apply_drift(X_test, "rotation", 1.0), 1.0),
        ("outliers", apply_drift(X_test, "outliers", 1.0), 1.0),
    ]
    
    # Save train/val once
    train_df = pd.DataFrame(X_train, columns=data.feature_names)
    train_df['Target'] = y_train
    train_df.to_csv('data/housing_train.csv', index=False)
    
    val_df = pd.DataFrame(X_val, columns=data.feature_names)
    val_df['Target'] = y_val
    val_df.to_csv('data/housing_val.csv', index=False)
    
    results = []
    
    for scenario_name, X_test_drift, severity in scenarios:
        test_df = pd.DataFrame(X_test_drift, columns=data.feature_names)
        test_df['Target'] = y_test
        test_df.to_csv('data/housing_test.csv', index=False)
        
        print(f"\n=== Testing: {scenario_name} (severity={severity}) ===")
        
        # Run PKBoost
        result = subprocess.run(
            ['cargo', 'run', '--release', '--bin', 'test_regression'],
            capture_output=True, text=True, cwd=os.getcwd()
        )
        
        if result.returncode != 0:
            print(f"  ERROR: cargo failed with code {result.returncode}")
            print(f"  stderr: {result.stderr[:500]}")
            continue
        
        # Debug: print first 500 chars of output
        # print(f"  DEBUG stdout: {result.stdout[:500]}")
        # print(f"  DEBUG stderr: {result.stderr[:500]}")
        
        # Parse output - look for Test Set Results section
        output_lines = result.stdout + result.stderr
        lines = output_lines.split('\n')
        found = False
        
        # Find the Test Set Results section
        for i, line in enumerate(lines):
            if '=== Test Set Results ===' in line:
                # Next 3 lines should have RMSE, MAE, R²
                try:
                    rmse_line = lines[i+1] if i+1 < len(lines) else ''
                    mae_line = lines[i+2] if i+2 < len(lines) else ''
                    r2_line = lines[i+3] if i+3 < len(lines) else ''
                    
                    rmse = float(rmse_line.split(':')[1].strip())
                    mae = float(mae_line.split(':')[1].strip())
                    r2 = float(r2_line.split(':')[1].strip())
                    
                    results.append({
                        'scenario': scenario_name,
                        'severity': severity,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2
                    })
                    print(f"  ✓ RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
                    found = True
                    break
                except (ValueError, IndexError) as e:
                    print(f"  Parse error: {e}")
                    continue
        
        if not found:
            print(f"  ✗ Could not parse output")
    
    # Save results
    print(f"\nCollected {len(results)} results")
    
    if not results:
        print("ERROR: No results collected.")
        print("Try running manually: cargo run --release --bin test_regression")
        return
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('regression_drift_results.csv', index=False)
    
    print("\n=== SUMMARY ===")
    print(results_df.to_string(index=False))
    
    if 'baseline' in results_df['scenario'].values:
        baseline_rmse = results_df[results_df['scenario'] == 'baseline']['rmse'].values[0]
        results_df['degradation_%'] = ((results_df['rmse'] - baseline_rmse) / baseline_rmse * 100).round(2)
        
        print(f"\nBaseline RMSE: {baseline_rmse:.4f}")
        print(f"Average degradation: {results_df[results_df['scenario'] != 'baseline']['degradation_%'].mean():.2f}%")
        print(f"Worst case: {results_df['degradation_%'].max():.2f}%")
    else:
        print("\nWARNING: Baseline scenario not found in results.")

if __name__ == "__main__":
    prepare_drift_data()
