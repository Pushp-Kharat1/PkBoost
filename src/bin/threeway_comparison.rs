use pkboost::*;
use std::time::Instant;
use std::error::Error;
use std::process::Command;
use std::fs;

fn load_csv(path: &str) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let target_idx = headers.iter().position(|h| h == "Class").ok_or("Class column not found")?;
    let mut features = Vec::new();
    let mut labels = Vec::new();
    for result in reader.records() {
        let record = result?;
        let mut row = Vec::new();
        for (i, value) in record.iter().enumerate() {
            if i == target_idx {
                labels.push(value.parse()?);
            } else {
                row.push(if value.is_empty() { f64::NAN } else { value.parse()? });
            }
        }
        features.push(row);
    }
    Ok((features, labels))
}

fn add_drift(x: &mut [Vec<f64>], drift_features: &[usize], noise_std: f64) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    for sample in x.iter_mut() {
        for &feat_idx in drift_features {
            if feat_idx < sample.len() {
                sample[feat_idx] += rng.gen::<f64>() * noise_std - noise_std / 2.0;
            }
        }
    }
}

fn save_for_python(x: &[Vec<f64>], y: &[f64], path: &str) -> Result<(), Box<dyn Error>> {
    let mut wtr = csv::Writer::from_path(path)?;
    let n_features = x[0].len();
    let mut header: Vec<String> = (0..n_features).map(|i| format!("V{}", i+1)).collect();
    header.push("Class".to_string());
    wtr.write_record(&header)?;
    for (features, &label) in x.iter().zip(y.iter()) {
        let mut record: Vec<String> = features.iter().map(|v| v.to_string()).collect();
        record.push(label.to_string());
        wtr.write_record(&record)?;
    }
    wtr.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║     THREE-WAY COMPARISON: PKBoost vs XGBoost vs LightGBM     ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Load data
    println!("[1/6] Loading Credit Card dataset...");
    let (x_train, y_train) = load_csv("data/creditcard_train.csv")?;
    let (x_val, y_val) = load_csv("data/creditcard_val.csv")?;
    let (mut x_test, y_test) = load_csv("data/creditcard_test.csv")?;
    println!("  ✓ Train: {} samples", x_train.len());
    println!("  ✓ Val:   {} samples", x_val.len());
    println!("  ✓ Test:  {} samples\n", x_test.len());

    // ═══════════════════════════════════════════════════════════════
    // BASELINE COMPARISON (No Drift)
    // ═══════════════════════════════════════════════════════════════
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    BASELINE (No Drift)                        ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // PKBoost Baseline
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ PKBoost v2.0.1 (with Progressive Precision)                │");
    println!("└─────────────────────────────────────────────────────────────┘");
    let start = Instant::now();
    let mut pkboost = OptimizedPKBoostShannon::auto(&x_train, &y_train);
    pkboost.n_estimators = 100;
    pkboost.fit(&x_train, &y_train, Some((&x_val, &y_val)), false)?;
    let pkboost_train_time = start.elapsed();
    
    let start = Instant::now();
    let pkboost_probs = pkboost.predict_proba(&x_test)?;
    let pkboost_pred_time = start.elapsed();
    
    let pkboost_pr_auc = calculate_pr_auc(&y_test, &pkboost_probs);
    let pkboost_roc_auc = calculate_roc_auc(&y_test, &pkboost_probs);
    
    println!("  Training time:   {:?}", pkboost_train_time);
    println!("  Prediction time: {:?}", pkboost_pred_time);
    println!("  PR-AUC:          {:.4}", pkboost_pr_auc);
    println!("  ROC-AUC:         {:.4}\n", pkboost_roc_auc);

    // Save data for Python models
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Preparing data for XGBoost & LightGBM (Python)             │");
    println!("└─────────────────────────────────────────────────────────────┘");
    fs::create_dir_all("temp")?;
    save_for_python(&x_train, &y_train, "temp/train.csv")?;
    save_for_python(&x_val, &y_val, "temp/val.csv")?;
    save_for_python(&x_test, &y_test, "temp/test.csv")?;
    println!("  ✓ Data saved to temp/ directory\n");

    // Create Python comparison script
    let python_script = r#"
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import warnings
warnings.filterwarnings('ignore')

# Load data
train = pd.read_csv('temp/train.csv')
val = pd.read_csv('temp/val.csv')
test = pd.read_csv('temp/test.csv')

X_train = train.drop('Class', axis=1)
y_train = train['Class']
X_val = val.drop('Class', axis=1)
y_val = val['Class']
X_test = test.drop('Class', axis=1)
y_test = test['Class']

print("┌─────────────────────────────────────────────────────────────┐")
print("│ XGBoost (Default Parameters)                                │")
print("└─────────────────────────────────────────────────────────────┘")

start = time.time()
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_train_time = time.time() - start

start = time.time()
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_pred_time = time.time() - start

xgb_pr_auc = average_precision_score(y_test, xgb_probs)
xgb_roc_auc = roc_auc_score(y_test, xgb_probs)

print(f"  Training time:   {xgb_train_time:.2f}s")
print(f"  Prediction time: {xgb_pred_time:.4f}s")
print(f"  PR-AUC:          {xgb_pr_auc:.4f}")
print(f"  ROC-AUC:         {xgb_roc_auc:.4f}\n")

print("┌─────────────────────────────────────────────────────────────┐")
print("│ LightGBM (Default Parameters)                               │")
print("└─────────────────────────────────────────────────────────────┘")

start = time.time()
lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
lgb_train_time = time.time() - start

start = time.time()
lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
lgb_pred_time = time.time() - start

lgb_pr_auc = average_precision_score(y_test, lgb_probs)
lgb_roc_auc = roc_auc_score(y_test, lgb_probs)

print(f"  Training time:   {lgb_train_time:.2f}s")
print(f"  Prediction time: {lgb_pred_time:.4f}s")
print(f"  PR-AUC:          {lgb_pr_auc:.4f}")
print(f"  ROC-AUC:         {lgb_roc_auc:.4f}\n")

# Save results
with open('temp/baseline_results.txt', 'w') as f:
    f.write(f"XGBoost,{xgb_train_time:.2f},{xgb_pred_time:.4f},{xgb_pr_auc:.4f},{xgb_roc_auc:.4f}\n")
    f.write(f"LightGBM,{lgb_train_time:.2f},{lgb_pred_time:.4f},{lgb_pr_auc:.4f},{lgb_roc_auc:.4f}\n")
"#;
    fs::write("temp/compare_baseline.py", python_script)?;

    println!("  Running Python models...");
    let output = Command::new("python")
        .arg("temp/compare_baseline.py")
        .output();

    match output {
        Ok(out) => {
            println!("{}", String::from_utf8_lossy(&out.stdout));
            if !out.stderr.is_empty() {
                eprintln!("  Warnings: {}", String::from_utf8_lossy(&out.stderr));
            }
        }
        Err(e) => println!("  ⚠ Python not available or error: {}\n", e),
    }

    // ═══════════════════════════════════════════════════════════════
    // DRIFT COMPARISON
    // ═══════════════════════════════════════════════════════════════
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                  DRIFT SCENARIO (10 Features)                 ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Add drift to test set
    let drift_features: Vec<usize> = (0..10).collect();
    add_drift(&mut x_test, &drift_features, 2.0);
    println!("  ✓ Added drift to {} features (noise_std=2.0)\n", drift_features.len());

    // PKBoost under drift
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ PKBoost v2.0.1 (Under Drift)                               │");
    println!("└─────────────────────────────────────────────────────────────┘");
    let start = Instant::now();
    let pkboost_drift_probs = pkboost.predict_proba(&x_test)?;
    let pkboost_drift_pred_time = start.elapsed();
    
    let pkboost_drift_pr_auc = calculate_pr_auc(&y_test, &pkboost_drift_probs);
    let pkboost_drift_roc_auc = calculate_roc_auc(&y_test, &pkboost_drift_probs);
    let pkboost_degradation = ((pkboost_pr_auc - pkboost_drift_pr_auc) / pkboost_pr_auc * 100.0).abs();
    
    println!("  Prediction time: {:?}", pkboost_drift_pred_time);
    println!("  PR-AUC:          {:.4}", pkboost_drift_pr_auc);
    println!("  ROC-AUC:         {:.4}", pkboost_drift_roc_auc);
    println!("  Degradation:     {:.2}%\n", pkboost_degradation);

    // Save drifted data for Python
    save_for_python(&x_test, &y_test, "temp/test_drift.csv")?;

    let python_drift_script = r#"
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import warnings
warnings.filterwarnings('ignore')

# Load baseline results
baseline = {}
with open('temp/baseline_results.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        baseline[parts[0]] = {'pr_auc': float(parts[3])}

# Load models and drifted test data
train = pd.read_csv('temp/train.csv')
val = pd.read_csv('temp/val.csv')
test_drift = pd.read_csv('temp/test_drift.csv')

X_train = train.drop('Class', axis=1)
y_train = train['Class']
X_val = val.drop('Class', axis=1)
y_val = val['Class']
X_test_drift = test_drift.drop('Class', axis=1)
y_test_drift = test_drift['Class']

# Retrain models (they were already trained)
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

print("┌─────────────────────────────────────────────────────────────┐")
print("│ XGBoost (Under Drift)                                       │")
print("└─────────────────────────────────────────────────────────────┘")

start = time.time()
xgb_drift_probs = xgb_model.predict_proba(X_test_drift)[:, 1]
xgb_drift_pred_time = time.time() - start

xgb_drift_pr_auc = average_precision_score(y_test_drift, xgb_drift_probs)
xgb_drift_roc_auc = roc_auc_score(y_test_drift, xgb_drift_probs)
xgb_degradation = abs((baseline['XGBoost']['pr_auc'] - xgb_drift_pr_auc) / baseline['XGBoost']['pr_auc'] * 100)

print(f"  Prediction time: {xgb_drift_pred_time:.4f}s")
print(f"  PR-AUC:          {xgb_drift_pr_auc:.4f}")
print(f"  ROC-AUC:         {xgb_drift_roc_auc:.4f}")
print(f"  Degradation:     {xgb_degradation:.2f}%\n")

print("┌─────────────────────────────────────────────────────────────┐")
print("│ LightGBM (Under Drift)                                      │")
print("└─────────────────────────────────────────────────────────────┘")

start = time.time()
lgb_drift_probs = lgb_model.predict_proba(X_test_drift)[:, 1]
lgb_drift_pred_time = time.time() - start

lgb_drift_pr_auc = average_precision_score(y_test_drift, lgb_drift_probs)
lgb_drift_roc_auc = roc_auc_score(y_test_drift, lgb_drift_probs)
lgb_degradation = abs((baseline['LightGBM']['pr_auc'] - lgb_drift_pr_auc) / baseline['LightGBM']['pr_auc'] * 100)

print(f"  Prediction time: {lgb_drift_pred_time:.4f}s")
print(f"  PR-AUC:          {lgb_drift_pr_auc:.4f}")
print(f"  ROC-AUC:         {lgb_drift_roc_auc:.4f}")
print(f"  Degradation:     {lgb_degradation:.2f}%\n")

# Save drift results
with open('temp/drift_results.txt', 'w') as f:
    f.write(f"XGBoost,{xgb_drift_pred_time:.4f},{xgb_drift_pr_auc:.4f},{xgb_drift_roc_auc:.4f},{xgb_degradation:.2f}\n")
    f.write(f"LightGBM,{lgb_drift_pred_time:.4f},{lgb_drift_pr_auc:.4f},{lgb_drift_roc_auc:.4f},{lgb_degradation:.2f}\n")
"#;
    fs::write("temp/compare_drift.py", python_drift_script)?;

    println!("  Running Python models under drift...");
    let output = Command::new("python")
        .arg("temp/compare_drift.py")
        .output();

    match output {
        Ok(out) => {
            println!("{}", String::from_utf8_lossy(&out.stdout));
        }
        Err(e) => println!("  ⚠ Python not available: {}\n", e),
    }

    // ═══════════════════════════════════════════════════════════════
    // SUMMARY TABLE
    // ═══════════════════════════════════════════════════════════════
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                      SUMMARY TABLE                            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("┌─────────────┬──────────────┬──────────────┬──────────────┐");
    println!("│   Model     │ Baseline     │ Under Drift  │ Degradation  │");
    println!("│             │ PR-AUC       │ PR-AUC       │ (%)          │");
    println!("├─────────────┼──────────────┼──────────────┼──────────────┤");
    println!("│ PKBoost     │ {:.4}       │ {:.4}       │ {:.2}%       │", 
        pkboost_pr_auc, pkboost_drift_pr_auc, pkboost_degradation);

    // Read Python results if available
    if let Ok(baseline_content) = fs::read_to_string("temp/baseline_results.txt") {
        if let Ok(drift_content) = fs::read_to_string("temp/drift_results.txt") {
            for (baseline_line, drift_line) in baseline_content.lines().zip(drift_content.lines()) {
                let baseline_parts: Vec<&str> = baseline_line.split(',').collect();
                let drift_parts: Vec<&str> = drift_line.split(',').collect();
                if baseline_parts.len() >= 4 && drift_parts.len() >= 4 {
                    println!("│ {:<11} │ {:<12} │ {:<12} │ {:<12} │",
                        baseline_parts[0], baseline_parts[3], drift_parts[2], 
                        format!("{}%", drift_parts[4]));
                }
            }
        }
    }
    println!("└─────────────┴──────────────┴──────────────┴──────────────┘\n");

    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║                         CONCLUSION                            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!("  ✓ PKBoost with progressive precision tested");
    println!("  ✓ Comparison with XGBoost and LightGBM complete");
    println!("  ✓ Drift resilience evaluated");
    println!("  ✓ Results saved to temp/ directory\n");

    Ok(())
}
