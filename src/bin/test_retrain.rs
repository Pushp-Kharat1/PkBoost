use mimalloc::MiMalloc;
use pkboost::*;
use std::error::Error;
use rand::prelude::*;
use std::time::Instant;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn load_data(path: &str) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let mut features = Vec::new();
    let mut labels = Vec::new();
    let target_col_index = headers.iter().position(|h| h == "Class")
        .ok_or("'Class' column not found")?;

    for result in reader.records() {    
        let record = result?;
        let mut feature_row = Vec::new();
        for (i, value) in record.iter().enumerate() {
            if i == target_col_index {
                labels.push(value.parse::<f64>()?);
            } else {
                let parsed_value = if value.is_empty() {
                    f64::NAN
                } else {
                    value.parse::<f64>()?
                };
                feature_row.push(parsed_value);
            }
        }
        features.push(feature_row);
    }
    Ok((features, labels))
}

fn introduce_drift(x: &mut Vec<Vec<f64>>, drift_features: &[usize], noise_level: f64) {
    let mut rng = rand::thread_rng();
    for row in x.iter_mut() {
        for &feat_idx in drift_features {
            if feat_idx < row.len() {
                row[feat_idx] += rng.gen_range(-noise_level..noise_level);
            }
        }
    }
}

fn evaluate_batch(
    model: &OptimizedPKBoostShannon,
    x_batch: &Vec<Vec<f64>>,
    y_batch: &Vec<f64>,
    observation_count: usize,
    phase: &str,
) -> Result<f64, String> {
    let predictions = model.predict_proba(x_batch)?;
    let pr_auc = calculate_pr_auc(y_batch, &predictions);
    let roc_auc = calculate_roc_auc(y_batch, &predictions);

    let f1 = {
        let mut tp = 0;
        let mut fp = 0;
        let mut fn_count = 0;
        for (i, &pred_prob) in predictions.iter().enumerate() {
            let pred_class = if pred_prob >= 0.5 { 1 } else { 0 };
            let actual = if y_batch[i] > 0.5 { 1 } else { 0 };
            match (pred_class, actual) {
                (1, 1) => tp += 1,
                (1, 0) => fp += 1,
                (0, 1) => fn_count += 1,
                _ => {}
            }
        }
        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if tp + fn_count > 0 { tp as f64 / (tp + fn_count) as f64 } else { 0.0 };
        if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 }
    };

    println!(
        "RETRAIN_MODEL_METRIC | Phase: {} | Obs: {} | PR-AUC: {:.4} | ROC-AUC: {:.4} | F1@0.5: {:.4}",
        phase, observation_count, pr_auc, roc_auc, f1
    );

    Ok(pr_auc)
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("\n=== RETRAIN BASELINE DRIFT SIMULATION ===\n");
    
    let (x_train, y_train) = load_data("data/train_large.csv")?;
    let (x_val, y_val) = load_data("data/val_large.csv")?;
    let (mut x_test, y_test) = load_data("data/test_large.csv")?;
    
    // Calculate adaptive retrain threshold
    let pos_ratio = y_train.iter().sum::<f64>() / y_train.len() as f64;
    let retrain_threshold = if pos_ratio < 0.05 {
        0.058
    } else if pos_ratio < 0.15 {
        0.060
    } else {
        0.065
    };
    
    println!("Adaptive retrain threshold: {:.4} PR-AUC\n", retrain_threshold);
    
    println!("Training initial model...");
    let mut model = OptimizedPKBoostShannon::auto(&x_train, &y_train);
    model.fit(&x_train, &y_train, Some((&x_val, &y_val)), false)?;
    
    let mut total_obs = 0;
    let batch_size = 5000;
    let mut recent_pr_aucs = Vec::new();
    let mut retrain_count = 0;
    
    println!("\n=== PHASE 1: NORMAL DATA ===");
    let first_batch = &x_test[0..batch_size.min(x_test.len())].to_vec();
    let first_labels = &y_test[0..batch_size.min(y_test.len())].to_vec();
    total_obs += first_batch.len();
    let pr_auc = evaluate_batch(&model, first_batch, first_labels, total_obs, "Normal")?;
    recent_pr_aucs.push(pr_auc);
    
    println!("\n=== PHASE 2: INTRODUCING CONCEPT DRIFT ===");
    let drift_features = vec![5, 10, 15, 20, 25, 30, 35, 40, 45, 50];
    introduce_drift(&mut x_test, &drift_features, 5.0);
    
    let mut accumulated_x = Vec::new();
    let mut accumulated_y = Vec::new();
    
    for i in (batch_size..x_test.len()).step_by(batch_size) {
        let end = (i + batch_size).min(x_test.len());
        if i == end { continue; }
        
        let x_batch = x_test[i..end].to_vec();
        if x_batch.len() < 100 { continue; }
        
        let y_batch = y_test[i..end].to_vec();
        total_obs += x_batch.len();
        
        accumulated_x.extend(x_batch.clone());
        accumulated_y.extend(y_batch.clone());
        if accumulated_x.len() > 10000 {
            accumulated_x.drain(0..5000);
            accumulated_y.drain(0..5000);
        }
        
        let pr_auc = evaluate_batch(&model, &x_batch, &y_batch, total_obs, "Drift")?;
        recent_pr_aucs.push(pr_auc);
        if recent_pr_aucs.len() > 3 {
            recent_pr_aucs.remove(0);
        }
        
        let avg_recent = recent_pr_aucs.iter().sum::<f64>() / recent_pr_aucs.len() as f64;
        if avg_recent < 0.060 && accumulated_x.len() > 5000 {
            println!("\n=== PERFORMANCE DEGRADATION DETECTED - RETRAINING ===");
            let retrain_start = Instant::now();
            
            model = OptimizedPKBoostShannon::auto(&accumulated_x, &accumulated_y);
            model.fit(&accumulated_x, &accumulated_y, None, false)?;
            
            let retrain_time = retrain_start.elapsed().as_secs_f64();
            retrain_count += 1;
            println!("Retrain {} completed in {:.2}s", retrain_count, retrain_time);
            recent_pr_aucs.clear();
        }
    }
    
    println!("\n=== SIMULATION COMPLETE ===");
    println!("Total retrains: {}", retrain_count);
    
    Ok(())
}