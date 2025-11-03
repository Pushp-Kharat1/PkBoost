use mimalloc::MiMalloc;
use pkboost::*;
use std::error::Error;
use std::fs::File;
use std::io::Write;

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

fn evaluate_and_observe_batch(
    alb: &mut AdversarialLivingBooster,
    x_batch: &Vec<Vec<f64>>,
    y_batch: &Vec<f64>,
    observation_count: usize,
    phase: &str,
    log_file: &mut File,
) -> Result<f64, String> {
    let predictions = alb.predict_proba(x_batch)?;
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

    writeln!(
        log_file,
        "{},{},{:.4},{:.4},{:.4},{:.4},{:?},{}",
        observation_count, phase, pr_auc, roc_auc, f1,
        alb.get_vulnerability_score(),
        alb.get_state(),
        alb.get_metamorphosis_count()
    ).map_err(|e| e.to_string())?;

    println!(
        "ALB_MODEL_METRIC | Phase: {} | Obs: {} | PR-AUC: {:.4} | ROC-AUC: {:.4} | F1@0.5: {:.4} | Vuln: {:.4} | State: {:?}",
        phase, observation_count, pr_auc, roc_auc, f1, alb.get_vulnerability_score(), alb.get_state()
    );

    alb.observe_batch(x_batch, y_batch, true)?;
    Ok(pr_auc)
}

fn diagnostic_main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== VULNERABILITY SYSTEM DIAGNOSTIC ===\n");

    let dataset = std::env::var("DRIFT_DATASET").unwrap_or_else(|_| "creditcard".to_string());
    println!("Dataset: {}\n", dataset);

    let (x_train, y_train) = load_data(&format!("data/creditcard_train.csv"))?;
    let (x_val, y_val) = load_data(&format!("data/creditcard_val.csv", ))?;
    let (mut x_test, y_test) = load_data(&format!("data/creditcard_test.csv",))?;
    
    let mut alb = AdversarialLivingBooster::new(&x_train, &y_train);
    alb.fit_initial(&x_train, &y_train, Some((&x_val, &y_val)), false)?;
    
    println!("=== PHASE 1: NORMAL DATA ===");
    let batch_size = 5000;
    let first_batch = &x_test[0..batch_size.min(x_test.len())].to_vec();
    let first_labels = &y_test[0..batch_size.min(y_test.len())].to_vec();
    
    let preds_normal = alb.predict_proba(first_batch)?;
    
    println!("Predictions on normal data:");
    println!("  Min pred: {:.6}", preds_normal.iter().cloned().fold(f64::INFINITY, f64::min));
    println!("  Max pred: {:.6}", preds_normal.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    println!("  Mean pred: {:.6}", preds_normal.iter().sum::<f64>() / preds_normal.len() as f64);
    
    // Count errors on minority class
    let mut errors_on_pos = 0;
    let mut total_pos = 0;
    for (i, &true_label) in first_labels.iter().enumerate() {
        if true_label > 0.5 {
            total_pos += 1;
            let pred_class = if preds_normal[i] >= 0.5 { 1 } else { 0 };
            if (pred_class as f64 - true_label).abs() > 0.5 {
                errors_on_pos += 1;
            }
        }
    }
    println!("  Errors on positive class: {}/{}", errors_on_pos, total_pos);
    
    alb.observe_batch(first_batch, first_labels, false)?;
    println!("  Vulnerability score after normal: {:.6}\n", alb.get_vulnerability_score());
    
    // Apply drift
    println!("=== APPLYING DRIFT ===");
    let drift_features = vec![5, 10, 15, 20, 25];
    for row in x_test.iter_mut() {
        for &feat_idx in &drift_features {
            if feat_idx < row.len() {
                row[feat_idx] = -row[feat_idx] + 10.0;
            }
        }
    }
    
    // Test on drifted data
    println!("=== PHASE 2: DRIFTED DATA ===");
    let drifted_batch = &x_test[batch_size..2*batch_size].to_vec();
    let drifted_labels = &y_test[batch_size..2*batch_size].to_vec();
    
    let preds_drifted = alb.predict_proba(drifted_batch)?;
    
    println!("Predictions on drifted data:");
    println!("  Min pred: {:.6}", preds_drifted.iter().cloned().fold(f64::INFINITY, f64::min));
    println!("  Max pred: {:.6}", preds_drifted.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
    println!("  Mean pred: {:.6}", preds_drifted.iter().sum::<f64>() / preds_drifted.len() as f64);
    
    // Count errors on minority class
    let mut errors_on_pos_drift = 0;
    let mut total_pos_drift = 0;
    for (i, &true_label) in drifted_labels.iter().enumerate() {
        if true_label > 0.5 {
            total_pos_drift += 1;
            let pred_class = if preds_drifted[i] >= 0.5 { 1 } else { 0 };
            if (pred_class as f64 - true_label).abs() > 0.5 {
                errors_on_pos_drift += 1;
            }
        }
    }
    println!("  Errors on positive class: {}/{}", errors_on_pos_drift, total_pos_drift);
    
    // Manually compute vulnerability scores for a few samples
    println!("\n=== MANUAL VULNERABILITY CHECK ===");
    println!("Checking individual sample vulnerabilities...");
    for i in 0..10.min(drifted_labels.len()) {
        if drifted_labels[i] > 0.5 {
            let confidence = (preds_drifted[i] - 0.5).abs() * 2.0;
            let error = (drifted_labels[i] - preds_drifted[i]).abs();
            println!("  Sample {}: pred={:.4}, true={:.1}, conf={:.4}, error={:.4}",
                     i, preds_drifted[i], drifted_labels[i], confidence, error);
        }
    }
    
    alb.observe_batch(drifted_batch, drifted_labels, false)?;
    let vuln_after = alb.get_vulnerability_score();
    println!("\n  Vulnerability score after drift: {:.6}", vuln_after);
    println!("  State: {:?}", alb.get_state());
    
    if vuln_after == 0.0 {
        println!("\n  DIAGNOSTIC: Vulnerability score is still 0!");
        println!("   This means either:");
        println!("   1. Model predictions are correct even on drifted data (unlikely)");
        println!("   2. Vulnerability recording isn't working");
        println!("   3. There are no positive examples in the batch");
    }
    
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Uncomment to run diagnostic:
    // return diagnostic_main();
    
    println!("\n=== ADVERSARIAL LIVING BOOSTER (ALB) DRIFT SIMULATION ===\n");

    let dataset = std::env::var("DRIFT_DATASET").unwrap_or_else(|_| "creditcard".to_string());
    println!("Dataset: {}\n", dataset);

    let mut log_file = File::create("alb_metrics.csv")?;
    writeln!(log_file, "observation,phase,pr_auc,roc_auc,f1,vuln_score,state,metamorphosis_count")?;
    
    let (x_train, y_train) = load_data(&format!("data/{}_train.csv", dataset))?;
    let (x_val, y_val) = load_data(&format!("data/{}_val.csv", dataset))?;
    let (mut x_test, y_test) = load_data(&format!("data/{}_test.csv", dataset))?;
    
    println!("Creating Adversarial Living Booster...");
    let mut alb = AdversarialLivingBooster::new(&x_train, &y_train);
    
    println!("\n⚠️  METAMORPHOSIS ENABLED (will trigger on drift detection) ⚠️\n");
    
    println!("\nInitial training...");
    alb.fit_initial(&x_train, &y_train, Some((&x_val, &y_val)), false)?;
    println!("Initial training complete. Model ready for streaming.");
    
    let mut total_obs = 0;
    let batch_size = 5000;
    
    println!("\n=== PHASE 1: NORMAL DATA ===");
    let first_batch = &x_test[0..batch_size.min(x_test.len())].to_vec();
    let first_labels = &y_test[0..batch_size.min(y_test.len())].to_vec();
    total_obs += first_batch.len();
    let baseline_pr = evaluate_and_observe_batch(&mut alb, first_batch, first_labels, total_obs, "Normal", &mut log_file)?;
    
    println!("\n=== PHASE 2: INTRODUCING GRADUAL REALISTIC DRIFT ===");

    let drift_features = vec![0, 1, 2, 3, 4];
    
    println!("Applying gradual covariate shift to {} features (simulating distribution change)\n", drift_features.len());

    // Realistic drift: small shift and scale (like sensor calibration drift)
    for row in x_test.iter_mut() {
        for &feat_idx in &drift_features {
            if feat_idx < row.len() {
                row[feat_idx] = row[feat_idx] * 1.2 + 0.5;  // 20% scale + small shift
            }
        }
    }

    let mut pre_meta_metrics = Vec::new();
    let mut post_meta_metrics = Vec::new();
    let mut metamorphosis_occurred = false;
    let mut metamorphosis_at_obs = 0;

    for i in (batch_size..x_test.len()).step_by(batch_size) {
        let end = (i + batch_size).min(x_test.len());
        if i == end { continue; }
        let x_batch = x_test[i..end].to_vec();
        
        if x_batch.len() < 100 {
            println!("Skipping small batch of {} samples", x_batch.len());
            continue;
        }

        let y_batch = y_test[i..end].to_vec();
        total_obs += x_batch.len();
        
        let pr_auc = evaluate_and_observe_batch(&mut alb, &x_batch, &y_batch, total_obs, "Drift", &mut log_file)?;
        
        // Track metrics before and after metamorphosis
        if !metamorphosis_occurred {
            pre_meta_metrics.push(pr_auc);
            if alb.get_metamorphosis_count() > 0 {
                metamorphosis_occurred = true;
                metamorphosis_at_obs = total_obs;
                println!("\nMETAMORPHOSIS DETECTED at observation {}!", total_obs);
                println!("Continuing to measure recovery...\n");
            }
        } else {
            post_meta_metrics.push(pr_auc);
            if post_meta_metrics.len() >= 3 {
                break;
            }
        }
    }
    
    println!("\n=== PERFORMANCE ANALYSIS ===");
    println!("Baseline (normal data): {:.4}", baseline_pr);
    
    if !pre_meta_metrics.is_empty() {
        let pre_avg = pre_meta_metrics.iter().sum::<f64>() / pre_meta_metrics.len() as f64;
        let pre_min = pre_meta_metrics.iter().cloned().fold(f64::INFINITY, f64::min);
        let pre_max = pre_meta_metrics.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("\nPre-Metamorphosis (drift):");
        println!("  Average PR-AUC: {:.4}", pre_avg);
        println!("  Range: [{:.4}, {:.4}]", pre_min, pre_max);
        println!("  Degradation: {:.1}%", ((baseline_pr - pre_avg) / baseline_pr * 100.0));
    }
    
    if metamorphosis_occurred && !post_meta_metrics.is_empty() {
        let post_avg = post_meta_metrics.iter().sum::<f64>() / post_meta_metrics.len() as f64;
        let post_min = post_meta_metrics.iter().cloned().fold(f64::INFINITY, f64::min);
        let post_max = post_meta_metrics.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let pre_avg = pre_meta_metrics.iter().sum::<f64>() / pre_meta_metrics.len() as f64;
        
        println!("\nPost-Metamorphosis (recovery):");
        println!("  Average PR-AUC: {:.4}", post_avg);
        println!("  Range: [{:.4}, {:.4}]", post_min, post_max);
        println!("  Recovery: {:.1}%", ((post_avg - pre_avg) / pre_avg * 100.0));
        println!("  vs Baseline: {:.1}%", ((post_avg - baseline_pr) / baseline_pr * 100.0));
        
        if post_avg > pre_avg + 0.01 {
            println!("\nRECOVERY CONFIRMED: Performance improved after metamorphosis");
        } else {
            println!("\nNO SIGNIFICANT RECOVERY: Metamorphosis did not improve performance");
        }
    } else {
        println!("\nMetamorphosis did not trigger during test");
        println!("This confirms vulnerability detection needs the class-weighting fix");
    }
    
    println!("\n=== FINAL STATUS ===");
    println!("System State: {:?}", alb.get_state());
    println!("Vulnerability Score: {:.4}", alb.get_vulnerability_score());
    println!("Metamorphosis Count: {}", alb.get_metamorphosis_count());
    println!("Metamorphosis occurred at observation: {}", metamorphosis_at_obs);
    println!("\nMetrics saved to: alb_metrics.csv");
    println!("=== END OF SIMULATION ===\n");
    
    Ok(())
}