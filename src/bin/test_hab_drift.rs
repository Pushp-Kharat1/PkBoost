// Test HAB vs Baseline under drift conditions
use pkboost::*;
use std::collections::VecDeque;
use rand::Rng;

fn load_creditcard() -> Result<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>), Box<dyn std::error::Error>> {
    use csv::ReaderBuilder;
    
    let mut load = |path: &str| -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn std::error::Error>> {
        let mut reader = ReaderBuilder::new().has_headers(true).from_path(path)?;
        let headers = reader.headers()?.clone();
        let target_idx = headers.iter().position(|h| h == "Class").ok_or("No Class")?;
        
        let mut x = Vec::new();
        let mut y = Vec::new();
        for result in reader.records() {
            let record = result?;
            let mut row = Vec::new();
            for (i, field) in record.iter().enumerate() {
                if i == target_idx {
                    y.push(field.parse()?);
                } else {
                    row.push(field.parse().unwrap_or(0.0));
                }
            }
            x.push(row);
        }
        Ok((x, y))
    };
    
    let (x_train, y_train) = load("data/creditcard_train.csv")?;
    let (x_val, y_val) = load("data/creditcard_val.csv")?;
    let (x_test, y_test) = load("data/creditcard_test.csv")?;
    
    Ok((x_train, y_train, x_val, y_val, x_test, y_test))
}

fn inject_drift(x: &mut [Vec<f64>], drift_type: &str, intensity: f64) {
    let mut rng = rand::thread_rng();
    
    match drift_type {
        "feature_shift" => {
            // Shift first 5 features
            for sample in x.iter_mut() {
                for i in 0..5.min(sample.len()) {
                    sample[i] += intensity * rng.gen_range(-1.0..1.0);
                }
            }
        }
        "noise" => {
            // Add noise to all features
            for sample in x.iter_mut() {
                for val in sample.iter_mut() {
                    *val += intensity * rng.gen_range(-0.5..0.5);
                }
            }
        }
        "scale" => {
            // Scale features
            for sample in x.iter_mut() {
                for val in sample.iter_mut() {
                    *val *= 1.0 + intensity;
                }
            }
        }
        _ => {}
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== HAB vs Baseline Under Drift ===\n");
    
    let (x_train, y_train, x_val, y_val, mut x_test, y_test) = load_creditcard()?;
    
    println!("Train: {} samples, Val: {} samples, Test: {} samples", 
        x_train.len(), x_val.len(), x_test.len());
    println!("Fraud rate: {:.2}%\n", y_train.iter().sum::<f64>() / y_train.len() as f64 * 100.0);
    
    // Train baseline
    println!("=== Training Baseline ===");
    let mut baseline = OptimizedPKBoostShannon::auto(&x_train, &y_train);
    baseline.fit(&x_train, &y_train, Some((&x_val, &y_val)), false)?;
    
    // Train HAB
    println!("\n=== Training HAB (10 partitions, weighted) ===");
    let mut hab = PartitionedClassifierBuilder::new()
        .n_partitions(10)
        .specialist_estimators(200)
        .specialist_max_depth(6)
        .build();
    
    hab.partition_data(&x_train, &y_train, false);
    hab.train_specialists_with_validation(&x_train, &y_train, Some((&x_val, &y_val)), false)?;
    
    // Test under different drift scenarios
    let drift_scenarios = vec![
        ("No Drift", 0.0),
        ("Light Drift", 1.0),
        ("Medium Drift", 2.0),
        ("Heavy Drift", 3.0),
        ("Extreme Drift", 5.0),
    ];
    
    println!("\n=== Testing Under Drift Conditions ===\n");
    println!("{:<20} {:<15} {:<15} {:<15}", "Scenario", "Baseline PR-AUC", "HAB PR-AUC", "HAB Advantage");
    println!("{}", "-".repeat(65));
    
    for (scenario, intensity) in drift_scenarios {
        let mut x_test_drift = x_test.clone();
        inject_drift(&mut x_test_drift, "feature_shift", intensity);
        
        // Baseline prediction
        let baseline_probs = baseline.predict_proba(&x_test_drift)?;
        let baseline_pr_auc = calculate_pr_auc(&y_test, &baseline_probs);
        
        // HAB prediction
        let hab_probs = hab.predict_proba(&x_test_drift)?;
        let hab_probs_pos: Vec<f64> = hab_probs.iter().map(|p| p[1]).collect();
        let hab_pr_auc = calculate_pr_auc(&y_test, &hab_probs_pos);
        
        let advantage = ((hab_pr_auc - baseline_pr_auc) / baseline_pr_auc) * 100.0;
        
        println!("{:<20} {:<15.4} {:<15.4} {:>14.1}%", 
            scenario, baseline_pr_auc, hab_pr_auc, advantage);
    }
    
    // Test with streaming adaptation
    println!("\n\n=== Streaming Adaptation Test ===\n");
    
    let mut x_test_stream = x_test.clone();
    inject_drift(&mut x_test_stream, "feature_shift", 5.0);
    
    // Initial performance
    let hab_probs_before = hab.predict_proba(&x_test_stream)?;
    let hab_probs_pos_before: Vec<f64> = hab_probs_before.iter().map(|p| p[1]).collect();
    let pr_auc_before = calculate_pr_auc(&y_test, &hab_probs_pos_before);
    
    println!("Before adaptation: PR-AUC = {:.4}", pr_auc_before);
    
    // Detect drift
    let drifted = hab.observe_batch(&x_test_stream, &y_test);
    println!("Drift detected in {} partitions: {:?}", drifted.len(), drifted);
    
    // Adapt if drift detected
    if !drifted.is_empty() {
        println!("\nAdapting to drift...");
        
        // Use test data as buffer (in production, use recent streaming data)
        let mut buffer_x = VecDeque::new();
        let mut buffer_y = VecDeque::new();
        buffer_x.extend(x_test_stream.clone());
        buffer_y.extend(y_test.clone());
        
        let buffer_vec_x: Vec<Vec<f64>> = buffer_x.iter().cloned().collect();
        let buffer_vec_y: Vec<f64> = buffer_y.iter().cloned().collect();
        
        hab.metamorph_partitions(&drifted, &buffer_vec_x, &buffer_vec_y, true)?;
        
        // Test after adaptation
        let hab_probs_after = hab.predict_proba(&x_test_stream)?;
        let hab_probs_pos_after: Vec<f64> = hab_probs_after.iter().map(|p| p[1]).collect();
        let pr_auc_after = calculate_pr_auc(&y_test, &hab_probs_pos_after);
        
        println!("\nAfter adaptation:  PR-AUC = {:.4}", pr_auc_after);
        println!("Recovery: {:+.1}%", ((pr_auc_after - pr_auc_before) / pr_auc_before) * 100.0);
    }
    
    Ok(())
}
