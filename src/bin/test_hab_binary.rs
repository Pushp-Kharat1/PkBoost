// Test HAB on binary classification with extreme imbalance
use pkboost::*;
use rand::Rng;

fn generate_fraud_data(n_samples: usize, fraud_rate: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let n_fraud = (n_samples as f64 * fraud_rate) as usize;
    let n_normal = n_samples - n_fraud;
    
    let mut x = Vec::new();
    let mut y = Vec::new();
    
    for _ in 0..n_normal {
        x.push(vec![
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        ]);
        y.push(0.0);
    }
    
    for _ in 0..n_fraud {
        x.push(vec![
            rng.gen_range(1.0..3.0),
            rng.gen_range(1.0..3.0),
            rng.gen_range(-2.0..0.0),
            rng.gen_range(1.5..2.5),
            rng.gen_range(-1.5..-0.5),
        ]);
        y.push(1.0);
    }
    
    (x, y)
}

fn main() -> Result<(), String> {
    println!("=== HAB Binary Classification Test ===\n");
    
    let fraud_rate = 0.002;
    println!("Generating data ({:.1}% fraud rate)...", fraud_rate * 100.0);
    let (x_train, y_train) = generate_fraud_data(10_000, fraud_rate);
    let (x_test, y_test) = generate_fraud_data(2_000, fraud_rate);
    
    println!("Train: {} samples, Test: {} samples\n", x_train.len(), x_test.len());
    
    // Train baseline
    println!("=== Training Baseline (Single Model) ===");
    let mut baseline = OptimizedPKBoostShannon::auto(&x_train, &y_train);
    baseline.fit(&x_train, &y_train, None, true)?;
    
    let baseline_probs = baseline.predict_proba(&x_test)?;
    let baseline_pr_auc = calculate_pr_auc(&y_test, &baseline_probs);
    
    // Train HAB
    println!("\n=== Training HAB (20 Partitions) ===");
    let mut hab = PartitionedClassifierBuilder::new()
        .n_partitions(20)
        .specialist_estimators(40)
        .specialist_max_depth(3)
        .task_type(TaskType::Binary)
        .build();
    
    hab.partition_data(&x_train, &y_train, true);
    hab.train_specialists(&x_train, &y_train, true)?;
    
    let hab_probs = hab.predict_proba(&x_test)?;
    let hab_probs_positive: Vec<f64> = hab_probs.iter().map(|p| p[1]).collect();
    let hab_pr_auc = calculate_pr_auc(&y_test, &hab_probs_positive);
    
    // Results
    println!("\n=== RESULTS ===");
    println!("Baseline PR-AUC: {:.4}", baseline_pr_auc);
    println!("HAB PR-AUC:      {:.4}", hab_pr_auc);
    println!("Improvement:     {:.1}%", ((hab_pr_auc - baseline_pr_auc) / baseline_pr_auc) * 100.0);
    
    Ok(())
}
