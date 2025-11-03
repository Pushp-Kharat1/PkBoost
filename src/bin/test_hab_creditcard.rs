// Test HAB on real Credit Card fraud dataset
use pkboost::*;
use csv::ReaderBuilder;
use std::error::Error;
use std::time::Instant;

fn load_csv(path: &str) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let headers = reader.headers()?.clone();
    let target_idx = headers.iter().position(|h| h == "Class").ok_or("No Class column")?;
    
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
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== HAB Credit Card Fraud Detection ===\n");
    
    println!("Loading data...");
    let (x_train, y_train) = load_csv("data/creditcard_train.csv")?;
    let (x_val, y_val) = load_csv("data/creditcard_val.csv")?;
    let (x_test, y_test) = load_csv("data/creditcard_test.csv")?;
    
    let fraud_rate = y_train.iter().sum::<f64>() / y_train.len() as f64;
    println!("Train: {} samples, Fraud: {:.2}%", x_train.len(), fraud_rate * 100.0);
    println!("Val: {} samples", x_val.len());
    println!("Test: {} samples\n", x_test.len());
    
    // Baseline
    println!("=== Training Baseline ===");
    let t0 = Instant::now();
    let mut baseline = OptimizedPKBoostShannon::auto(&x_train, &y_train);
    baseline.fit(&x_train, &y_train, Some((&x_val, &y_val)), true)?;
    let baseline_time = t0.elapsed();
    
    let baseline_probs = baseline.predict_proba(&x_test)?;
    let baseline_pr_auc = calculate_pr_auc(&y_test, &baseline_probs);
    let baseline_roc_auc = calculate_roc_auc(&y_test, &baseline_probs);
    
    println!("\nBaseline Results:");
    println!("  PR-AUC:  {:.4}", baseline_pr_auc);
    println!("  ROC-AUC: {:.4}", baseline_roc_auc);
    println!("  Time:    {:.2}s\n", baseline_time.as_secs_f64());
    
    // HAB with tuned configurations for accuracy
    let configs = vec![
        ("Balanced (10 parts, 200 trees)", 10, 200, 6),
        ("Fast (20 parts, 100 trees)", 20, 100, 5),
    ];
    
    for (name, n_parts, n_trees, depth) in configs {
        println!("\n=== Training HAB: {} ===", name);
        let t1 = Instant::now();
        let mut hab = PartitionedClassifierBuilder::new()
            .n_partitions(n_parts)
            .specialist_estimators(n_trees)
            .specialist_max_depth(depth)
            .task_type(TaskType::Binary)
            .build();
        
        hab.partition_data(&x_train, &y_train, true);
        hab.train_specialists_with_validation(&x_train, &y_train, Some((&x_val, &y_val)), true)?;
        let hab_time = t1.elapsed();
        
        let hab_probs = hab.predict_proba(&x_test)?;
        let hab_probs_pos: Vec<f64> = hab_probs.iter().map(|p| p[1]).collect();
        let hab_pr_auc = calculate_pr_auc(&y_test, &hab_probs_pos);
        let hab_roc_auc = calculate_roc_auc(&y_test, &hab_probs_pos);
        
        println!("\nHAB Results:");
        println!("  PR-AUC:  {:.4} ({:+.1}%)", hab_pr_auc, 
            ((hab_pr_auc - baseline_pr_auc) / baseline_pr_auc) * 100.0);
        println!("  ROC-AUC: {:.4} ({:+.1}%)", hab_roc_auc,
            ((hab_roc_auc - baseline_roc_auc) / baseline_roc_auc) * 100.0);
        println!("  Time:    {:.2}s ({:.1}x faster)\n", 
            hab_time.as_secs_f64(), baseline_time.as_secs_f64() / hab_time.as_secs_f64());
    }
    
    Ok(())
}
