// Test HAB with streaming data and drift detection
use pkboost::*;
use rand::Rng;
use std::collections::VecDeque;

fn generate_batch(_batch_id: usize, n: usize, fraud_rate: f64, drift_factor: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let n_fraud = (n as f64 * fraud_rate) as usize;
    let mut x = Vec::new();
    let mut y = Vec::new();
    
    // Normal transactions
    for _ in 0..(n - n_fraud) {
        x.push(vec![
            rng.gen_range(-1.0..1.0) + drift_factor,
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        ]);
        y.push(0.0);
    }
    
    // Fraud transactions (shift with drift)
    for _ in 0..n_fraud {
        x.push(vec![
            rng.gen_range(1.0..3.0) + drift_factor * 2.0,
            rng.gen_range(1.0..3.0) + drift_factor,
            rng.gen_range(-2.0..0.0),
            rng.gen_range(1.5..2.5),
            rng.gen_range(-1.5..-0.5),
        ]);
        y.push(1.0);
    }
    
    (x, y)
}

fn main() -> Result<(), String> {
    println!("=== HAB Streaming Adaptation Test ===\n");
    
    // Initial training
    println!("Generating initial training data...");
    let (x_train, y_train) = generate_batch(0, 10_000, 0.005, 0.0);
    let (x_test_init, y_test_init) = generate_batch(0, 2_000, 0.005, 0.0);
    
    println!("Training HAB...");
    let mut hab = PartitionedClassifierBuilder::new()
        .n_partitions(20)
        .specialist_estimators(40)
        .specialist_max_depth(3)
        .task_type(TaskType::Binary)
        .build();
    
    hab.partition_data(&x_train, &y_train, false);
    hab.train_specialists(&x_train, &y_train, false)?;
    
    // Initial performance
    let probs = hab.predict_proba(&x_test_init)?;
    let probs_pos: Vec<f64> = probs.iter().map(|p| p[1]).collect();
    let initial_pr_auc = calculate_pr_auc(&y_test_init, &probs_pos);
    println!("Initial PR-AUC: {:.4}\n", initial_pr_auc);
    
    // Simulate streaming with drift
    let mut buffer_x = VecDeque::new();
    let mut buffer_y = VecDeque::new();
    let buffer_limit = 5000;
    
    println!("=== Streaming Batches ===");
    for batch_id in 1..=10 {
        // Introduce drift gradually
        let drift = if batch_id > 5 { (batch_id - 5) as f64 * 0.5 } else { 0.0 };
        let (x_batch, y_batch) = generate_batch(batch_id, 1000, 0.005, drift);
        
        // Observe batch for drift
        let drifted = hab.observe_batch(&x_batch, &y_batch);
        
        // Add to buffer
        buffer_x.extend(x_batch.clone());
        buffer_y.extend(y_batch.clone());
        while buffer_x.len() > buffer_limit {
            buffer_x.pop_front();
            buffer_y.pop_front();
        }
        
        // Test performance
        let (x_test, y_test) = generate_batch(batch_id, 1000, 0.005, drift);
        let probs = hab.predict_proba(&x_test)?;
        let probs_pos: Vec<f64> = probs.iter().map(|p| p[1]).collect();
        let pr_auc = calculate_pr_auc(&y_test, &probs_pos);
        
        print!("Batch {}: PR-AUC={:.4}, Drift={:.1}", batch_id, pr_auc, drift);
        
        if !drifted.is_empty() {
            println!(" [DRIFT] in partitions: {:?}", drifted);
            
            // Metamorphosis
            let buffer_vec_x: Vec<Vec<f64>> = buffer_x.iter().cloned().collect();
            let buffer_vec_y: Vec<f64> = buffer_y.iter().cloned().collect();
            hab.metamorph_partitions(&drifted, &buffer_vec_x, &buffer_vec_y, true)?;
            
            // Test after adaptation
            let probs_after = hab.predict_proba(&x_test)?;
            let probs_pos_after: Vec<f64> = probs_after.iter().map(|p| p[1]).collect();
            let pr_auc_after = calculate_pr_auc(&y_test, &probs_pos_after);
            println!("  After metamorphosis: PR-AUC={:.4} ({:+.1}%)", 
                pr_auc_after, ((pr_auc_after - pr_auc) / pr_auc) * 100.0);
        } else {
            println!(" [OK] No drift");
        }
    }
    
    Ok(())
}
