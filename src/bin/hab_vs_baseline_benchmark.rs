// Comprehensive HAB vs Baseline benchmark
use pkboost::*;
use rand::Rng;
use std::time::Instant;

fn generate_complex_fraud(n: usize, fraud_rate: f64, noise_level: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let n_fraud = (n as f64 * fraud_rate) as usize;
    let mut x = Vec::new();
    let mut y = Vec::new();
    
    // Normal transactions - 3 clusters
    for i in 0..(n - n_fraud) {
        let cluster = i % 3;
        let base = match cluster {
            0 => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1 => vec![2.0, -1.0, 1.0, -2.0, 0.5, 1.5, -0.5, 0.0, 1.0, -1.0],
            _ => vec![-1.0, 2.0, -0.5, 1.0, -1.5, 0.5, 2.0, -1.0, 0.0, 1.5],
        };
        let sample: Vec<f64> = base.iter()
            .map(|&v| v + rng.gen_range(-noise_level..noise_level))
            .collect();
        x.push(sample);
        y.push(0.0);
    }
    
    // Fraud transactions - 2 rare patterns
    for i in 0..n_fraud {
        let pattern = i % 2;
        let base = match pattern {
            0 => vec![5.0, 5.0, -3.0, 4.0, -4.0, 3.0, 5.0, -2.0, 4.0, -3.0],
            _ => vec![-4.0, -5.0, 5.0, -3.0, 5.0, -4.0, -3.0, 5.0, -5.0, 4.0],
        };
        let sample: Vec<f64> = base.iter()
            .map(|&v| v + rng.gen_range(-noise_level..noise_level))
            .collect();
        x.push(sample);
        y.push(1.0);
    }
    
    (x, y)
}

fn main() -> Result<(), String> {
    println!("=== HAB vs Baseline Comprehensive Benchmark ===\n");
    
    let configs = vec![
        ("Easy (0.5% fraud, low noise)", 20_000, 0.005, 0.5),
        ("Medium (0.2% fraud, med noise)", 20_000, 0.002, 1.0),
        ("Hard (0.1% fraud, high noise)", 20_000, 0.001, 1.5),
    ];
    
    for (name, n_train, fraud_rate, noise) in configs {
        println!("\n=== {} ===", name);
        println!("Train: {} samples, Fraud: {:.1}%, Noise: {:.1}", n_train, fraud_rate * 100.0, noise);
        
        let (x_train, y_train) = generate_complex_fraud(n_train, fraud_rate, noise);
        let (x_test, y_test) = generate_complex_fraud(5_000, fraud_rate, noise);
        
        // Baseline
        print!("Training baseline... ");
        let t0 = Instant::now();
        let mut baseline = OptimizedPKBoostShannon::auto(&x_train, &y_train);
        baseline.n_estimators = 500;
        baseline.fit(&x_train, &y_train, None, false)?;
        let baseline_time = t0.elapsed();
        
        let baseline_probs = baseline.predict_proba(&x_test)?;
        let baseline_pr_auc = calculate_pr_auc(&y_test, &baseline_probs);
        println!("PR-AUC: {:.4}, Time: {:.2}s", baseline_pr_auc, baseline_time.as_secs_f64());
        
        // HAB with different partition counts
        for n_parts in [10, 20, 40] {
            print!("Training HAB ({} partitions)... ", n_parts);
            let t1 = Instant::now();
            let mut hab = PartitionedClassifierBuilder::new()
                .n_partitions(n_parts)
                .specialist_estimators(50)
                .specialist_max_depth(4)
                .task_type(TaskType::Binary)
                .build();
            
            hab.partition_data(&x_train, &y_train, false);
            hab.train_specialists(&x_train, &y_train, false)?;
            let hab_time = t1.elapsed();
            
            let hab_probs = hab.predict_proba(&x_test)?;
            let hab_probs_pos: Vec<f64> = hab_probs.iter().map(|p| p[1]).collect();
            let hab_pr_auc = calculate_pr_auc(&y_test, &hab_probs_pos);
            
            let improvement = ((hab_pr_auc - baseline_pr_auc) / baseline_pr_auc) * 100.0;
            let speedup = baseline_time.as_secs_f64() / hab_time.as_secs_f64();
            
            println!("PR-AUC: {:.4} ({:+.1}%), Time: {:.2}s ({:.1}x)", 
                hab_pr_auc, improvement, hab_time.as_secs_f64(), speedup);
        }
    }
    
    Ok(())
}
