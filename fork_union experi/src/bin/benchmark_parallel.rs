/// Benchmark: fork_union performance test
use std::time::Instant;

fn generate_synthetic_data(n_samples: usize, n_features: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    let x: Vec<Vec<f64>> = (0..n_samples)
        .map(|_| (0..n_features).map(|_| rng.gen_range(-10.0..10.0)).collect())
        .collect();
    
    let y: Vec<f64> = (0..n_samples).map(|_| if rng.gen::<f64>() > 0.5 { 1.0 } else { 0.0 }).collect();
    
    (x, y)
}

fn main() {
    println!("=== FORK_UNION Performance Benchmark ===\n");
    
    let sizes = vec![
        (5000, 50, "Small"),
        (20000, 100, "Medium"),
        (50000, 150, "Large"),
    ];
    
    for (n_samples, n_features, label) in sizes {
        println!("--- {} Dataset: {} samples, {} features ---", label, n_samples, n_features);
        
        let (x_train, y_train) = generate_synthetic_data(n_samples, n_features);
        
        let start = Instant::now();
        let mut hist_builder = pkboost::histogram_builder::OptimizedHistogramBuilder::new(32);
        hist_builder.fit(&x_train);
        let hist_time = start.elapsed();
        println!("  Histogram: {:.3}s", hist_time.as_secs_f64());
        
        let start = Instant::now();
        let mut model = pkboost::model::OptimizedPKBoostShannon::new();
        model.n_estimators = 50;
        model.max_depth = 6;
        model.learning_rate = 0.1;
        
        if let Err(e) = model.fit(&x_train, &y_train, None, false) {
            println!("  Training failed: {}", e);
            continue;
        }
        let train_time = start.elapsed();
        println!("  Training (50 trees): {:.3}s", train_time.as_secs_f64());
        
        let start = Instant::now();
        let _preds = model.predict_proba(&x_train).unwrap();
        let pred_time = start.elapsed();
        println!("  Prediction: {:.3}s", pred_time.as_secs_f64());
        
        let total = hist_time + train_time + pred_time;
        println!("  TOTAL: {:.3}s\n", total.as_secs_f64());
    }
    
    println!("=== Benchmark Complete ===");
}
