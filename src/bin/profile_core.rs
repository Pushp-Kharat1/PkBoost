// Profile core model performance to identify bottlenecks
use pkboost::*;
use std::time::Instant;

fn generate_data(n: usize, n_features: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    let x: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..n_features).map(|_| rng.gen_range(-10.0..10.0)).collect())
        .collect();
    
    let y: Vec<f64> = x.iter()
        .map(|row| row.iter().sum::<f64>() + rng.gen_range(-5.0..5.0))
        .collect();
    
    (x, y)
}

fn main() {
    println!("=== PKBoost Core Performance Profiling ===\n");
    
    let sizes = vec![1000, 5000, 10000];
    let features = vec![10, 20, 50];
    
    for &n in &sizes {
        for &f in &features {
            println!("Dataset: {} samples, {} features", n, f);
            
            let (x, y) = generate_data(n, f);
            let split = (n as f64 * 0.8) as usize;
            let x_train = x[..split].to_vec();
            let y_train = y[..split].to_vec();
            let x_val = x[split..].to_vec();
            let y_val = y[split..].to_vec();
            
            // Time histogram building
            let t0 = Instant::now();
            let mut hb = histogram_builder::OptimizedHistogramBuilder::new(32);
            hb.fit(&x_train);
            let hist_time = t0.elapsed();
            
            // Time transformation
            let t1 = Instant::now();
            let x_train_proc = hb.transform(&x_train);
            let transform_time = t1.elapsed();
            
            // Time model creation
            let t2 = Instant::now();
            let mut model = PKBoostRegressor::auto(&x_train, &y_train);
            model.n_estimators = 100; // Fixed for comparison
            let auto_time = t2.elapsed();
            
            // Time training (first 10 trees)
            model.histogram_builder = Some(hb);
            model.n_estimators = 10;
            let t3 = Instant::now();
            model.fit(&x_train, &y_train, Some((&x_val, &y_val)), false).unwrap();
            let train_10_time = t3.elapsed();
            
            // Time prediction
            let t4 = Instant::now();
            let _ = model.predict(&x_val).unwrap();
            let pred_time = t4.elapsed();
            
            println!("  Histogram build: {:?}", hist_time);
            println!("  Transform:       {:?}", transform_time);
            println!("  Auto-tune:       {:?}", auto_time);
            println!("  Train 10 trees:  {:?}", train_10_time);
            println!("  Predict:         {:?}", pred_time);
            println!("  Per-tree avg:    {:?}", train_10_time / 10);
            println!();
        }
    }
}
