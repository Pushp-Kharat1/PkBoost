// PKBoost drift benchmark - outputs results for Python comparison
use pkboost::*;
use std::f64::consts::PI;

fn generate_data(n: usize, start_idx: usize, coef: f64, intercept: f64, noise_level: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..n {
        let x_val = (start_idx + i) as f64 / 100.0;
        x.push(vec![x_val, x_val * 2.0, (x_val * 3.0).sin()]);
        let noise = ((start_idx + i) % 10) as f64 * noise_level;
        y.push(coef * x_val + intercept + noise);
    }
    (x, y)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("PKBOOST_BENCHMARK_START");
    
    // Training
    let (x_train, y_train) = generate_data(5000, 0, 2.0, 5.0, 0.1);
    
    let mut model = AdaptiveRegressor::new(&x_train, &y_train);
    let start = std::time::Instant::now();
    model.fit_initial(&x_train, &y_train, None, false)?;
    let train_time = start.elapsed().as_secs_f64();
    
    let train_preds = model.predict(&x_train)?;
    let train_rmse = calculate_rmse(&y_train, &train_preds);
    
    println!("TRAIN_TIME:{:.2}", train_time);
    println!("TRAIN_RMSE:{:.4}", train_rmse);
    
    // Test scenarios
    let scenarios = vec![
        ("1_Stable", 1000, 5000, 2.0, 5.0, 0.1, false, false),
        ("2_Sudden_Drift", 1000, 6000, 4.0, 15.0, 0.1, false, false),
        ("3_Gradual_Drift", 1000, 7000, 6.0, 15.0, 0.1, false, false),
        ("4_Outliers", 1000, 8000, 6.0, 15.0, 0.1, true, false),
        ("5_High_Noise", 1000, 9000, 6.0, 15.0, 5.0, false, false),
        ("6_Temporal", 1000, 10000, 6.0, 15.0, 0.1, false, true),
        ("7_Reversal", 1000, 11000, -6.0, 100.0, 0.1, false, false),
    ];
    
    for (name, n, start_idx, coef, intercept, noise, add_outliers, add_temporal) in scenarios {
        let (mut x_test, mut y_test) = generate_data(n, start_idx, coef, intercept, noise);
        
        if add_outliers {
            for i in (0..y_test.len()).step_by(20) {
                y_test[i] += 200.0;
            }
        }
        
        if add_temporal {
            for i in 0..y_test.len() {
                y_test[i] += i as f64 * 0.02;
            }
        }
        
        // Observe batch (adaptive features active)
        model.observe_batch(&x_test, &y_test, false)?;
        
        // Get predictions
        let preds = model.predict(&x_test)?;
        let test_rmse = calculate_rmse(&y_test, &preds);
        
        println!("SCENARIO:{}:RMSE:{:.4}", name, test_rmse);
    }
    
    println!("METAMORPHOSES:{}", model.get_metamorphosis_count());
    println!("PKBOOST_BENCHMARK_END");
    
    Ok(())
}
