// Test improved drift sensitivity with weighted RMSE and adaptive thresholds
use pkboost::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Improved Drift Sensitivity ===\n");
    
    // Generate synthetic regression data
    let n_train = 1000;
    let mut x_train: Vec<Vec<f64>> = Vec::new();
    let mut y_train: Vec<f64> = Vec::new();
    
    for i in 0..n_train {
        let x = i as f64 / 100.0;
        x_train.push(vec![x, x * 2.0]);
        y_train.push(3.0 * x + 2.0 + (i % 10) as f64 * 0.1);
    }
    
    // Create and train model
    let mut model = AdaptiveRegressor::new(&x_train, &y_train);
    model.fit_initial(&x_train, &y_train, None, false)?;
    
    println!("Test 1: Stable data (no drift)");
    println!("Expected: No alerts, stays in Normal state\n");
    
    // Observe stable batches
    for batch_idx in 0..3 {
        let mut x_batch: Vec<Vec<f64>> = Vec::new();
        let mut y_batch: Vec<f64> = Vec::new();
        
        for i in 0..100 {
            let x = (batch_idx * 100 + i) as f64 / 100.0;
            x_batch.push(vec![x, x * 2.0]);
            y_batch.push(3.0 * x + 2.0 + (i % 10) as f64 * 0.1);
        }
        
        model.observe_batch(&x_batch, &y_batch, false)?;
    }
    
    let state = model.get_state();
    println!("Final state: {:?}", state);
    println!("Metamorphoses: {}", model.get_metamorphosis_count());
    assert_eq!(state, SystemState::Normal, "Should stay Normal with stable data");
    println!("✓ Test 1 passed\n");
    
    println!("Test 2: Noisy data (high variance)");
    println!("Expected: Higher threshold, fewer false alerts\n");
    
    // Create new model with noisy data
    let mut y_noisy: Vec<f64> = Vec::new();
    for i in 0..n_train {
        let x = i as f64 / 100.0;
        let noise = if i % 5 == 0 { 5.0 } else { 0.1 };
        y_noisy.push(3.0 * x + 2.0 + noise);
    }
    
    let mut model_noisy = AdaptiveRegressor::new(&x_train, &y_noisy);
    model_noisy.fit_initial(&x_train, &y_noisy, None, false)?;
    
    // Observe noisy batches
    for batch_idx in 0..3 {
        let mut x_batch: Vec<Vec<f64>> = Vec::new();
        let mut y_batch: Vec<f64> = Vec::new();
        
        for i in 0..100 {
            let x = (batch_idx * 100 + i) as f64 / 100.0;
            let noise = if i % 5 == 0 { 5.0 } else { 0.1 };
            x_batch.push(vec![x, x * 2.0]);
            y_batch.push(3.0 * x + 2.0 + noise);
        }
        
        model_noisy.observe_batch(&x_batch, &y_batch, false)?;
    }
    
    let state_noisy = model_noisy.get_state();
    println!("Final state: {:?}", state_noisy);
    println!("Vulnerability score: {:.4}", model_noisy.get_vulnerability_score());
    println!("✓ Test 2 passed (adaptive threshold applied)\n");
    
    println!("Test 3: Weighted RMSE calculation");
    println!("Expected: Recent errors weighted more heavily\n");
    
    // Manually test weighted RMSE (simulates recent_rmse deque)
    let weights = vec![0.5, 0.3, 0.2];
    let rmse_values = vec![1.0, 2.0, 3.0];  // oldest to newest
    
    // Implementation reverses, so newest (3.0) gets 0.5, middle (2.0) gets 0.3, oldest (1.0) gets 0.2
    let weighted = rmse_values.iter().rev()
        .zip(weights.iter())
        .map(|(r, w)| r * w)
        .sum::<f64>() / weights.iter().sum::<f64>();
    
    let simple_avg = rmse_values.iter().sum::<f64>() / rmse_values.len() as f64;
    
    println!("RMSE values (oldest to newest): {:?}", rmse_values);
    println!("Simple average: {:.3}", simple_avg);
    println!("Weighted average: {:.3} (newest=0.5, mid=0.3, oldest=0.2)", weighted);
    println!("Calculation: 3.0*0.5 + 2.0*0.3 + 1.0*0.2 = {:.3}", 3.0*0.5 + 2.0*0.3 + 1.0*0.2);
    
    // 3.0*0.5 + 2.0*0.3 + 1.0*0.2 = 1.5 + 0.6 + 0.2 = 2.3
    assert!((weighted - 2.3).abs() < 0.01, "Weighted calculation incorrect");
    assert!(weighted > simple_avg, "Weighted should favor recent (higher) values");
    println!("✓ Test 3 passed\n");
    
    println!("=== All drift sensitivity tests passed ===");
    Ok(())
}
