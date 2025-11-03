// Test loss function selection
use pkboost::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Loss Function Selection ===\n");
    
    // Test 1: Clean data (should use MSE)
    println!("Test 1: Clean data (no outliers)");
    let y_clean = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let outlier_ratio = detect_outliers(&y_clean);
    println!("  Outlier ratio: {:.3}", outlier_ratio);
    
    let x_clean: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64]).collect();
    let model_clean = PKBoostRegressor::auto(&x_clean, &y_clean);
    match model_clean.loss_type {
        RegressionLossType::MSE => println!("  ✓ Selected MSE loss (correct)"),
        RegressionLossType::Huber { delta } => println!("  ✗ Selected Huber loss with delta={:.3} (unexpected)", delta),
    }
    
    // Test 2: Data with outliers (should use Huber)
    println!("\nTest 2: Data with outliers");
    let y_outliers = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0];
    let outlier_ratio = detect_outliers(&y_outliers);
    println!("  Outlier ratio: {:.3}", outlier_ratio);
    
    let x_outliers: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64]).collect();
    let model_outliers = PKBoostRegressor::auto(&x_outliers, &y_outliers);
    match model_outliers.loss_type {
        RegressionLossType::MSE => println!("  ✗ Selected MSE loss (unexpected)"),
        RegressionLossType::Huber { delta } => {
            println!("  ✓ Selected Huber loss (correct)");
            println!("  Delta: {:.3}", delta);
            let mad = calculate_mad(&y_outliers);
            println!("  MAD: {:.3}, Expected delta: {:.3}", mad, mad * 1.35);
        }
    }
    
    // Test 3: Train and predict with both loss types
    println!("\nTest 3: Training with auto-selected loss");
    
    // Generate synthetic data with outliers
    let n = 100;
    let mut x_train: Vec<Vec<f64>> = Vec::new();
    let mut y_train: Vec<f64> = Vec::new();
    
    for i in 0..n {
        let x_val = i as f64 / 10.0;
        x_train.push(vec![x_val]);
        
        // y = 2x + 1, with some outliers
        let y_val = if i % 20 == 0 {
            2.0 * x_val + 1.0 + 50.0  // Outlier
        } else {
            2.0 * x_val + 1.0 + (i as f64 % 3.0 - 1.0) * 0.1  // Normal noise
        };
        y_train.push(y_val);
    }
    
    let mut model = PKBoostRegressor::auto(&x_train, &y_train);
    println!("  Loss type: {:?}", model.loss_type);
    
    model.fit(&x_train, &y_train, None, false)?;
    println!("  Training complete with {} trees", model.trees.len());
    
    let predictions = model.predict(&x_train)?;
    let rmse = calculate_rmse(&y_train, &predictions);
    let mae = calculate_mae(&y_train, &predictions);
    println!("  RMSE: {:.4}, MAE: {:.4}", rmse, mae);
    
    println!("\n=== All tests completed ===");
    Ok(())
}
