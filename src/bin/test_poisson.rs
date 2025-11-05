use pkboost::*;
use rand::Rng;

fn generate_poisson_data(n_samples: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = rand::thread_rng();
    let mut x = Vec::new();
    let mut y = Vec::new();
    
    for _ in 0..n_samples {
        let x1: f64 = rng.gen_range(-2.0..2.0);
        let x2: f64 = rng.gen_range(-2.0..2.0);
        
        // True model: E[Y] = exp(0.5 + 0.3*x1 + 0.7*x2)
        let lambda = (0.5 + 0.3 * x1 + 0.7 * x2).exp();
        
        // Sample from Poisson distribution (approximate with normal for large lambda)
        let count = if lambda < 10.0 {
            // Use simple rejection sampling for small lambda
            let mut k = 0;
            let mut p = (-lambda).exp();
            let mut s = p;
            let u: f64 = rng.gen();
            while u > s {
                k += 1;
                p *= lambda / k as f64;
                s += p;
            }
            k as f64
        } else {
            // Normal approximation for large lambda
            let normal: f64 = rng.gen_range(0.0..1.0);
            (lambda + lambda.sqrt() * (2.0 * std::f64::consts::PI * normal).cos()).max(0.0)
        };
        
        x.push(vec![x1, x2]);
        y.push(count);
    }
    
    (x, y)
}

fn main() -> Result<(), String> {
    println!("=== Poisson Regression Test ===\n");
    
    // Generate synthetic count data
    println!("Generating synthetic Poisson data...");
    let (x_train, y_train) = generate_poisson_data(5000);
    let (x_test, y_test) = generate_poisson_data(1000);
    
    let y_mean = y_train.iter().sum::<f64>() / y_train.len() as f64;
    let y_max = y_train.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("Train: {} samples, Y mean: {:.2}, Y max: {:.0}", x_train.len(), y_mean, y_max);
    println!("Test: {} samples\n", x_test.len());
    
    // Train with MSE (baseline)
    println!("=== Training with MSE Loss ===");
    let mut model_mse = PKBoostRegressor::auto(&x_train, &y_train);
    model_mse.n_estimators = 200;
    model_mse.fit(&x_train, &y_train, None, false)?;
    
    let pred_mse = model_mse.predict(&x_test)?;
    let rmse_mse = calculate_rmse(&y_test, &pred_mse);
    let mae_mse = calculate_mae(&y_test, &pred_mse);
    
    println!("MSE Model - RMSE: {:.4}, MAE: {:.4}", rmse_mse, mae_mse);
    
    // Train with Poisson loss
    println!("\n=== Training with Poisson Loss ===");
    let mut model_poisson = PKBoostRegressor::auto(&x_train, &y_train)
        .with_loss(RegressionLossType::Poisson);
    model_poisson.n_estimators = 200;
    model_poisson.fit(&x_train, &y_train, None, false)?;
    
    let pred_poisson = model_poisson.predict(&x_test)?;
    let rmse_poisson = calculate_rmse(&y_test, &pred_poisson);
    let mae_poisson = calculate_mae(&y_test, &pred_poisson);
    
    println!("Poisson Model - RMSE: {:.4}, MAE: {:.4}", rmse_poisson, mae_poisson);
    
    // Calculate Poisson deviance
    let poisson_deviance = PoissonLoss::loss(&y_test, &pred_poisson.iter().map(|p| p.ln()).collect::<Vec<_>>());
    println!("Poisson Deviance: {:.4}", poisson_deviance);
    
    // Compare predictions
    println!("\n=== Sample Predictions ===");
    println!("True | MSE  | Poisson");
    println!("-----|------|--------");
    for i in 0..10 {
        println!("{:4.0} | {:4.1} | {:4.1}", y_test[i], pred_mse[i], pred_poisson[i]);
    }
    
    // Summary
    println!("\n=== Results Summary ===");
    let improvement = ((rmse_mse - rmse_poisson) / rmse_mse) * 100.0;
    println!("RMSE Improvement: {:+.1}%", improvement);
    
    if rmse_poisson < rmse_mse {
        println!("✅ Poisson loss performs better for count data!");
    } else {
        println!("⚠️  MSE performed better (may need more tuning)");
    }
    
    Ok(())
}
