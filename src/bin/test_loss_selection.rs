use pkboost::{PKBoostRegressor, RegressionLossType, auto_params, DataStats};
use ndarray::{Array1, Array2};
use rand::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Sample data generation for testing loss selection
    let mut rng = thread_rng();
    let n_samples = 100;
    let n_features = 5;
    let mut x_train = Array2::zeros((n_samples, n_features));
    let mut y_train = Array1::zeros(n_samples);
    for i in 0..n_samples {
        for j in 0..n_features {
            x_train[[i, j]] = rng.gen_range(-1.0..1.0);
        }
        y_train[i] = x_train.row(i).dot(&Array1::from(vec![1.0, 2.0, -1.0, 0.5, -0.5])) + rng.gen_range(-0.1..0.1);
    }

    // Clean data stats and auto-params
    let stats_clean = DataStats::from_data(&x_train, &y_train);
    let params_clean = auto_params(&stats_clean, RegressionLossType::MSE)?;
    let model_clean = PKBoostRegressor::new(params_clean);

    // Train on clean data
    model_clean.fit(&x_train, &y_train)?;

    // Outlier-contaminated data (e.g., add outliers)
    let mut x_outliers = x_train.clone();
    let mut y_outliers = y_train.clone();
    for _ in 0..10 {
        let idx = rng.gen_range(0..n_samples);
        y_outliers[idx] += rng.gen_range(5.0..10.0);  // Outliers
    }
    let stats_outliers = DataStats::from_data(&x_outliers, &y_outliers);
    let params_outliers = auto_params(&stats_outliers, RegressionLossType::MSE)?;
    let model_outliers = PKBoostRegressor::new(params_outliers);
    model_outliers.fit(&x_outliers, &y_outliers)?;

    // Fixed matches with all variants covered
    println!("=== Loss Selection Test ===");
    match model_clean.loss_type {
        RegressionLossType::MSE => println!(" ✓ Selected MSE loss (expected for clean data)"),
        RegressionLossType::Huber { delta } => println!(" ✗ Selected Huber loss with delta={:.3} (unexpected for clean data)", delta),
        RegressionLossType::Poisson => println!(" ✗ Selected Poisson loss (unexpected for clean data)"),
    }

    match model_outliers.loss_type {
        RegressionLossType::MSE => println!(" ✗ Selected MSE loss (unexpected for outliers)"),
        RegressionLossType::Huber { delta } => println!(" ✓ Selected Huber loss with delta={:.3} (expected for outliers)", delta),
        RegressionLossType::Poisson => println!(" ✗ Selected Poisson loss (unexpected for outliers)"),
    }

    println!("Test complete.");
    Ok(())
}