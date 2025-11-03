use pkboost::*;
use std::error::Error;

fn load_regression_data(path: &str, target_col: &str) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let target_idx = headers.iter().position(|h| h == target_col)
        .ok_or(format!("'{}' column not found", target_col))?;

    let mut features = Vec::new();
    let mut labels = Vec::new();

    for result in reader.records() {
        let record = result?;
        let mut row = Vec::new();
        for (i, value) in record.iter().enumerate() {
            if i == target_idx {
                labels.push(value.parse()?);
            } else {
                row.push(if value.is_empty() { f64::NAN } else { value.parse()? });
            }
        }
        features.push(row);
    }
    Ok((features, labels))
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("\n=== PKBoost Regressor Test (California Housing) ===\n");
    
    // Load real dataset
    println!("Loading California Housing dataset...");
    let (x_train, y_train) = load_regression_data("data/housing_train.csv", "Target")?;
    let (x_val, y_val) = load_regression_data("data/housing_val.csv", "Target")?;
    let (x_test, y_test) = load_regression_data("data/housing_test.csv", "Target")?;
    
    println!("Train: {} samples, {} features", x_train.len(), x_train[0].len());
    println!("Val: {} samples", x_val.len());
    println!("Test: {} samples\n", x_test.len());
    
    let mut model = PKBoostRegressor::auto(&x_train, &y_train);
    model.fit(&x_train, &y_train, Some((&x_val, &y_val)), true)?;
    
    // Evaluate on test set
    let test_preds = model.predict(&x_test)?;
    let test_rmse = calculate_rmse(&y_test, &test_preds);
    let test_mae = calculate_mae(&y_test, &test_preds);
    let test_r2 = calculate_r2(&y_test, &test_preds);
    
    println!("\n=== Test Set Results ===");
    println!("RMSE: {:.4}", test_rmse);
    println!("MAE: {:.4}", test_mae);
    println!("R²: {:.4}", test_r2);
    
    // Show some predictions vs actual
    println!("\n=== Sample Predictions ===");
    for i in 0..5.min(test_preds.len()) {
        println!("Actual: {:.2}, Predicted: {:.2}, Error: {:.2}", 
                 y_test[i], test_preds[i], (y_test[i] - test_preds[i]).abs());
    }
    
    Ok(())
}
