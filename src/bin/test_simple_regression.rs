// Simple regression test to verify PKBoost works
use pkboost::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Simple Regression Test ===\n");
    
    // Generate simple linear data: y = 2x + 3
    let mut x_train: Vec<Vec<f64>> = Vec::new();
    let mut y_train: Vec<f64> = Vec::new();
    
    for i in 0..1000 {
        let x = i as f64;
        x_train.push(vec![x]);
        y_train.push(2.0 * x + 3.0 + (i % 5) as f64 * 0.1);
    }
    
    println!("Data: y = 2x + 3 (with small noise)");
    println!("  X range: [0, 999]");
    println!("  Y range: [{:.1}, {:.1}]", y_train[0], y_train[999]);
    println!("  Y mean: {:.1}", y_train.iter().sum::<f64>() / y_train.len() as f64);
    
    // Train model
    let mut model = PKBoostRegressor::auto(&x_train, &y_train);
    println!("\nTraining PKBoost...");
    model.fit(&x_train, &y_train, None, true)?;
    
    // Test predictions
    let x_test = vec![vec![100.0], vec![500.0], vec![900.0]];
    let y_expected = vec![203.0, 1003.0, 1803.0];
    
    let predictions = model.predict(&x_test)?;
    
    println!("\nPredictions:");
    for (i, (pred, exp)) in predictions.iter().zip(y_expected.iter()).enumerate() {
        let error = (pred - exp).abs();
        let pct_error = (error / exp) * 100.0;
        println!("  x={:.0}: pred={:.2}, expected={:.2}, error={:.2} ({:.1}%)", 
            x_test[i][0], pred, exp, error, pct_error);
    }
    
    let avg_error: f64 = predictions.iter().zip(y_expected.iter())
        .map(|(p, e)| ((p - e).abs() / e) * 100.0)
        .sum::<f64>() / predictions.len() as f64;
    
    println!("\nAverage error: {:.1}%", avg_error);
    
    if avg_error < 5.0 {
        println!("✓ Model works correctly!");
    } else {
        println!("✗ Model has issues (>5% error)");
    }
    
    Ok(())
}
