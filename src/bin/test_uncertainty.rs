// Test uncertainty quantification
use pkboost::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Uncertainty Quantification ===\n");
    
    // Generate training data
    let n_train = 200;
    let mut x_train: Vec<Vec<f64>> = Vec::new();
    let mut y_train: Vec<f64> = Vec::new();
    
    for i in 0..n_train {
        let x = i as f64 / 20.0;
        x_train.push(vec![x]);
        y_train.push(2.0 * x + 1.0 + (i % 5) as f64 * 0.1);
    }
    
    // Train model
    let mut model = PKBoostRegressor::auto(&x_train, &y_train);
    model.fit(&x_train, &y_train, None, false)?;
    
    println!("Test 1: Basic uncertainty prediction");
    let x_test: Vec<Vec<f64>> = vec![vec![5.0], vec![10.0], vec![15.0]];
    let (predictions, uncertainties) = model.predict_with_uncertainty(&x_test)?;
    
    println!("  Sample predictions with uncertainty:");
    for (i, (pred, unc)) in predictions.iter().zip(uncertainties.iter()).enumerate() {
        println!("    x={:.1}: pred={:.3} ± {:.3}", x_test[i][0], pred, unc);
    }
    
    assert_eq!(predictions.len(), 3);
    assert_eq!(uncertainties.len(), 3);
    println!("  ✓ Returns correct dimensions\n");
    
    println!("Test 2: Interpolation vs extrapolation");
    // Interpolation (within training range)
    let x_interp: Vec<Vec<f64>> = vec![vec![5.0]];
    let (pred_interp, unc_interp) = model.predict_with_uncertainty(&x_interp)?;
    
    // Extrapolation (outside training range)
    let x_extrap: Vec<Vec<f64>> = vec![vec![20.0]];
    let (pred_extrap, unc_extrap) = model.predict_with_uncertainty(&x_extrap)?;
    
    println!("  Interpolation (x=5.0): uncertainty = {:.4}", unc_interp[0]);
    println!("  Extrapolation (x=20.0): uncertainty = {:.4}", unc_extrap[0]);
    
    // Extrapolation typically has higher uncertainty
    if unc_extrap[0] > unc_interp[0] {
        println!("  ✓ Extrapolation has higher uncertainty (expected)\n");
    } else {
        println!("  Note: Extrapolation uncertainty not higher (can happen)\n");
    }
    
    println!("Test 3: Uncertainty is non-negative");
    let x_many: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64 / 5.0]).collect();
    let (_, uncertainties) = model.predict_with_uncertainty(&x_many)?;
    
    let all_positive = uncertainties.iter().all(|&u| u >= 0.0);
    assert!(all_positive, "All uncertainties should be non-negative");
    
    let avg_uncertainty = uncertainties.iter().sum::<f64>() / uncertainties.len() as f64;
    let max_uncertainty = uncertainties.iter().fold(0.0f64, |a, &b| a.max(b));
    let min_uncertainty = uncertainties.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    
    println!("  Average uncertainty: {:.4}", avg_uncertainty);
    println!("  Min uncertainty: {:.4}", min_uncertainty);
    println!("  Max uncertainty: {:.4}", max_uncertainty);
    println!("  ✓ All uncertainties non-negative\n");
    
    println!("Test 4: Consistency with regular predict");
    let x_test: Vec<Vec<f64>> = vec![vec![3.0], vec![7.0]];
    let regular_preds = model.predict(&x_test)?;
    let (uncertainty_preds, _) = model.predict_with_uncertainty(&x_test)?;
    
    for (i, (reg, unc)) in regular_preds.iter().zip(uncertainty_preds.iter()).enumerate() {
        let diff = (reg - unc).abs();
        println!("  Sample {}: regular={:.4}, with_unc={:.4}, diff={:.6}", i, reg, unc, diff);
        assert!(diff < 0.001, "Predictions should match");
    }
    println!("  ✓ Predictions consistent\n");
    
    println!("Test 5: AdaptiveRegressor wrapper");
    let mut adaptive = AdaptiveRegressor::new(&x_train, &y_train);
    adaptive.fit_initial(&x_train, &y_train, None, false)?;
    
    let x_test: Vec<Vec<f64>> = vec![vec![5.0]];
    let (pred, unc) = adaptive.predict_with_uncertainty(&x_test)?;
    
    println!("  Prediction: {:.3} ± {:.3}", pred[0], unc[0]);
    println!("  ✓ AdaptiveRegressor wrapper works\n");
    
    println!("=== All uncertainty tests passed ===");
    Ok(())
}
