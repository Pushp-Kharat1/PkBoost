// Test combined drift scoring
use pkboost::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Combined Drift Scoring ===\n");
    
    // Test scoring calculation
    println!("Test 1: Score calculation");
    let entropy_score = 0.8;
    let temporal_score = 0.6;
    let variance_score = 0.4;
    
    let combined = DRIFT_WEIGHT_ENTROPY * entropy_score + 
                   DRIFT_WEIGHT_TEMPORAL * temporal_score + 
                   DRIFT_WEIGHT_VARIANCE * variance_score;
    
    println!("  Entropy: {:.1} × {:.1} = {:.2}", entropy_score, DRIFT_WEIGHT_ENTROPY, entropy_score * DRIFT_WEIGHT_ENTROPY);
    println!("  Temporal: {:.1} × {:.1} = {:.2}", temporal_score, DRIFT_WEIGHT_TEMPORAL, temporal_score * DRIFT_WEIGHT_TEMPORAL);
    println!("  Variance: {:.1} × {:.1} = {:.2}", variance_score, DRIFT_WEIGHT_VARIANCE, variance_score * DRIFT_WEIGHT_VARIANCE);
    println!("  Combined: {:.3}", combined);
    
    let expected = 0.4 * 0.8 + 0.3 * 0.6 + 0.3 * 0.4;
    assert!((combined - expected).abs() < 0.001, "Score calculation incorrect");
    println!("  ✓ Calculation correct\n");
    
    // Test strategy selection
    println!("Test 2: Strategy selection");
    
    // Severe drift
    let severe_score = 0.75;
    if severe_score > SEVERE_DRIFT_THRESHOLD {
        println!("  Score {:.2} > {:.2} → Severe drift ({} trees)", 
            severe_score, SEVERE_DRIFT_THRESHOLD, TREES_SEVERE_DRIFT);
    }
    assert_eq!(TREES_SEVERE_DRIFT, 120);
    
    // Temporal drift
    let temporal = 0.55;
    if temporal > TEMPORAL_DRIFT_THRESHOLD {
        println!("  Temporal {:.2} > {:.2} → Temporal drift ({} trees)", 
            temporal, TEMPORAL_DRIFT_THRESHOLD, TREES_TEMPORAL_DRIFT);
    }
    assert_eq!(TREES_TEMPORAL_DRIFT, 90);
    
    // Variance drift
    let variance = 0.65;
    if variance > VARIANCE_DRIFT_THRESHOLD {
        println!("  Variance {:.2} > {:.2} → Variance drift ({} trees)", 
            variance, VARIANCE_DRIFT_THRESHOLD, TREES_VARIANCE_DRIFT);
    }
    assert_eq!(TREES_VARIANCE_DRIFT, 80);
    
    // Localized
    println!("  Otherwise → Localized drift ({} trees)", TREES_LOCALIZED_DRIFT);
    assert_eq!(TREES_LOCALIZED_DRIFT, 40);
    println!("  ✓ All strategies correct\n");
    
    // Test weight balance
    println!("Test 3: Weight balance");
    let total_weight = DRIFT_WEIGHT_ENTROPY + DRIFT_WEIGHT_TEMPORAL + DRIFT_WEIGHT_VARIANCE;
    println!("  Total weight: {:.1}", total_weight);
    assert!((total_weight - 1.0).abs() < 0.001, "Weights should sum to 1.0");
    println!("  ✓ Weights balanced\n");
    
    // Test with real model
    println!("Test 4: Integration test");
    let n = 500;
    let mut x_train: Vec<Vec<f64>> = Vec::new();
    let mut y_train: Vec<f64> = Vec::new();
    
    for i in 0..n {
        let x = i as f64 / 50.0;
        x_train.push(vec![x, x * 2.0]);
        y_train.push(2.0 * x + 1.0);
    }
    
    let mut model = AdaptiveRegressor::new(&x_train, &y_train);
    model.fit_initial(&x_train, &y_train, None, false)?;
    
    // Introduce drift
    let mut x_drift: Vec<Vec<f64>> = Vec::new();
    let mut y_drift: Vec<f64> = Vec::new();
    
    for i in 0..200 {
        let x = (i + 500) as f64 / 50.0;
        x_drift.push(vec![x, x * 2.0]);
        // Shift relationship
        y_drift.push(3.0 * x + 5.0);
    }
    
    model.observe_batch(&x_drift, &y_drift, true)?;
    
    println!("  Model state: {:?}", model.get_state());
    println!("  ✓ Integration test complete\n");
    
    println!("=== All combined scoring tests passed ===");
    Ok(())
}
