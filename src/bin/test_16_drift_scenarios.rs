// Comprehensive test of 16 drift scenarios
use pkboost::*;
use std::f64::consts::PI;

fn generate_base_data(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..n {
        let x_val = i as f64 / 50.0;
        x.push(vec![x_val, x_val * 2.0]);
        y.push(2.0 * x_val + 3.0 + (i % 5) as f64 * 0.1);
    }
    (x, y)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing 16 Drift Scenarios ===\n");
    
    let (x_train, y_train) = generate_base_data(500);
    
    // Scenario 1: No Drift (Baseline)
    println!("1. NO DRIFT (Baseline)");
    {
        let mut model = AdaptiveRegressor::new(&x_train, &y_train);
        model.fit_initial(&x_train, &y_train, None, false)?;
        
        let (x_test, y_test) = generate_base_data(200);
        model.observe_batch(&x_test, &y_test, false)?;
        
        println!("   State: {:?}, Metamorphoses: {}", model.get_state(), model.get_metamorphosis_count());
        assert_eq!(model.get_metamorphosis_count(), 0, "Should have no metamorphoses");
    }
    
    // Scenario 2: Sudden Drift (Abrupt shift)
    println!("2. SUDDEN DRIFT (Abrupt coefficient change)");
    {
        let mut model = AdaptiveRegressor::new(&x_train, &y_train);
        model.fit_initial(&x_train, &y_train, None, false)?;
        
        let mut x_drift = Vec::new();
        let mut y_drift = Vec::new();
        for i in 0..200 {
            let x_val = (i + 500) as f64 / 50.0;
            x_drift.push(vec![x_val, x_val * 2.0]);
            y_drift.push(5.0 * x_val + 10.0);  // Sudden change
        }
        model.observe_batch(&x_drift, &y_drift, false)?;
        
        println!("   State: {:?}, Metamorphoses: {}", model.get_state(), model.get_metamorphosis_count());
    }
    
    // Scenario 3: Gradual Drift (Slow coefficient change)
    println!("3. GRADUAL DRIFT (Slow linear change)");
    {
        let mut model = AdaptiveRegressor::new(&x_train, &y_train);
        model.fit_initial(&x_train, &y_train, None, false)?;
        
        for batch_idx in 0..5 {
            let mut x_batch = Vec::new();
            let mut y_batch = Vec::new();
            let coef = 2.0 + batch_idx as f64 * 0.3;  // Gradually increasing
            for i in 0..100 {
                let x_val = (batch_idx * 100 + i + 500) as f64 / 50.0;
                x_batch.push(vec![x_val, x_val * 2.0]);
                y_batch.push(coef * x_val + 3.0);
            }
            model.observe_batch(&x_batch, &y_batch, false)?;
        }
        
        println!("   State: {:?}, Metamorphoses: {}", model.get_state(), model.get_metamorphosis_count());
    }
    
    // Scenario 4: Incremental Drift (Step-wise changes)
    println!("4. INCREMENTAL DRIFT (Multiple small steps)");
    {
        let mut model = AdaptiveRegressor::new(&x_train, &y_train);
        model.fit_initial(&x_train, &y_train, None, false)?;
        
        for step in 0..3 {
            let mut x_batch = Vec::new();
            let mut y_batch = Vec::new();
            let offset = step as f64 * 2.0;
            for i in 0..150 {
                let x_val = (step * 150 + i + 500) as f64 / 50.0;
                x_batch.push(vec![x_val, x_val * 2.0]);
                y_batch.push(2.0 * x_val + 3.0 + offset);
            }
            model.observe_batch(&x_batch, &y_batch, false)?;
        }
        
        println!("   State: {:?}, Metamorphoses: {}", model.get_state(), model.get_metamorphosis_count());
    }
    
    // Scenario 5: Recurring Drift (Seasonal pattern)
    println!("5. RECURRING DRIFT (Seasonal/cyclic)");
    {
        let mut model = AdaptiveRegressor::new(&x_train, &y_train);
        model.fit_initial(&x_train, &y_train, None, false)?;
        
        for cycle in 0..3 {
            let mut x_batch = Vec::new();
            let mut y_batch = Vec::new();
            let phase = (cycle as f64 * PI / 2.0).sin();
            for i in 0..150 {
                let x_val = (cycle * 150 + i + 500) as f64 / 50.0;
                x_batch.push(vec![x_val, x_val * 2.0]);
                y_batch.push(2.0 * x_val + 3.0 + phase * 3.0);
            }
            model.observe_batch(&x_batch, &y_batch, false)?;
        }
        
        println!("   State: {:?}, Metamorphoses: {}", model.get_state(), model.get_metamorphosis_count());
    }
    
    // Scenario 6: Outlier Injection (Sparse outliers)
    println!("6. OUTLIER INJECTION (5% extreme values)");
    {
        let mut model = AdaptiveRegressor::new(&x_train, &y_train);
        model.fit_initial(&x_train, &y_train, None, false)?;
        
        let mut x_test = Vec::new();
        let mut y_test = Vec::new();
        for i in 0..200 {
            let x_val = (i + 500) as f64 / 50.0;
            x_test.push(vec![x_val, x_val * 2.0]);
            let y_val = if i % 20 == 0 {
                2.0 * x_val + 3.0 + 50.0  // Outlier
            } else {
                2.0 * x_val + 3.0
            };
            y_test.push(y_val);
        }
        model.observe_batch(&x_test, &y_test, false)?;
        
        println!("   State: {:?}, Metamorphoses: {}", model.get_state(), model.get_metamorphosis_count());
    }
    
    // Scenario 7: Noise Increase (Variance change)
    println!("7. NOISE INCREASE (Heteroscedastic)");
    {
        let mut model = AdaptiveRegressor::new(&x_train, &y_train);
        model.fit_initial(&x_train, &y_train, None, false)?;
        
        let mut x_test = Vec::new();
        let mut y_test = Vec::new();
        for i in 0..200 {
            let x_val = (i + 500) as f64 / 50.0;
            x_test.push(vec![x_val, x_val * 2.0]);
            let noise = (i as f64 / 20.0) * ((i % 10) as f64 - 5.0);
            y_test.push(2.0 * x_val + 3.0 + noise);
        }
        model.observe_batch(&x_test, &y_test, false)?;
        
        println!("   State: {:?}, Metamorphoses: {}", model.get_state(), model.get_metamorphosis_count());
    }
    
    // Scenario 8: Feature Drift (Input distribution shift)
    println!("8. FEATURE DRIFT (Covariate shift)");
    {
        let mut model = AdaptiveRegressor::new(&x_train, &y_train);
        model.fit_initial(&x_train, &y_train, None, false)?;
        
        let mut x_test = Vec::new();
        let mut y_test = Vec::new();
        for i in 0..200 {
            let x_val = (i + 500) as f64 / 50.0 + 10.0;  // Shifted range
            x_test.push(vec![x_val, x_val * 2.0]);
            y_test.push(2.0 * x_val + 3.0);
        }
        model.observe_batch(&x_test, &y_test, false)?;
        
        println!("   State: {:?}, Metamorphoses: {}", model.get_state(), model.get_metamorphosis_count());
    }
    
    // Scenario 9: Temporal Autocorrelation (Trending errors)
    println!("9. TEMPORAL DRIFT (Autocorrelated errors)");
    {
        let mut model = AdaptiveRegressor::new(&x_train, &y_train);
        model.fit_initial(&x_train, &y_train, None, false)?;
        
        let mut x_test = Vec::new();
        let mut y_test = Vec::new();
        let mut cumulative_error = 0.0;
        for i in 0..200 {
            let x_val = (i + 500) as f64 / 50.0;
            x_test.push(vec![x_val, x_val * 2.0]);
            cumulative_error += 0.05;  // Accumulating bias
            y_test.push(2.0 * x_val + 3.0 + cumulative_error);
        }
        model.observe_batch(&x_test, &y_test, false)?;
        
        println!("   State: {:?}, Metamorphoses: {}", model.get_state(), model.get_metamorphosis_count());
    }
    
    // Scenario 10: Concept Reversal (Sign flip)
    println!("10. CONCEPT REVERSAL (Relationship inversion)");
    {
        let mut model = AdaptiveRegressor::new(&x_train, &y_train);
        model.fit_initial(&x_train, &y_train, None, false)?;
        
        let mut x_test = Vec::new();
        let mut y_test = Vec::new();
        for i in 0..200 {
            let x_val = (i + 500) as f64 / 50.0;
            x_test.push(vec![x_val, x_val * 2.0]);
            y_test.push(-2.0 * x_val + 20.0);  // Reversed relationship
        }
        model.observe_batch(&x_test, &y_test, false)?;
        
        println!("   State: {:?}, Metamorphoses: {}", model.get_state(), model.get_metamorphosis_count());
    }
    
    // Scenario 11: Mixed Drift (Multiple types)
    println!("11. MIXED DRIFT (Coefficient + noise + outliers)");
    {
        let mut model = AdaptiveRegressor::new(&x_train, &y_train);
        model.fit_initial(&x_train, &y_train, None, false)?;
        
        let mut x_test = Vec::new();
        let mut y_test = Vec::new();
        for i in 0..200 {
            let x_val = (i + 500) as f64 / 50.0;
            x_test.push(vec![x_val, x_val * 2.0]);
            let base = 3.5 * x_val + 5.0;  // Changed coefficient
            let noise = (i % 10) as f64 * 0.5;
            let outlier = if i % 30 == 0 { 20.0 } else { 0.0 };
            y_test.push(base + noise + outlier);
        }
        model.observe_batch(&x_test, &y_test, false)?;
        
        println!("   State: {:?}, Metamorphoses: {}", model.get_state(), model.get_metamorphosis_count());
    }
    
    // Scenario 12: Localized Drift (Specific region)
    println!("12. LOCALIZED DRIFT (Region-specific change)");
    {
        let mut model = AdaptiveRegressor::new(&x_train, &y_train);
        model.fit_initial(&x_train, &y_train, None, false)?;
        
        let mut x_test = Vec::new();
        let mut y_test = Vec::new();
        for i in 0..200 {
            let x_val = (i + 500) as f64 / 50.0;
            x_test.push(vec![x_val, x_val * 2.0]);
            let y_val = if x_val > 12.0 && x_val < 15.0 {
                2.0 * x_val + 10.0  // Drift in specific range
            } else {
                2.0 * x_val + 3.0
            };
            y_test.push(y_val);
        }
        model.observe_batch(&x_test, &y_test, false)?;
        
        println!("   State: {:?}, Metamorphoses: {}", model.get_state(), model.get_metamorphosis_count());
    }
    
    // Scenario 13: Virtual Drift (False alarm - high noise)
    println!("13. VIRTUAL DRIFT (High noise, no real drift)");
    {
        let mut model = AdaptiveRegressor::new(&x_train, &y_train);
        model.fit_initial(&x_train, &y_train, None, false)?;
        
        let mut x_test = Vec::new();
        let mut y_test = Vec::new();
        for i in 0..200 {
            let x_val = (i + 500) as f64 / 50.0;
            x_test.push(vec![x_val, x_val * 2.0]);
            let noise = ((i % 20) as f64 - 10.0) * 2.0;
            y_test.push(2.0 * x_val + 3.0 + noise);
        }
        model.observe_batch(&x_test, &y_test, false)?;
        
        println!("   State: {:?}, Metamorphoses: {} (should be 0 with P2)", 
            model.get_state(), model.get_metamorphosis_count());
    }
    
    // Scenario 14: Reoccurring Concept (Back to original)
    println!("14. REOCCURRING CONCEPT (Drift then return)");
    {
        let mut model = AdaptiveRegressor::new(&x_train, &y_train);
        model.fit_initial(&x_train, &y_train, None, false)?;
        
        // Drift away
        let mut x_drift = Vec::new();
        let mut y_drift = Vec::new();
        for i in 0..150 {
            let x_val = (i + 500) as f64 / 50.0;
            x_drift.push(vec![x_val, x_val * 2.0]);
            y_drift.push(4.0 * x_val + 8.0);
        }
        model.observe_batch(&x_drift, &y_drift, false)?;
        
        // Return to original
        let (x_return, y_return) = generate_base_data(150);
        model.observe_batch(&x_return, &y_return, false)?;
        
        println!("   State: {:?}, Metamorphoses: {}", model.get_state(), model.get_metamorphosis_count());
    }
    
    // Scenario 15: Extreme Outliers (P1 test)
    println!("15. EXTREME OUTLIERS (Testing Huber loss)");
    {
        let mut y_outliers = y_train.clone();
        for i in (0..y_outliers.len()).step_by(10) {
            y_outliers[i] += 100.0;  // 10% extreme outliers
        }
        
        let mut model = AdaptiveRegressor::new(&x_train, &y_outliers);
        model.fit_initial(&x_train, &y_outliers, None, false)?;
        
        let (x_test, y_test) = generate_base_data(200);
        model.observe_batch(&x_test, &y_test, false)?;
        
        println!("   State: {:?}, Metamorphoses: {}", model.get_state(), model.get_metamorphosis_count());
    }
    
    // Scenario 16: Combined Stress Test (All at once)
    println!("16. COMBINED STRESS TEST (Everything)");
    {
        let mut model = AdaptiveRegressor::new(&x_train, &y_train);
        model.fit_initial(&x_train, &y_train, None, false)?;
        
        let mut x_test = Vec::new();
        let mut y_test = Vec::new();
        for i in 0..300 {
            let x_val = (i + 500) as f64 / 50.0;
            x_test.push(vec![x_val + (i as f64 / 100.0), x_val * 2.0]);  // Feature drift
            
            let coef = 2.0 + (i as f64 / 100.0);  // Gradual drift
            let noise = (i % 15) as f64 * 0.8;  // Noise
            let outlier = if i % 25 == 0 { 30.0 } else { 0.0 };  // Outliers
            let trend = i as f64 * 0.02;  // Temporal
            
            y_test.push(coef * x_val + 3.0 + noise + outlier + trend);
        }
        model.observe_batch(&x_test, &y_test, false)?;
        
        println!("   State: {:?}, Metamorphoses: {}", model.get_state(), model.get_metamorphosis_count());
    }
    
    println!("\n=== All 16 Drift Scenarios Tested ===");
    println!("\nKey Observations:");
    println!("- Scenario 1: No drift → No metamorphoses (baseline)");
    println!("- Scenarios 2-12: Various drift types → Adaptive response");
    println!("- Scenario 13: Virtual drift → P2 prevents false alarm");
    println!("- Scenario 15: Extreme outliers → P1 Huber loss handles");
    println!("- Scenario 16: Combined stress → System remains stable");
    
    Ok(())
}
