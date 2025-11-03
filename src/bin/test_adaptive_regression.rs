use mimalloc::MiMalloc;
use pkboost::*;
use std::error::Error;
use std::fs::File;
use std::io::Write;
use rand::SeedableRng;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn load_data(path: &str, target_col: &str) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
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
    println!("\n=== ADAPTIVE REGRESSOR DRIFT TEST ===\n");
    
    let mut log_file = File::create("adaptive_regression_metrics.csv")?;
    writeln!(log_file, "observation,phase,rmse,mae,r2,vuln_score,state,metamorphosis_count")?;
    
    // Load California Housing dataset
    println!("Loading California Housing dataset...");
    let (x_train, y_train) = load_data("data/housing_train.csv", "Target")?;
    let (x_val, y_val) = load_data("data/housing_val.csv", "Target")?;
    let (mut x_test, y_test) = load_data("data/housing_test.csv", "Target")?;
    
    println!("Train: {} samples, {} features", x_train.len(), x_train[0].len());
    println!("Val: {} samples", x_val.len());
    println!("Test: {} samples\n", x_test.len());
    
    // Create adaptive regressor
    let mut model = AdaptiveRegressor::new(&x_train, &y_train);
    
    println!("⚠️  METAMORPHOSIS ENABLED (will trigger on drift detection) ⚠️\n");
    
    // Initial training
    println!("Initial training...");
    model.fit_initial(&x_train, &y_train, Some((&x_val, &y_val)), false)?;
    println!("Initial training complete. Model ready for streaming.\n");
    
    let batch_size = 1000;
    let mut total_obs = 0;
    
    // Phase 1: Normal data
    println!("=== PHASE 1: NORMAL DATA ===");
    let phase1_end = (x_test.len() / 2).min(5000);
    for batch_start in (0..phase1_end).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(phase1_end);
        let x_batch: Vec<Vec<f64>> = x_test[batch_start..batch_end].to_vec();
        let y_batch: Vec<f64> = y_test[batch_start..batch_end].to_vec();
        
        let preds = model.predict(&x_batch)?;
        let rmse = calculate_rmse(&y_batch, &preds);
        let mae = calculate_mae(&y_batch, &preds);
        let r2 = calculate_r2(&y_batch, &preds);
        
        total_obs += x_batch.len();
        
        writeln!(
            log_file,
            "{},normal,{:.4},{:.4},{:.4},{:.4},{:?},{}",
            total_obs, rmse, mae, r2,
            model.get_vulnerability_score(),
            model.get_state(),
            model.get_metamorphosis_count()
        )?;
        
        println!("Obs {}: RMSE={:.4}, MAE={:.4}, R²={:.4}, State={:?}",
                 total_obs, rmse, mae, r2, model.get_state());
        
        model.observe_batch(&x_batch, &y_batch, false)?;
    }
    
    // Apply drift: Add noise and scale features
    println!("\n=== APPLYING DRIFT ===");
    println!("  - Adding Gaussian noise (σ=0.5)");
    println!("  - Scaling features 0,1,2 by 2.0x");
    println!("  - Shifting features 3,4 by +1.0\n");
    
    use rand::Rng;
    // Seeded RNG for reproducibility
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    
    for row in x_test.iter_mut() {
        // Add noise to all features
        for val in row.iter_mut() {
            *val += rng.gen_range(-0.5..0.5);
        }
        // Scale some features
        if row.len() > 2 {
            row[0] *= 2.0;
            row[1] *= 2.0;
            row[2] *= 2.0;
        }
        // Shift some features
        if row.len() > 4 {
            row[3] += 1.0;
            row[4] += 1.0;
        }
    }
    
    // Phase 2: Drifted data
    println!("=== PHASE 2: DRIFTED DATA ===");
    let phase2_start = phase1_end;
    let phase2_end = x_test.len();
    
    for batch_start in (phase2_start..phase2_end).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(phase2_end);
        let x_batch: Vec<Vec<f64>> = x_test[batch_start..batch_end].to_vec();
        let y_batch: Vec<f64> = y_test[batch_start..batch_end].to_vec();
        
        let preds = model.predict(&x_batch)?;
        let rmse = calculate_rmse(&y_batch, &preds);
        let mae = calculate_mae(&y_batch, &preds);
        let r2 = calculate_r2(&y_batch, &preds);
        
        total_obs += x_batch.len();
        
        writeln!(
            log_file,
            "{},drifted,{:.4},{:.4},{:.4},{:.4},{:?},{}",
            total_obs, rmse, mae, r2,
            model.get_vulnerability_score(),
            model.get_state(),
            model.get_metamorphosis_count()
        )?;
        
        println!("Obs {}: RMSE={:.4}, MAE={:.4}, R²={:.4}, State={:?}, Vuln={:.4}",
                 total_obs, rmse, mae, r2, model.get_state(), model.get_vulnerability_score());
        
        model.observe_batch(&x_batch, &y_batch, true)?;
    }
    
    println!("\n=== FINAL SUMMARY ===");
    println!("Total observations: {}", total_obs);
    println!("Metamorphoses triggered: {}", model.get_metamorphosis_count());
    println!("Final state: {:?}", model.get_state());
    println!("\nMetrics saved to: adaptive_regression_metrics.csv");
    
    Ok(())
}
