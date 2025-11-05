use pkboost::*;
use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;

// Load CSV and separate features (X) and labels (y)
fn load_csv_data(path: &str, target_col: usize) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(file);

    let mut x = Vec::new();
    let mut y = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let mut features = Vec::new();

        for (i, val) in record.iter().enumerate() {
            let v: f64 = val.parse::<f64>()?;
            if i == target_col {
                y.push(v);
            } else {
                features.push(v);
            }
        }
        x.push(features);
    }

    Ok((x, y))
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== HAB Binary Classification Test (Real Dataset) ===\n");

    // Paths to your data files
    let train_path = "data/creditcard_train.csv";
    let val_path = "data/creditcard_val.csv";
    let test_path = "data/creditcard_test.csv";

    // Set your target column index here (0-based)
    let target_col = 5;

    println!("Loading training data from {train_path}...");
    let (x_train, y_train) = load_csv_data(train_path, target_col)?;

    println!("Loading validation data from {val_path}...");
    let (x_val, y_val) = load_csv_data(val_path, target_col)?;

    println!("Loading test data from {test_path}...");
    let (x_test, y_test) = load_csv_data(test_path, target_col)?;

    println!(
        "Train: {} | Val: {} | Test: {}\n",
        x_train.len(),
        x_val.len(),
        x_test.len()
    );

    // === Baseline Model ===
    println!("=== Training Baseline (Single PKBoost) ===");
    let mut baseline = OptimizedPKBoostShannon::auto(&x_train, &y_train);
    baseline.fit(&x_train, &y_train, Some((&x_val, &y_val)), true)?;

    let baseline_probs = baseline.predict_proba(&x_test)?;
    let baseline_pr_auc = calculate_pr_auc(&y_test, &baseline_probs);

    // === Class-Weighted Ensemble ===
    println!("\n=== Training Class-Weighted Ensemble ===");
    
    // Train specialist focusing on negatives (high weight on class 0)
    println!("Training negative-focused specialist...");
    let mut neg_specialist = OptimizedPKBoostShannon::auto(&x_train, &y_train);
    neg_specialist.scale_pos_weight = 0.1;  // Downweight positives
    neg_specialist.fit(&x_train, &y_train, None, false)?;
    
    // Train specialist focusing on positives (high weight on class 1)
    println!("Training positive-focused specialist...");
    let mut pos_specialist = OptimizedPKBoostShannon::auto(&x_train, &y_train);
    pos_specialist.scale_pos_weight = 10.0;  // Upweight positives
    pos_specialist.fit(&x_train, &y_train, None, false)?;
    
    // Ensemble prediction: weighted average
    let neg_probs = neg_specialist.predict_proba(&x_test)?;
    let pos_probs = pos_specialist.predict_proba(&x_test)?;
    let hab_probs_positive: Vec<f64> = neg_probs.iter().zip(pos_probs.iter())
        .map(|(&neg, &pos)| 0.3 * neg + 0.7 * pos)  // Favor positive specialist
        .collect();
    let hab_pr_auc = calculate_pr_auc(&y_test, &hab_probs_positive);

    // === Results ===
    println!("\n=== RESULTS ===");
    println!("Baseline PR-AUC:        {:.4}", baseline_pr_auc);
    println!("Class-Weighted PR-AUC:  {:.4}", hab_pr_auc);
    println!("Improvement:            {:.1}%", ((hab_pr_auc - baseline_pr_auc) / baseline_pr_auc) * 100.0);

    Ok(())
}
