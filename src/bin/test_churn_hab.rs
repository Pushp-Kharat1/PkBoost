use pkboost::*;
use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;

fn load_csv_data(path: &str) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
    let headers = rdr.headers()?.clone();
    let target_idx = headers.iter().position(|h| h == "Class").ok_or("Class column not found")?;

    let mut x = Vec::new();
    let mut y = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let mut features = Vec::new();
        for (i, val) in record.iter().enumerate() {
            let v: f64 = val.parse()?;
            if i == target_idx {
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
    println!("=== E-Commerce Churn: HAB Ensemble Test ===\n");

    let (x_train, y_train) = load_csv_data("data/churn_train.csv")?;
    let (x_val, y_val) = load_csv_data("data/churn_val.csv")?;
    let (x_test, y_test) = load_csv_data("data/churn_test.csv")?;

    println!("Train: {} | Val: {} | Test: {}\n", x_train.len(), x_val.len(), x_test.len());

    // Baseline
    println!("=== Baseline (Single Model) ===");
    let mut baseline = OptimizedPKBoostShannon::auto(&x_train, &y_train);
    baseline.fit(&x_train, &y_train, Some((&x_val, &y_val)), true)?;
    let baseline_probs = baseline.predict_proba(&x_test)?;
    let baseline_pr_auc = calculate_pr_auc(&y_test, &baseline_probs);

    // Feature Bagging Ensemble
    println!("\n=== Feature Bagging Ensemble ===");
    let n_features = x_train[0].len();
    
    println!("Training specialist 1 (features 0-{})...", n_features*2/3);
    let feat1: Vec<usize> = (0..n_features*2/3).collect();
    let x1: Vec<Vec<f64>> = x_train.iter().map(|row| feat1.iter().map(|&i| row[i]).collect()).collect();
    let x1_test: Vec<Vec<f64>> = x_test.iter().map(|row| feat1.iter().map(|&i| row[i]).collect()).collect();
    let mut spec1 = OptimizedPKBoostShannon::auto(&x1, &y_train);
    spec1.fit(&x1, &y_train, None, false)?;
    
    println!("Training specialist 2 (features {}-{})...", n_features/3, n_features-1);
    let feat2: Vec<usize> = (n_features/3..n_features).collect();
    let x2: Vec<Vec<f64>> = x_train.iter().map(|row| feat2.iter().map(|&i| row[i]).collect()).collect();
    let x2_test: Vec<Vec<f64>> = x_test.iter().map(|row| feat2.iter().map(|&i| row[i]).collect()).collect();
    let mut spec2 = OptimizedPKBoostShannon::auto(&x2, &y_train);
    spec2.fit(&x2, &y_train, None, false)?;
    
    println!("Training specialist 3 (random subset)...");
    let feat3: Vec<usize> = (0..n_features).step_by(2).collect();
    let x3: Vec<Vec<f64>> = x_train.iter().map(|row| feat3.iter().map(|&i| row[i]).collect()).collect();
    let x3_test: Vec<Vec<f64>> = x_test.iter().map(|row| feat3.iter().map(|&i| row[i]).collect()).collect();
    let mut spec3 = OptimizedPKBoostShannon::auto(&x3, &y_train);
    spec3.fit(&x3, &y_train, None, false)?;
    
    let p1 = spec1.predict_proba(&x1_test)?;
    let p2 = spec2.predict_proba(&x2_test)?;
    let p3 = spec3.predict_proba(&x3_test)?;
    let hab_probs: Vec<f64> = (0..p1.len()).map(|i| (p1[i] + p2[i] + p3[i]) / 3.0).collect();
    let hab_pr_auc = calculate_pr_auc(&y_test, &hab_probs);

    println!("\n=== RESULTS ===");
    println!("Baseline PR-AUC:        {:.4}", baseline_pr_auc);
    println!("Feature Bagging PR-AUC: {:.4}", hab_pr_auc);
    println!("Improvement:            {:.1}%", ((hab_pr_auc - baseline_pr_auc) / baseline_pr_auc) * 100.0);

    Ok(())
}
