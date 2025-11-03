// Test drift resilience on Dry Bean dataset
use pkboost::MultiClassPKBoost;
use std::time::Instant;
use csv;
use rand::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Dry Bean Drift Resilience Test ===\n");
    
    let (x_train, y_train) = load_csv("data/drybean_train.csv")?;
    let (x_test, y_test) = load_csv("data/drybean_test.csv")?;
    
    println!("Dataset: {} train, {} test, {} features\n", x_train.len(), x_test.len(), x_train[0].len());
    
    // Train model
    println!("Training PKBoost...");
    let mut model = MultiClassPKBoost::new(7);
    model.fit(&x_train, &y_train, None, false)?;
    
    // Baseline performance
    let baseline_preds = model.predict(&x_test)?;
    let baseline_acc = accuracy(&baseline_preds, &y_test);
    let baseline_f1 = macro_f1(&baseline_preds, &y_test, 7);
    
    println!("\n=== Baseline (No Drift) ===");
    println!("Accuracy: {:.2}%", baseline_acc * 100.0);
    println!("Macro-F1: {:.4}", baseline_f1);
    
    // Test under different drift intensities
    let drift_levels = vec![0.5, 1.0, 2.0, 3.0];
    
    println!("\n=== Performance Under Drift ===");
    println!("{:<12} {:<12} {:<12} {:<15}", "Drift Level", "Accuracy", "Macro-F1", "Degradation");
    println!("{}", "-".repeat(55));
    
    for &drift in &drift_levels {
        let x_drifted = inject_drift(&x_test, drift);
        let preds = model.predict(&x_drifted)?;
        let acc = accuracy(&preds, &y_test);
        let f1 = macro_f1(&preds, &y_test, 7);
        let degradation = ((baseline_acc - acc) / baseline_acc * 100.0).abs();
        
        println!("{:<12.1} {:<12.2} {:<12.4} {:<15.1}%", 
                 drift, acc * 100.0, f1, degradation);
    }
    
    Ok(())
}

fn inject_drift(x: &[Vec<f64>], intensity: f64) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    let n_features = x[0].len();
    let n_drift_features = (n_features / 2).max(1); // Drift 50% of features
    
    x.iter().map(|row| {
        row.iter().enumerate().map(|(i, &val)| {
            if i < n_drift_features {
                // Add Gaussian noise
                let u1: f64 = rng.gen();
                let u2: f64 = rng.gen();
                let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                val + noise * intensity
            } else {
                val
            }
        }).collect()
    }).collect()
}

fn load_csv(path: &str) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn std::error::Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let n_cols = headers.len();
    
    let mut features = Vec::new();
    let mut labels = Vec::new();
    
    for result in reader.records() {
        let record = result?;
        let mut row = Vec::new();
        
        for (i, value) in record.iter().enumerate() {
            if i == n_cols - 1 {
                labels.push(value.parse()?);
            } else {
                row.push(value.parse()?);
            }
        }
        features.push(row);
    }
    
    Ok((features, labels))
}

fn accuracy(preds: &[usize], true_y: &[f64]) -> f64 {
    preds.iter().zip(true_y.iter())
        .filter(|(&pred, &true_val)| pred == true_val as usize)
        .count() as f64 / true_y.len() as f64
}

fn macro_f1(preds: &[usize], true_y: &[f64], n_classes: usize) -> f64 {
    let mut f1_sum = 0.0;
    for class in 0..n_classes {
        let tp = preds.iter().zip(true_y.iter())
            .filter(|(&p, &t)| p == class && t as usize == class)
            .count() as f64;
        let fp = preds.iter().zip(true_y.iter())
            .filter(|(&p, &t)| p == class && t as usize != class)
            .count() as f64;
        let fn_count = preds.iter().zip(true_y.iter())
            .filter(|(&p, &t)| p != class && t as usize == class)
            .count() as f64;
        
        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_count > 0.0 { tp / (tp + fn_count) } else { 0.0 };
        let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        
        f1_sum += f1;
    }
    f1_sum / n_classes as f64
}
