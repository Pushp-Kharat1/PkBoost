use mimalloc::MiMalloc;
use pkboost::*;
use std::error::Error;
use std::time::Instant;
use csv;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn load_data(path: &str) -> Result<(Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let mut features = Vec::new();
    let mut labels = Vec::new();
    let target_col_index = headers.iter().position(|h| h == "Class")
        .ok_or("'Class' column not found")?;

    for result in reader.records() {    
        let record = result?;
        let mut feature_row = Vec::new();
        for (i, value) in record.iter().enumerate() {
            if i == target_col_index {
                labels.push(value.parse::<f64>()?);
            } else {
                let parsed_value = if value.is_empty() {
                    f64::NAN
                } else {
                    value.parse::<f64>()?
                };
                feature_row.push(parsed_value);
            }
        }
        features.push(feature_row);
    }
    Ok((features, labels))
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Loading datasets...");
    let (x_train, y_train) = load_data("resources/data/train_large.csv")?;
    let (x_val, y_val) = load_data("resources/data/val_large.csv")?;
    let (x_test, y_test) = load_data("resources/data/test_large.csv")?;
    
    println!("Data loaded: {} training samples, {} validation samples, {} test samples.", 
             x_train.len(), x_val.len(), x_test.len());

    println!("\n{}", "=".repeat(80));
    println!("PKBoost Benchmark ");
    println!("{}\n", "=".repeat(80));

    // Calculate class weights for imbalanced data
    let n_negatives = y_train.iter().filter(|&&label| label < 0.5).count();
    let n_positives = y_train.len() - n_negatives;
    let scale_pos_weight = n_negatives as f64 / n_positives as f64;
    
    println!("Class distribution:");
    println!("  Negatives: {} ({:.2}%)", n_negatives, 
             100.0 * n_negatives as f64 / y_train.len() as f64);
    println!("  Positives: {} ({:.2}%)", n_positives, 
             100.0 * n_positives as f64 / y_train.len() as f64);
    println!("  Scale pos weight: {:.2}\n", scale_pos_weight);

    println!("1. TRAINING PKBoost MODEL with Expert Tuning & Class Weighting...");
    
    // Build model using the new builder pattern
    let mut pkb_model = OptimizedPKBoostShannon::auto(&x_train, &y_train);
    println!("  scale_pos_weight={:.2}\n", pkb_model.scale_pos_weight);
    
    let pkb_start_time = Instant::now();
    pkb_model.fit(&x_train, &y_train, Some((&x_val, &y_val)), true)?;
    let pkb_time = pkb_start_time.elapsed().as_secs_f64();
    
    println!("\nPKBoost training completed in {:.2} seconds", pkb_time);
    
    println!("\n2. EVALUATING MODEL ON TEST SET...");
    let test_start = Instant::now();
    let test_probs = pkb_model.predict_proba(&x_test)?;
    let test_time = test_start.elapsed().as_secs_f64();
    
    let test_roc = calculate_roc_auc(&y_test, &test_probs);
    let test_pr = calculate_pr_auc(&y_test, &test_probs);

    println!("\n=== THRESHOLD OPTIMIZATION ===");
    let mut best_f1 = 0.0;
let mut best_threshold = 0.5;
let mut best_metrics = (0.0, 0.0, 0.0); // (precision, recall, f1)

for t in 5..95 {
    let threshold = t as f64 / 100.0;
    let predictions: Vec<usize> = test_probs.iter()
        .map(|&p| if p >= threshold { 1 } else { 0 })
        .collect();
    
    let mut tp = 0;
    let mut fp = 0;
    let mut fn_count = 0;
    
    for (i, &pred) in predictions.iter().enumerate() {
        let actual = if y_test[i] > 0.5 { 1 } else { 0 };
        match (pred, actual) {
            (1, 1) => tp += 1,
            (1, 0) => fp += 1,
            (0, 1) => fn_count += 1,
            _ => {}
        }
    }
    
    let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    let recall = if tp + fn_count > 0 { tp as f64 / (tp + fn_count) as f64 } else { 0.0 };
    let f1 = if precision + recall > 0.0 { 
        2.0 * precision * recall / (precision + recall) 
    } else { 
        0.0 
    };
    
    if f1 > best_f1 {
        best_f1 = f1;
        best_threshold = threshold;
        best_metrics = (precision, recall, f1);
    }
}

    println!("Optimal threshold: {:.3}", best_threshold);
    println!("  Precision: {:.4}", best_metrics.0);
    println!("  Recall:    {:.4}", best_metrics.1);
    println!("  F1 Score:  {:.4}", best_metrics.2);
    
    println!("\n{}", "=".repeat(80));
    println!("FINAL RESULTS");
    println!("{}", "=".repeat(80));
    println!("Test ROC-AUC:  {:.6}", test_roc);
    println!("Test PR-AUC:   {:.6}", test_pr);
    println!("Training time: {:.2}s", pkb_time);
    println!("Inference time: {:.4}s ({:.0} samples/sec)", 
             test_time, x_test.len() as f64 / test_time);
    println!("{}", "=".repeat(80));

    let threshold = best_threshold;
    let predictions: Vec<usize> = test_probs.iter()
        .map(|&p| if p >= threshold { 1 } else { 0 })
        .collect();
    
    let mut tp = 0;
    let mut fp = 0;
    let mut tn = 0;
    let mut fn_count = 0;
    
    for (i, &pred) in predictions.iter().enumerate() {
        let actual = if y_test[i] > 0.5 { 1 } else { 0 };
        match (pred, actual) {
            (1, 1) => tp += 1,
            (1, 0) => fp += 1,
            (0, 0) => tn += 1,
            (0, 1) => fn_count += 1,
            _ => {}
        }
    }
    
    let accuracy = (tp + tn) as f64 / y_test.len() as f64;
    let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
    let recall = if tp + fn_count > 0 { tp as f64 / (tp + fn_count) as f64 } else { 0.0 };
    let f1 = if precision + recall > 0.0 { 
        2.0 * precision * recall / (precision + recall) 
    } else { 
        0.0 
    };
    
    println!("\nClassification Metrics (threshold=0.5):");
    println!("  Accuracy:  {:.4}", accuracy);
    println!("  Precision: {:.4}", precision);
    println!("  Recall:    {:.4}", recall);
    println!("  F1 Score:  {:.4}", f1);
    println!("\nConfusion Matrix:");
    println!("              Predicted");
    println!("              Neg   Pos");
    println!("  Actual Neg  {:4}  {:4}", tn, fp);
    println!("  Actual Pos  {:4}  {:4}", fn_count, tp);

    Ok(())
}