// Benchmark: ULB Credit Card Fraud Detection
// Uses pre-split train/val/test CSVs, trains PKBoost, reports metrics + timing.

use pkboost::metrics::{calculate_pr_auc, calculate_roc_auc};
use pkboost::model::OptimizedPKBoostShannon;

use ndarray::{Array1, Array2};
use std::io::{BufRead, BufReader};
use std::time::Instant;

const TRAIN_CSV: &str = "data/creditcard_train.csv";
const VAL_CSV: &str = "data/creditcard_val.csv";
const TEST_CSV: &str = "data/creditcard_test.csv";

fn load_csv(path: &str) -> (Array2<f64>, Array1<f64>) {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("Cannot open {}: {}", path, e));
    let reader = BufReader::new(file);

    let mut features: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<f64> = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line.expect("Failed to read line");
        if i == 0 {
            continue;
        } // skip header

        let values: Vec<&str> = line.split(',').collect();
        if values.len() < 2 {
            continue;
        }

        // Last column = Class (0/1), skip first column (Time)
        let label: f64 = values.last().unwrap().trim().parse().unwrap_or(0.0);
        let row: Vec<f64> = values[1..values.len() - 1]
            .iter()
            .map(|s| s.trim().parse().unwrap_or(f64::NAN))
            .collect();

        features.push(row);
        labels.push(label);
    }

    let n_samples = features.len();
    let n_features = features[0].len();
    let flat: Vec<f64> = features.into_iter().flatten().collect();
    let x = Array2::from_shape_vec((n_samples, n_features), flat).expect("shape mismatch");
    let y = Array1::from(labels);

    let pos = y.iter().filter(|&&v| v > 0.5).count();
    println!(
        "  {} => {} samples, {} features | fraud: {} ({:.3}%)",
        path,
        n_samples,
        n_features,
        pos,
        100.0 * pos as f64 / n_samples as f64
    );

    (x, y)
}

fn evaluate(name: &str, y_true: &[f64], y_proba: &[f64]) {
    let roc_auc = calculate_roc_auc(y_true, y_proba);
    let pr_auc = calculate_pr_auc(y_true, y_proba);

    let (mut tp, mut fp, mut tn, mut fn_) = (0usize, 0, 0, 0);
    for (i, &prob) in y_proba.iter().enumerate() {
        match (prob > 0.5, y_true[i] > 0.5) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => tn += 1,
            (false, true) => fn_ += 1,
        }
    }

    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    println!("  === {} ===", name);
    println!("  ROC-AUC:   {:.6}", roc_auc);
    println!("  PR-AUC:    {:.6}", pr_auc);
    println!("  Precision: {:.4} ({}/{})", precision, tp, tp + fp);
    println!("  Recall:    {:.4} ({}/{})", recall, tp, tp + fn_);
    println!("  F1-Score:  {:.4}", f1);
    println!("  TP={} FP={} TN={} FN={}", tp, fp, tn, fn_);
    println!();
}

fn main() {
    println!("========================================");
    println!(" PKBoost Benchmark - Credit Card Fraud");
    println!("========================================\n");

    // Load
    println!("[LOAD]");
    let (x_train, y_train) = load_csv(TRAIN_CSV);
    let (x_val, y_val) = load_csv(VAL_CSV);
    let (x_test, y_test) = load_csv(TEST_CSV);

    // Train
    println!("\n[TRAIN]");
    let mut model = OptimizedPKBoostShannon::auto(x_train.view(), y_train.view());
    println!("  n_estimators:     {}", model.n_estimators);
    println!("  learning_rate:    {:.4}", model.learning_rate);
    println!("  max_depth:        {}", model.max_depth);
    println!("  histogram_bins:   {}", model.histogram_bins);
    println!("  scale_pos_weight: {:.2}", model.scale_pos_weight);

    let t0 = Instant::now();
    model
        .fit(
            x_train.view(),
            y_train.view(),
            Some((x_val.view(), y_val.view())),
            true,
        )
        .expect("Training failed!");
    let train_secs = t0.elapsed().as_secs_f64();

    println!("\n[TIMING]");
    println!("  Training:    {:.2}s", train_secs);
    println!("  Trees built: {}", model.trees.len());
    println!(
        "  Throughput:  {:.0} samples*trees/sec",
        (x_train.nrows() as f64 * model.trees.len() as f64) / train_secs
    );

    // Evaluate
    println!("\n[EVAL]");
    let val_proba = model
        .predict_proba(x_val.view())
        .expect("val predict failed");
    evaluate(
        "Validation",
        y_val.as_slice().unwrap(),
        val_proba.as_slice().unwrap(),
    );

    let test_proba = model
        .predict_proba(x_test.view())
        .expect("test predict failed");
    evaluate(
        "Test",
        y_test.as_slice().unwrap(),
        test_proba.as_slice().unwrap(),
    );

    println!("[DONE]");
}
