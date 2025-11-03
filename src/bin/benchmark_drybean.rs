// Real-world benchmark: Dry Bean Dataset (7 classes, imbalanced)
use pkboost::MultiClassPKBoost;
use std::time::Instant;
use csv;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Dry Bean Dataset Benchmark ===\n");
    
    let (x_train, y_train) = load_csv("data/drybean_train.csv")?;
    let (x_test, y_test) = load_csv("data/drybean_test.csv")?;
    
    println!("Train: {} samples, {} features", x_train.len(), x_train[0].len());
    println!("Test: {} samples\n", x_test.len());
    
    print_class_distribution(&y_train, "Train");
    print_class_distribution(&y_test, "Test");
    println!();
    
    // PKBoost
    println!("Training PKBoost...");
    let start = Instant::now();
    let mut model = MultiClassPKBoost::new(7);
    model.fit(&x_train, &y_train, None, false)?;
    let time = start.elapsed().as_secs_f64();
    
    let preds = model.predict(&x_test)?;
    let acc = accuracy(&preds, &y_test);
    let macro_f1 = macro_f1(&preds, &y_test, 7);
    let weighted_f1 = weighted_f1(&preds, &y_test, 7);
    
    println!("\n=== Results ===");
    println!("Time: {:.2}s", time);
    println!("Accuracy: {:.2}%", acc * 100.0);
    println!("Macro-F1: {:.4}", macro_f1);
    println!("Weighted-F1: {:.4}", weighted_f1);
    
    println!("\n=== Per-Class Performance ===");
    println!("{:<10} {:<10} {:<12} {:<10} {:<10}", "Class", "Samples", "Precision", "Recall", "F1");
    println!("{}", "-".repeat(55));
    
    let class_names = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"];
    for class in 0..7 {
        let (prec, rec, f1) = per_class_metrics(&preds, &y_test, class);
        let n_samples = y_test.iter().filter(|&&y| y as usize == class).count();
        println!("{:<10} {:<10} {:<12.4} {:<10.4} {:<10.4}", 
                 class_names[class], n_samples, prec, rec, f1);
    }
    
    Ok(())
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

fn print_class_distribution(y: &[f64], label: &str) {
    let mut counts = vec![0; 7];
    for &label_val in y {
        counts[label_val as usize] += 1;
    }
    println!("{} distribution:", label);
    let class_names = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"];
    for (i, &count) in counts.iter().enumerate() {
        let pct = count as f64 / y.len() as f64 * 100.0;
        println!("  {}: {} ({:.1}%)", class_names[i], count, pct);
    }
}

fn accuracy(preds: &[usize], true_y: &[f64]) -> f64 {
    preds.iter().zip(true_y.iter())
        .filter(|(&pred, &true_val)| pred == true_val as usize)
        .count() as f64 / true_y.len() as f64
}

fn macro_f1(preds: &[usize], true_y: &[f64], n_classes: usize) -> f64 {
    let mut f1_sum = 0.0;
    for class in 0..n_classes {
        let (_, _, f1) = per_class_metrics(preds, true_y, class);
        f1_sum += f1;
    }
    f1_sum / n_classes as f64
}

fn weighted_f1(preds: &[usize], true_y: &[f64], n_classes: usize) -> f64 {
    let mut f1_sum = 0.0;
    let mut weight_sum = 0.0;
    for class in 0..n_classes {
        let weight = true_y.iter().filter(|&&y| y as usize == class).count() as f64;
        let (_, _, f1) = per_class_metrics(preds, true_y, class);
        f1_sum += f1 * weight;
        weight_sum += weight;
    }
    f1_sum / weight_sum
}

fn per_class_metrics(preds: &[usize], true_y: &[f64], class: usize) -> (f64, f64, f64) {
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
    
    (precision, recall, f1)
}
