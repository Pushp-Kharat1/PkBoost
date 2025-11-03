// Benchmark: PKBoost vs XGBoost vs LightGBM on imbalanced multi-class
use pkboost::MultiClassPKBoost;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Imbalanced Multi-Class Benchmark ===\n");
    
    let (x_train, y_train, x_test, y_test) = generate_imbalanced_multiclass();
    
    println!("Dataset: {} train, {} test", x_train.len(), x_test.len());
    print_class_distribution(&y_train, "Train");
    print_class_distribution(&y_test, "Test");
    println!();
    
    // PKBoost
    println!("--- PKBoost ---");
    let start = Instant::now();
    let mut pkb = MultiClassPKBoost::new(5);
    pkb.fit(&x_train, &y_train, None, false)?;
    let pkb_time = start.elapsed().as_secs_f64();
    let pkb_preds = pkb.predict(&x_test)?;
    let pkb_acc = accuracy(&pkb_preds, &y_test);
    let pkb_f1 = macro_f1(&pkb_preds, &y_test, 5);
    println!("Time: {:.2}s | Accuracy: {:.2}% | Macro-F1: {:.4}", pkb_time, pkb_acc * 100.0, pkb_f1);
    
    println!("\n=== Results Summary ===");
    println!("{:<12} {:<10} {:<12} {:<10}", "Model", "Time(s)", "Accuracy", "Macro-F1");
    println!("{}", "-".repeat(50));
    println!("{:<12} {:<10.2} {:<12.2} {:<10.4}", "PKBoost", pkb_time, pkb_acc * 100.0, pkb_f1);
    
    Ok(())
}

fn generate_imbalanced_multiclass() -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    use rand::prelude::*;
    let mut rng = rand::thread_rng();
    
    // 5 classes with severe imbalance: 50%, 25%, 15%, 7%, 3%
    let class_ratios = [0.50, 0.25, 0.15, 0.07, 0.03];
    let n_train = 5000;
    let n_test = 1000;
    let n_features = 20;
    
    let mut x_train = Vec::new();
    let mut y_train = Vec::new();
    
    for (class_id, &ratio) in class_ratios.iter().enumerate() {
        let n_samples = (n_train as f64 * ratio) as usize;
        let mean = class_id as f64 * 0.5;  // Reduced separation (0, 0.5, 1.0, 1.5, 2.0)
        
        for _ in 0..n_samples {
            let mut features = Vec::new();
            for feat_idx in 0..n_features {
                if feat_idx < 5 {
                    // Informative features with Box-Muller normal distribution
                    let u1: f64 = rng.gen();
                    let u2: f64 = rng.gen();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    features.push(mean + z * 2.0);  // std=2.0 for overlap
                } else {
                    // Noise features
                    features.push(rng.gen::<f64>() * 4.0 - 2.0);
                }
            }
            x_train.push(features);
            y_train.push(class_id as f64);
        }
    }
    
    let mut x_test = Vec::new();
    let mut y_test = Vec::new();
    
    for (class_id, &ratio) in class_ratios.iter().enumerate() {
        let n_samples = (n_test as f64 * ratio) as usize;
        let mean = class_id as f64 * 0.5;
        
        for _ in 0..n_samples {
            let mut features = Vec::new();
            for feat_idx in 0..n_features {
                if feat_idx < 5 {
                    let u1: f64 = rng.gen();
                    let u2: f64 = rng.gen();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    features.push(mean + z * 2.0);
                } else {
                    features.push(rng.gen::<f64>() * 4.0 - 2.0);
                }
            }
            x_test.push(features);
            y_test.push(class_id as f64);
        }
    }
    
    // Shuffle to avoid ordering bias
    let mut combined_train: Vec<_> = x_train.into_iter().zip(y_train.into_iter()).collect();
    combined_train.shuffle(&mut rng);
    let (x_train, y_train): (Vec<_>, Vec<_>) = combined_train.into_iter().unzip();
    
    let mut combined_test: Vec<_> = x_test.into_iter().zip(y_test.into_iter()).collect();
    combined_test.shuffle(&mut rng);
    let (x_test, y_test): (Vec<_>, Vec<_>) = combined_test.into_iter().unzip();
    
    (x_train, y_train, x_test, y_test)
}

fn print_class_distribution(y: &[f64], label: &str) {
    let mut counts = vec![0; 5];
    for &label_val in y {
        counts[label_val as usize] += 1;
    }
    println!("{} distribution:", label);
    for (i, &count) in counts.iter().enumerate() {
        let pct = count as f64 / y.len() as f64 * 100.0;
        println!("  Class {}: {} ({:.1}%)", i, count, pct);
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
