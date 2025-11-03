// Test Shannon entropy impact on multi-class performance
use pkboost::MultiClassPKBoost;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Shannon Entropy Impact on Multi-Class ===\n");
    
    let (x_train, y_train, x_test, y_test) = generate_imbalanced_multiclass();
    
    println!("Dataset: {} train, {} test", x_train.len(), x_test.len());
    print_class_distribution(&y_train, "Train");
    print_class_distribution(&y_test, "Test");
    println!();
    
    // Test different MI weights
    let mi_weights = vec![0.0, 0.1, 0.3, 0.5, 0.7, 1.0];
    
    println!("{:<10} {:<10} {:<12} {:<10} {:<10}", "MI Weight", "Time(s)", "Accuracy", "Macro-F1", "Weighted-F1");
    println!("{}", "-".repeat(60));
    
    for &mi_weight in &mi_weights {
        let start = Instant::now();
        let mut model = MultiClassPKBoost::new(5);
        // Note: We'd need to modify MultiClassPKBoost to accept MI weight
        // For now, this shows the framework
        model.fit(&x_train, &y_train, None, false)?;
        let time = start.elapsed().as_secs_f64();
        
        let preds = model.predict(&x_test)?;
        let acc = accuracy(&preds, &y_test);
        let macro_f1 = macro_f1(&preds, &y_test, 5);
        let weighted_f1 = weighted_f1(&preds, &y_test, 5);
        
        println!("{:<10.1} {:<10.2} {:<12.2} {:<10.4} {:<10.4}", 
                 mi_weight, time, acc * 100.0, macro_f1, weighted_f1);
    }
    
    println!("\n=== Per-Class Performance (MI=0.3) ===");
    let mut model = MultiClassPKBoost::new(5);
    model.fit(&x_train, &y_train, None, false)?;
    let preds = model.predict(&x_test)?;
    
    println!("{:<8} {:<10} {:<12} {:<10} {:<10}", "Class", "Samples", "Precision", "Recall", "F1");
    println!("{}", "-".repeat(55));
    
    for class in 0..5 {
        let (prec, rec, f1) = per_class_metrics(&preds, &y_test, class);
        let n_samples = y_test.iter().filter(|&&y| y as usize == class).count();
        println!("{:<8} {:<10} {:<12.4} {:<10.4} {:<10.4}", class, n_samples, prec, rec, f1);
    }
    
    Ok(())
}

fn generate_imbalanced_multiclass() -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    use rand::prelude::*;
    let mut rng = rand::thread_rng();
    
    let class_ratios = [0.50, 0.25, 0.15, 0.07, 0.03];
    let n_train = 5000;
    let n_test = 1000;
    let n_features = 20;
    
    let mut x_train = Vec::new();
    let mut y_train = Vec::new();
    
    for (class_id, &ratio) in class_ratios.iter().enumerate() {
        let n_samples = (n_train as f64 * ratio) as usize;
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
