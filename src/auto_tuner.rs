use crate::adaptive_parallel::get_parallel_config;
use crate::model::OptimizedPKBoostShannon;

pub struct VulnerabilityCalibration {
    pub baseline_vulnerability: f64,
    pub alert_threshold: f64,
    pub metamorphosis_threshold: f64,
}

impl VulnerabilityCalibration {
    pub fn calibrate(
        model: &OptimizedPKBoostShannon,
        x_val: &Vec<Vec<f64>>,
        y_val: &[f64],
    ) -> Self {
        let preds = model.predict_proba(x_val).unwrap_or_default();
        let pos_ratio = y_val.iter().sum::<f64>() / y_val.len() as f64;
        let pos_class_weight = if pos_ratio > 1e-9 {
            (1.0 / pos_ratio).min(1000.0)
        } else {
            1000.0
        };

        let mut vulnerabilities = Vec::new();
        for (&pred, &true_y) in preds.iter().zip(y_val.iter()) {
            let confidence = (pred - 0.5).abs() * 2.0;
            let error = (true_y - pred).abs();
            let class_weight = if true_y > 0.5 { pos_class_weight } else { 1.0 };
            let vuln = confidence * error.powi(2) * class_weight;
            vulnerabilities.push(vuln);
        }

        let baseline = vulnerabilities.iter().sum::<f64>() / vulnerabilities.len().max(1) as f64;

        let (alert_thresh, meta_thresh) = match pos_ratio {
            p if p < 0.02 => (baseline * 1.5, baseline * 2.0),
            p if p < 0.10 => (baseline * 1.8, baseline * 2.5),
            p if p < 0.20 => (baseline * 2.0, baseline * 3.0),
            _ => (baseline * 2.5, baseline * 3.5),
        };

        Self {
            baseline_vulnerability: baseline,
            alert_threshold: alert_thresh,
            metamorphosis_threshold: meta_thresh,
        }
    }
}

/// Aggressive auto-tuner optimized for speed without sacrificing accuracy.
/// Key principles:
/// 1. Higher learning rate with early stopping (faster convergence)
/// 2. Rely on early stopping, not fixed tree counts
/// 3. Shallow trees for speed, regularization for generalization
pub fn auto_tune_principled(
    model: &mut OptimizedPKBoostShannon,
    n_samples: usize,
    n_features: usize,
    pos_ratio: f64,
) {
    let _config = get_parallel_config();

    let imbalance_level = match pos_ratio {
        p if p < 0.02 || p > 0.98 => "extreme",
        p if p < 0.10 || p > 0.90 => "high",
        p if p < 0.20 || p > 0.80 => "moderate",
        _ => "balanced",
    };

    let data_complexity = match (n_samples, n_features) {
        (s, f) if s < 1000 || f < 10 => "trivial",
        (s, f) if s < 10000 && f < 50 => "simple",
        (s, f) if s > 100000 || f > 200 => "complex",
        _ => "standard",
    };

    println!("\n=== Auto-Tuner ===");
    println!(
        "Dataset Profile: {} samples, {} features",
        n_samples, n_features
    );
    println!("Imbalance: {:.1}% ({})", pos_ratio * 100.0, imbalance_level);
    println!("Complexity: {}", data_complexity);

    // LEARNING RATE: Aggressive for fast convergence
    // Higher lr = fewer trees needed, relies on early stopping
    model.learning_rate = match data_complexity {
        "trivial" => 0.15,
        "simple" => 0.10,
        "standard" => 0.08,
        "complex" => 0.05,
        _ => 0.08,
    };

    // Slight reduction for extreme imbalance (more careful fitting)
    if imbalance_level == "extreme" {
        model.learning_rate *= 0.9;
    }
    model.learning_rate = model.learning_rate.clamp(0.03, 0.15);

    // MAX DEPTH: Shallow trees are faster and prevent overfitting
    // For imbalanced data, even shallower to avoid overfitting minority class
    model.max_depth = match (data_complexity, imbalance_level) {
        ("trivial", _) => 4,
        ("simple", "extreme" | "high") => 4,
        ("simple", _) => 5,
        ("standard", "extreme") => 4,
        ("standard", "high") => 5,
        ("standard", _) => 6,
        ("complex", "extreme" | "high") => 5,
        ("complex", _) => 6,
        _ => 5,
    };

    // N_ESTIMATORS: Set high enough to reach convergence, early stopping handles the rest
    // Formula: base trees scale with log(samples), not inversely with lr
    let base_trees = ((n_samples as f64).ln() * 50.0) as usize;
    model.n_estimators = match data_complexity {
        "trivial" => base_trees.clamp(100, 500),
        "simple" => base_trees.clamp(200, 800),
        "standard" => base_trees.clamp(300, 1000),
        "complex" => base_trees.clamp(400, 1000),
        _ => base_trees.clamp(300, 1000),
    };

    // EARLY STOPPING: Balance speed vs accuracy
    // More rounds for imbalanced data (needs more careful fitting)
    let base_es = match n_samples {
        s if s < 5000 => 50,
        s if s < 20000 => 40,
        s if s < 100000 => 35,
        _ => 30,
    };
    model.early_stopping_rounds = match imbalance_level {
        "extreme" => base_es + 20, // Extra patience for extreme imbalance
        "high" => base_es + 10,
        _ => base_es,
    };

    // REGULARIZATION: Scale with features, stronger for imbalanced
    model.reg_lambda = match imbalance_level {
        "extreme" => 0.15 * (n_features as f64).sqrt(),
        "high" => 0.12 * (n_features as f64).sqrt(),
        _ => 0.10 * (n_features as f64).sqrt(),
    };
    model.gamma = 0.1;

    // MIN CHILD WEIGHT: Prevent overfitting on rare samples
    let pos_samples = n_samples as f64 * pos_ratio;
    model.min_child_weight = match imbalance_level {
        "extreme" => (pos_samples * 0.02).max(2.0).min(10.0),
        "high" => (pos_samples * 0.01).max(1.5).min(8.0),
        _ => (pos_samples * 0.005).max(1.0).min(5.0),
    };

    // SUBSAMPLING: Always subsample for speed and regularization
    model.subsample = 0.8;
    model.colsample_bytree = if n_features > 100 { 0.6 } else { 0.8 };

    // MI WEIGHT: Shannon entropy contribution
    model.mi_weight = match imbalance_level {
        "balanced" | "moderate" => 0.3,
        _ => 0.1,
    };

    // HISTOGRAM BINS: 16 is 2x faster than 32 with minimal accuracy loss
    model.histogram_bins = 16;

    println!("\nDerived Parameters:");
    println!("• Learning Rate: {:.4}", model.learning_rate);
    println!("• Max Depth: {}", model.max_depth);
    println!(
        "• Estimators: {} (early stop @ {})",
        model.n_estimators, model.early_stopping_rounds
    );
    println!("• Col Sample: {:.2}", model.colsample_bytree);
    println!("• Reg Lambda: {:.2}", model.reg_lambda);
    println!("• Min Child Weight: {:.1}", model.min_child_weight);
    println!("• Gamma: {:.1}", model.gamma);
    println!("• MI Weight: {:.1}", model.mi_weight);
    println!();
}
