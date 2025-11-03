use crate::model::OptimizedPKBoostShannon;
use crate::adaptive_parallel::get_parallel_config;

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
        let pos_class_weight = (1.0 / pos_ratio).min(1000.0);
        
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

pub fn auto_tune_principled(model: &mut OptimizedPKBoostShannon, n_samples: usize, n_features: usize, pos_ratio: f64) {
    let _config = get_parallel_config();
    
    let imbalance_level = match pos_ratio {
        p if p < 0.02 || p > 0.98 => "extreme",
        p if p < 0.10 || p > 0.90 => "high", 
        p if p < 0.20 || p > 0.80 => "moderate",
        _ => "balanced"
    };
    
    let data_complexity = match (n_samples, n_features) {
        (s, f) if s < 1000 || f < 10 => "trivial",
        (s, f) if s < 10000 && f < 50 => "simple", 
        (s, f) if s > 100000 || f > 200 => "complex",
        _ => "standard"
    };

    println!("\n=== Auto-Tuner ===");
    println!("Dataset Profile: {} samples, {} features", n_samples, n_features);
    println!("Imbalance: {:.1}% ({})", pos_ratio * 100.0, imbalance_level);
    println!("Complexity: {}", data_complexity);

    let base_lr = if n_samples < 5000 {
    0.1
} else if n_samples < 50000 {
    0.05
} else {
    0.03
};

let imbalance_factor = match imbalance_level {
    "extreme" => 0.85,
    "high" => 0.90,
    "moderate" => 0.95,
    _ => 1.0
};

model.learning_rate = f64::clamp(base_lr * imbalance_factor, 0.01, 0.15);
    
    let feature_depth = (n_features as f64).ln() as usize;
    let imbalance_penalty = match imbalance_level {
        "extreme" => 2,
        "high" => 1,
        _ => 0
    };
    model.max_depth = (feature_depth + 3).saturating_sub(imbalance_penalty).clamp(4, 10);
    
    model.reg_lambda = 0.1 * (n_features as f64).sqrt();
    model.gamma = 0.1;
    
    let pos_samples = (n_samples as f64 * pos_ratio) as f64;
    model.min_child_weight = (pos_samples * 0.01).max(1.0).min(20.0);
    
    model.subsample = 0.8;
    model.colsample_bytree = if n_features > 100 { 0.6 } else { 0.8 };
    
    model.mi_weight = match imbalance_level {
        "balanced" | "moderate" => 0.3,
        _ => 0.1
    };
    
    let base_estimators = (n_samples as f64).ln() as usize * 100;
    model.n_estimators = (base_estimators as f64 / model.learning_rate).ceil() as usize;
    model.n_estimators = model.n_estimators.clamp(200, 2000);
    
    model.early_stopping_rounds = ((n_samples as f64).ln() * 10.0) as usize;
    model.early_stopping_rounds = model.early_stopping_rounds.clamp(30, 150);
    model.histogram_bins = 32;

    println!("\nDerived Parameters:");
    println!("• Learning Rate: {:.4}", model.learning_rate);
    println!("• Max Depth: {}", model.max_depth);
    println!("• Estimators: {}", model.n_estimators);
    println!("• Col Sample: {:.2}", model.colsample_bytree);
    println!("• Reg Lambda: {:.2}", model.reg_lambda);
    println!("• Min Child Weight: {:.1}", model.min_child_weight);
    println!("• Gamma: {:.1}", model.gamma);
    println!("• MI Weight: {:.1}", model.mi_weight);
    println!("• Early Stopping Rounds: {}", model.early_stopping_rounds);
    println!();
}
