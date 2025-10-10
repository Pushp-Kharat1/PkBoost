use crate::model::OptimizedPKBoostShannon;
use crate::adaptive_parallel::get_parallel_config;

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
