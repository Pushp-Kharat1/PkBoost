// Adaptive regression with drift detection and metamorphosis
// Regression equivalent of AdversarialLivingBooster

use crate::regression::PKBoostRegressor;
use crate::metabolism::FeatureMetabolism;
use crate::tree::{OptimizedTreeShannon, TreeParams};
use crate::optimized_data::TransposedData;
use crate::regression::calculate_rmse;
use crate::constants::*;
use rayon::prelude::*;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SystemState {
    Normal,
    Alert { checks_in_alert: usize },
    Metamorphosis,
}

pub struct RegressionVulnerability {
    pub error: f64,
    pub sample_idx: usize,
}

#[derive(Debug)]
pub struct DriftDiagnostics {
    pub error_entropy: f64,
    pub feature_entropy: Vec<f64>,
    pub drift_type: DriftType,
    pub residual_autocorrelation: f64,  // Temporal patterns
    pub heteroscedasticity_score: f64,  // Variance changes
}

#[derive(Debug, Clone, Copy)]
pub enum DriftType {
    Systemic,      // High entropy, widespread errors
    Localized,     // Low entropy, concentrated errors
    FeatureShift,  // Specific features changed
}

pub struct AdaptiveRegressor {
    primary: PKBoostRegressor,
    metabolism: FeatureMetabolism,
    state: SystemState,
    alert_trigger_threshold: usize,
    metamorphosis_trigger_threshold: usize,
    vulnerability_alert_threshold: f64,
    baseline_rmse: f64,
    consecutive_vulnerable_checks: usize,
    observations_count: usize,
    metamorphosis_count: usize,
    recent_x: VecDeque<Vec<f64>>,
    recent_y: VecDeque<f64>,
    buffer_size: usize,
    metamorphosis_cooldown: usize,
    iterations_since_metamorphosis: usize,
    recent_rmse: VecDeque<f64>,
    recent_vulnerabilities: VecDeque<RegressionVulnerability>,
    vulnerability_ema: f64,
    ema_alpha: f64,
}

impl AdaptiveRegressor {
    pub fn new(x_train: &Vec<Vec<f64>>, y_train: &[f64]) -> Self {
        let n_features = x_train.get(0).map_or(0, |row| row.len());
        let n_samples = x_train.len();
        
        let (alert_thresh, meta_thresh) = if n_samples < 50_000 {
            (1, 2)
        } else if n_samples < 200_000 {
            (2, 3)
        } else {
            (3, 5)
        };
        
        let buffer_sz = if n_samples < 50_000 { 10000 } else { 15000 };
        let cooldown = if n_samples < 50_000 { 1000 } else { 5000 };
        
        println!("\n=== Adaptive Regressor Configuration ===");
        println!("Dataset: {} samples, {} features", n_samples, n_features);
        println!("Alert trigger: {} checks", alert_thresh);
        println!("Metamorphosis trigger: {} checks", meta_thresh);
        println!("Buffer size: {} samples", buffer_sz);
        println!("Cooldown: {} observations", cooldown);
        println!("=========================================\n");
        
        Self {
            primary: PKBoostRegressor::auto(x_train, y_train),
            metabolism: FeatureMetabolism::new(n_features),
            state: SystemState::Normal,
            alert_trigger_threshold: alert_thresh,
            metamorphosis_trigger_threshold: meta_thresh,
            vulnerability_alert_threshold: 0.5,
            baseline_rmse: 0.0,
            consecutive_vulnerable_checks: 0,
            observations_count: 0,
            metamorphosis_count: 0,
            recent_x: VecDeque::with_capacity(buffer_sz),
            recent_y: VecDeque::with_capacity(buffer_sz),
            buffer_size: buffer_sz,
            metamorphosis_cooldown: cooldown,
            iterations_since_metamorphosis: 0,
            recent_rmse: VecDeque::with_capacity(5),
            recent_vulnerabilities: VecDeque::with_capacity(5000),
            vulnerability_ema: 0.0,
            ema_alpha: VULNERABILITY_EMA_ALPHA,
        }
    }
    
    pub fn fit_initial(&mut self, x: &Vec<Vec<f64>>, y: &[f64], eval_set: Option<(&Vec<Vec<f64>>, &[f64])>, verbose: bool) -> Result<(), String> {
        if verbose {
            println!("\n=== INITIAL TRAINING (Adaptive Regressor) ===");
        }
        self.primary.fit(x, y, eval_set, verbose)?;
        
        // Validate model learned
        let train_preds = self.primary.predict(x)?;
        let train_rmse = calculate_rmse(y, &train_preds);
        let y_mean = y.iter().sum::<f64>() / y.len() as f64;
        let y_std = (y.iter().map(|yi| (yi - y_mean).powi(2)).sum::<f64>() / y.len() as f64).sqrt();
        
        if train_rmse > y_std * 0.95 || !train_rmse.is_finite() {
            return Err(format!("Model failed to learn! RMSE: {:.4}, Baseline: {:.4}", train_rmse, y_std));
        }
        
        if let Some((x_val, y_val)) = eval_set {
            let val_preds = self.primary.predict(x_val)?;
            self.baseline_rmse = calculate_rmse(y_val, &val_preds);
            
            // Critical fix: never allow invalid baseline
            if self.baseline_rmse < 0.001 || !self.baseline_rmse.is_finite() {
                let val_mean = y_val.iter().sum::<f64>() / y_val.len() as f64;
                self.baseline_rmse = (y_val.iter().map(|y| (y - val_mean).powi(2)).sum::<f64>() / y_val.len() as f64).sqrt();
                if verbose {
                    println!("⚠️  Invalid baseline, using Y std: {:.4}", self.baseline_rmse);
                }
            }
            
            self.vulnerability_alert_threshold = self.baseline_rmse * 1.5;
            
            if verbose {
                println!("Train RMSE: {:.4}, Baseline: {:.4}", train_rmse, self.baseline_rmse);
                println!("Vulnerability threshold: {:.4}", self.vulnerability_alert_threshold);
            }
        } else {
            self.baseline_rmse = train_rmse;
            self.vulnerability_alert_threshold = self.baseline_rmse * 1.5;
        }
        
        if verbose {
            println!("Initial training complete. Model ready for streaming.");
        }
        Ok(())
    }
    
    pub fn observe_batch(&mut self, x: &Vec<Vec<f64>>, y: &[f64], verbose: bool) -> Result<(), String> {
        self.observations_count += x.len();
        self.iterations_since_metamorphosis += x.len();
        
        for (xi, &yi) in x.iter().zip(y.iter()) {
            if self.recent_x.len() >= self.buffer_size {
                self.recent_x.pop_front();
                self.recent_y.pop_front();
            }
            self.recent_x.push_back(xi.clone());
            self.recent_y.push_back(yi);
        }
        
        let preds = self.primary.predict(x)?;
        let batch_rmse = calculate_rmse(y, &preds);
        self.recent_rmse.push_back(batch_rmse);
        if self.recent_rmse.len() > 5 {
            self.recent_rmse.pop_front();
        }
        
        // Track vulnerabilities (large errors)
        for (i, (&pred, &true_y)) in preds.iter().zip(y.iter()).enumerate() {
            let error = (pred - true_y).abs();
            if error > self.baseline_rmse {
                let vuln = RegressionVulnerability { error, sample_idx: i };
                if self.recent_vulnerabilities.len() >= 5000 {
                    self.recent_vulnerabilities.pop_front();
                }
                self.recent_vulnerabilities.push_back(vuln);
                self.vulnerability_ema = self.ema_alpha * error + (1.0 - self.ema_alpha) * self.vulnerability_ema;
            }
        }
        
        let usage = self.primary.get_feature_usage();
        self.metabolism.update(&usage, self.observations_count);
        
        if self.iterations_since_metamorphosis > self.metamorphosis_cooldown {
            self.update_state(verbose);
        }
        
        if let SystemState::Metamorphosis = self.state {
            if verbose {
                println!("\n=== METAMORPHOSIS TRIGGERED at observation {} ===", self.observations_count);
            }
            self.execute_metamorphosis(verbose)?;
            self.iterations_since_metamorphosis = 0;
        }
        
        if verbose && self.observations_count % 5000 < x.len() {
            println!("Status @ {}: RMSE: {:.4}, State: {:?}, Vuln: {:.4}", 
                self.observations_count, batch_rmse, self.state, self.vulnerability_ema);
        }
        
        Ok(())
    }
    
    fn calculate_weighted_rmse(&self) -> Option<f64> {
        if self.recent_rmse.len() < 3 { return None; }
        let weights = vec![RMSE_WEIGHT_RECENT, RMSE_WEIGHT_MIDDLE, RMSE_WEIGHT_OLDEST];
        let weighted_sum: f64 = self.recent_rmse.iter().rev()
            .zip(weights.iter())
            .map(|(r, w)| r * w)
            .sum();
        Some(weighted_sum / (RMSE_WEIGHT_RECENT + RMSE_WEIGHT_MIDDLE + RMSE_WEIGHT_OLDEST))
    }
    
    fn update_state(&mut self, verbose: bool) {
        let weighted_rmse = match self.calculate_weighted_rmse() {
            Some(r) => r,
            None => return,
        };
        
        let degradation = (weighted_rmse - self.baseline_rmse) / self.baseline_rmse;
        
        // Adaptive threshold based on noise level
        let adaptive_threshold = if self.vulnerability_ema > self.baseline_rmse * NOISE_DETECTION_MULTIPLIER {
            BASE_DEGRADATION_THRESHOLD * NOISY_DATA_THRESHOLD_MULTIPLIER
        } else {
            BASE_DEGRADATION_THRESHOLD
        };
        
        let is_vulnerable = degradation > adaptive_threshold || self.vulnerability_ema > self.vulnerability_alert_threshold;
        
        match self.state {
            SystemState::Normal => {
                if is_vulnerable {
                    self.consecutive_vulnerable_checks += 1;
                    if self.consecutive_vulnerable_checks >= self.alert_trigger_threshold {
                        if verbose {
                            println!("-- ALERT: RMSE degradation {:.1}% (threshold: {:.1}%) --", 
                                degradation * 100.0, adaptive_threshold * 100.0);
                        }
                        self.state = SystemState::Alert { checks_in_alert: 1 };
                    }
                } else {
                    self.consecutive_vulnerable_checks = 0;
                }
            },
            SystemState::Alert { checks_in_alert } => {
                if is_vulnerable {
                    if checks_in_alert + 1 >= self.metamorphosis_trigger_threshold {
                        if verbose {
                            println!("-- METAMORPHOSIS: Persistent degradation --");
                        }
                        self.state = SystemState::Metamorphosis;
                    } else {
                        self.state = SystemState::Alert { checks_in_alert: checks_in_alert + 1 };
                    }
                } else {
                    if verbose { println!("-- System returned to NORMAL --"); }
                    self.consecutive_vulnerable_checks = 0;
                    self.state = SystemState::Normal;
                }
            },
            SystemState::Metamorphosis => {},
        }
    }
    
    fn calculate_residual_autocorrelation(&self, errors: &[f64]) -> f64 {
        // Lag-1 autocorrelation to detect systematic drift
        if errors.len() < 2 { return 0.0; }
        let mean = errors.iter().sum::<f64>() / errors.len() as f64;
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in 0..errors.len()-1 {
            numerator += (errors[i] - mean) * (errors[i+1] - mean);
        }
        for &e in errors {
            denominator += (e - mean).powi(2);
        }
        
        if denominator < EPSILON { 0.0 } else { numerator / denominator }
    }
    
    fn calculate_heteroscedasticity(&self, predictions: &[f64], errors: &[f64]) -> f64 {
        // Variance increases with prediction magnitude = heteroscedasticity
        if predictions.len() < 10 { return 0.0; }
        
        let n_bins = 10.min(predictions.len() / 10);
        let mut pred_sorted: Vec<(f64, f64)> = predictions.iter()
            .zip(errors.iter())
            .map(|(&p, &e)| (p, e.abs()))
            .collect();
        pred_sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        let chunk_size = pred_sorted.len() / n_bins;
        let mut bin_vars = Vec::new();
        
        for chunk in pred_sorted.chunks(chunk_size) {
            let chunk_errors: Vec<f64> = chunk.iter().map(|(_, e)| *e).collect();
            let mean = chunk_errors.iter().sum::<f64>() / chunk_errors.len() as f64;
            let var = chunk_errors.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / chunk_errors.len() as f64;
            bin_vars.push(var);
        }
        
        // Calculate variance of variances (high = heteroscedastic)
        if bin_vars.is_empty() { return 0.0; }
        let mean_var = bin_vars.iter().sum::<f64>() / bin_vars.len() as f64;
        bin_vars.iter().map(|v| (v - mean_var).powi(2)).sum::<f64>().sqrt()
    }
    
    fn calculate_error_entropy(&self, errors: &[f64]) -> f64 {
        if errors.is_empty() { return 0.0; }
        
        let mut sorted_errors: Vec<f64> = errors.iter()
            .copied()
            .filter(|e| e.is_finite() && *e >= 0.0)
            .collect();
        
        if sorted_errors.is_empty() { return 0.0; }
        sorted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n_bins = 10.min(sorted_errors.len());
        let mut bins = vec![0; n_bins];
        let max_error = sorted_errors.last().unwrap() + EPSILON;
        
        for &err in &sorted_errors {
            let bin_idx = ((err / max_error) * n_bins as f64)
                .floor().min((n_bins - 1) as f64) as usize;
            bins[bin_idx] += 1;
        }
        
        let total = sorted_errors.len() as f64;
        let entropy: f64 = bins.iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = count as f64 / total;
                -p * p.log2()
            })
            .sum();
        
        entropy.max(0.0)
    }
    
    fn diagnose_drift(&self) -> DriftDiagnostics {
        let val_size = 2000.min(self.recent_x.len());
        let val_x: Vec<Vec<f64>> = self.recent_x.iter().rev().take(val_size).cloned().collect();
        let val_y: Vec<f64> = self.recent_y.iter().rev().take(val_size).cloned().collect();
        
        let preds = self.primary.predict(&val_x).unwrap_or_default();
        let errors: Vec<f64> = preds.iter().zip(val_y.iter())
            .map(|(p, y)| (p - y).abs())
            .collect();
        
        let error_entropy = self.calculate_error_entropy(&errors);
        
        // Calculate per-feature entropy (measure feature distribution shift)
        let n_features = val_x.get(0).map_or(0, |row| row.len());
        let mut feature_entropy = Vec::new();
        
        for feat_idx in 0..n_features {
            let feat_vals: Vec<f64> = val_x.iter().map(|row| row[feat_idx]).collect();
            let feat_ent = self.calculate_error_entropy(&feat_vals);
            feature_entropy.push(feat_ent);
        }
        
        // Classify drift type
        let drift_type = if error_entropy > SYSTEMIC_DRIFT_ENTROPY {
            DriftType::Systemic
        } else if error_entropy < LOCALIZED_DRIFT_ENTROPY {
            DriftType::Localized
        } else {
            DriftType::FeatureShift
        };
        
        // Calculate residual autocorrelation (temporal drift)
        let residual_autocorrelation = self.calculate_residual_autocorrelation(&errors);
        
        // Calculate heteroscedasticity (variance changes)
        let heteroscedasticity_score = self.calculate_heteroscedasticity(&preds, &errors);
        
        DriftDiagnostics { 
            error_entropy, 
            feature_entropy, 
            drift_type,
            residual_autocorrelation,
            heteroscedasticity_score,
        }
    }
    
    fn execute_metamorphosis(&mut self, verbose: bool) -> Result<(), String> {
        let checkpoint_trees = self.primary.trees.clone();
        
        // Dynamic validation size based on buffer
        let val_size = match self.recent_x.len() {
            0..=4000 => (self.recent_x.len() / 3).max(MIN_VALIDATION_SIZE),
            4001..=15000 => 2000,
            _ => ((self.recent_x.len() as f64 * 0.2) as usize).max(2000),
        };
        
        let val_x: Vec<Vec<f64>> = self.recent_x.iter().rev().take(val_size).cloned().collect();
        let val_y: Vec<f64> = self.recent_y.iter().rev().take(val_size).cloned().collect();
        let pre_preds = self.primary.predict(&val_x)?;
        let checkpoint_rmse = calculate_rmse(&val_y, &pre_preds);
        
        // Diagnose drift using Shannon entropy
        let diagnostics = self.diagnose_drift();
        
        // Identify drifted features using distribution shift
        let drifted_features: Vec<usize> = if self.recent_x.len() > 2000 {
            let n_features = self.recent_x.get(0).map_or(0, |r| r.len());
            (0..n_features).filter(|&feat_idx| {
                let recent: Vec<f64> = self.recent_x.iter().rev().take(1000)
                    .filter_map(|r| r.get(feat_idx).copied())
                    .collect();
                let older: Vec<f64> = self.recent_x.iter().take(1000)
                    .filter_map(|r| r.get(feat_idx).copied())
                    .collect();
                
                if recent.len() < 100 || older.len() < 100 { return false; }
                
                let recent_mean = recent.iter().sum::<f64>() / recent.len() as f64;
                let older_mean = older.iter().sum::<f64>() / older.len() as f64;
                let drift_score = (recent_mean - older_mean).abs() / older_mean.abs().max(EPSILON);
                
                drift_score > FEATURE_DRIFT_THRESHOLD
            }).collect()
        } else {
            // Fallback to entropy for small buffers
            let avg_feat_entropy = diagnostics.feature_entropy.iter().sum::<f64>() / diagnostics.feature_entropy.len() as f64;
            diagnostics.feature_entropy.iter().enumerate()
                .filter(|(_, &ent)| ent > avg_feat_entropy * 1.2)
                .map(|(i, _)| i)
                .collect()
        };
        
        if verbose {
            println!("  - Checkpointing {} trees", checkpoint_trees.len());
            println!("  - Drift Analysis:");
            println!("    * Error Entropy: {:.3} (0=concentrated, 3.3=uniform)", diagnostics.error_entropy);
            println!("    * Drift Type: {:?}", diagnostics.drift_type);
            println!("    * Residual Autocorr: {:.3} (temporal drift)", diagnostics.residual_autocorrelation);
            println!("    * Heteroscedasticity: {:.3} (variance changes)", diagnostics.heteroscedasticity_score);
            if !drifted_features.is_empty() {
                println!("    * Drifted features (high entropy): {:?}", drifted_features);
            }
        }
        
        // Profile buffer data for auto-tuning (like classifier does)
        let buffer_size = self.recent_x.len();
        let buffer_x: Vec<Vec<f64>> = self.recent_x.iter().cloned().collect();
        let buffer_y: Vec<f64> = self.recent_y.iter().cloned().collect();
        
        let predictions = self.primary.predict(&buffer_x).unwrap_or_default();
        let degradation = (checkpoint_rmse - self.baseline_rmse) / self.baseline_rmse;
        
        // Calculate error variance (model struggling)
        let error_variance: f64 = buffer_y.iter().zip(predictions.iter())
            .map(|(y, p)| (y - p).powi(2))
            .sum::<f64>() / buffer_size as f64;
        
        // Calculate prediction variance (heteroscedastic noise)
        let pred_mean = predictions.iter().sum::<f64>() / buffer_size as f64;
        let pred_variance: f64 = predictions.iter()
            .map(|p| (p - pred_mean).powi(2))
            .sum::<f64>() / buffer_size as f64;
        
        // Classify buffer complexity considering both error and prediction variance
        let complexity_score = error_variance + pred_variance * 0.5;
        let complexity_level = if complexity_score > HIGH_COMPLEXITY_THRESHOLD {
            "high"
        } else if complexity_score > MODERATE_COMPLEXITY_THRESHOLD {
            "moderate"
        } else {
            "low"
        };
        
        // Combined drift scoring
        let temporal_score = diagnostics.residual_autocorrelation.abs();
        let variance_score = (diagnostics.heteroscedasticity_score / self.baseline_rmse.max(EPSILON)).min(1.0);
        let entropy_score = (diagnostics.error_entropy / 3.5).min(1.0);
        
        let combined_drift_score = 
            DRIFT_WEIGHT_ENTROPY * entropy_score + 
            DRIFT_WEIGHT_TEMPORAL * temporal_score + 
            DRIFT_WEIGHT_VARIANCE * variance_score;
        
        // Base strategy from combined score
        let base_trees = if combined_drift_score > SEVERE_DRIFT_THRESHOLD {
            TREES_SEVERE_DRIFT
        } else if temporal_score > TEMPORAL_DRIFT_THRESHOLD {
            TREES_TEMPORAL_DRIFT
        } else if variance_score > VARIANCE_DRIFT_THRESHOLD {
            TREES_VARIANCE_DRIFT
        } else {
            TREES_LOCALIZED_DRIFT
        };
        
        // Auto-tune based on buffer size (like n_samples in classifier)
        let size_factor = if buffer_size < SMALL_BUFFER_THRESHOLD {
            SIZE_FACTOR_SMALL
        } else if buffer_size > LARGE_BUFFER_THRESHOLD {
            SIZE_FACTOR_LARGE
        } else {
            1.0
        };
        
        // Auto-tune based on complexity (like imbalance_factor in classifier)
        let complexity_factor = match complexity_level {
            "high" => COMPLEXITY_FACTOR_HIGH,
            "moderate" => 1.0,
            "low" => COMPLEXITY_FACTOR_LOW,
            _ => 1.0,
        };
        
        // Auto-tune based on degradation severity
        let severity_factor = if degradation > 1.5 {
            SEVERITY_FACTOR_VERY_SEVERE
        } else if degradation > 1.0 {
            SEVERITY_FACTOR_SEVERE
        } else if degradation < 0.3 {
            SEVERITY_FACTOR_MILD
        } else {
            1.0
        };
        
        let n_new_trees = ((base_trees as f64 * size_factor * complexity_factor * severity_factor) as usize)
            .clamp(MIN_TREES_PER_METAMORPHOSIS, MAX_TREES_PER_METAMORPHOSIS);
        
        // Auto-tune learning rate (like base_lr * imbalance_factor in classifier)
        let lr_adjustment = match complexity_level {
            "high" => LR_ADJUSTMENT_HIGH_COMPLEXITY,
            "moderate" => 1.0,
            "low" => LR_ADJUSTMENT_LOW_COMPLEXITY,
            _ => 1.0,
        };
        let lr_multiplier = BASE_LR_MULTIPLIER * lr_adjustment;
        
        let prune_threshold = match diagnostics.drift_type {
            DriftType::Systemic => 0.95,
            DriftType::FeatureShift => 0.90,
            DriftType::Localized => 1.0,
        };
        
        if verbose {
            let strategy_name = if combined_drift_score > SEVERE_DRIFT_THRESHOLD {
                "Severe (combined high scores)"
            } else if temporal_score > TEMPORAL_DRIFT_THRESHOLD {
                "Temporal (autocorrelated errors)"
            } else if variance_score > VARIANCE_DRIFT_THRESHOLD {
                "Variance (heteroscedastic)"
            } else {
                "Localized (concentrated errors)"
            };
            println!("    * Combined Drift Score: {:.3}", combined_drift_score);
            println!("      - Entropy: {:.3} (weight: {:.1})", entropy_score, DRIFT_WEIGHT_ENTROPY);
            println!("      - Temporal: {:.3} (weight: {:.1})", temporal_score, DRIFT_WEIGHT_TEMPORAL);
            println!("      - Variance: {:.3} (weight: {:.1})", variance_score, DRIFT_WEIGHT_VARIANCE);
            println!("    * Strategy: {}", strategy_name);
            println!("    * Buffer Profile:");
            println!("      - Size: {} samples (factor: {:.2}x)", buffer_size, size_factor);
            println!("      - Error variance: {:.3}, Pred variance: {:.3}", error_variance, pred_variance);
            println!("      - Complexity: {} (score: {:.3})", complexity_level, complexity_score);
            println!("      - Degradation: {:.1}% (factor: {:.2}x)", degradation * 100.0, severity_factor);
            println!("    * Auto-tuned Parameters:");
            println!("      - Trees: {} (base: {}, total factor: {:.2}x)", 
                n_new_trees, base_trees, size_factor * complexity_factor * severity_factor);
            println!("      - Learning rate: {:.2}x (base: {:.1}x, adj: {:.2}x)", 
                lr_multiplier, BASE_LR_MULTIPLIER, lr_adjustment);
        }
        
        // Also identify truly dead features (usage < 1%)
        let usage = self.primary.get_feature_usage();
        let total_usage: usize = usage.iter().sum();
        let dead_features: Vec<usize> = usage.iter().enumerate()
            .filter(|(_, &u)| u < total_usage / 100)
            .map(|(i, _)| i)
            .collect();
        
        if verbose && !dead_features.is_empty() {
            println!("    * Dead features (<1% usage): {:?}", dead_features);
        }
        
        // Prune trees using drifted features more aggressively
        let pruned = if !drifted_features.is_empty() {
            let drift_prune_threshold = prune_threshold * 0.7;  // Lower threshold for drifted features
            let pruned_count = self.primary.prune_trees(&drifted_features, drift_prune_threshold);
            let max_prune = (checkpoint_trees.len() * 30) / 100;
            pruned_count.min(max_prune)
        } else if !dead_features.is_empty() {
            let pruned_count = self.primary.prune_trees(&dead_features, prune_threshold);
            let max_prune = (checkpoint_trees.len() * 30) / 100;
            pruned_count.min(max_prune)
        } else {
            0
        };
        
        if verbose {
            println!("  - Pruned {} trees (max 30%)", pruned);
        }
        
        if self.recent_x.len() > 1000 {
            if verbose {
                println!("  - Adding {} new trees on buffer", n_new_trees);
            }
            self.add_incremental_trees(n_new_trees, lr_multiplier, verbose)?;
        }
        
        // Validate on same data
        if val_size > 100 {
            let post_preds = self.primary.predict(&val_x)?;
            let post_rmse = calculate_rmse(&val_y, &post_preds);
            
            // Rollback only if performance degrades
            if post_rmse > checkpoint_rmse * METAMORPHOSIS_ROLLBACK_TOLERANCE {
                if verbose {
                    println!("  ⚠️  ROLLBACK: {:.4} → {:.4} (degraded)", checkpoint_rmse, post_rmse);
                }
                self.primary.trees = checkpoint_trees;
                self.state = SystemState::Normal;
                self.consecutive_vulnerable_checks = 0;
                return Ok(());
            }
            
            let improvement = if checkpoint_rmse > EPSILON {
                ((checkpoint_rmse - post_rmse) / checkpoint_rmse) * 100.0
            } else {
                0.0
            };
            if verbose {
                println!("  ✅ ACCEPTED: {:.4} → {:.4} ({:+.1}%)", checkpoint_rmse, post_rmse, improvement);
                println!("    Baseline RMSE was: {:.4}", self.baseline_rmse);
            }
            
            // Update baseline if improved
            if post_rmse < self.baseline_rmse * 1.1 {
                self.baseline_rmse = post_rmse;
                if verbose {
                    println!("    Updated baseline to: {:.4}", self.baseline_rmse);
                }
            }
        }
        
        self.metamorphosis_count += 1;
        self.state = SystemState::Normal;
        self.consecutive_vulnerable_checks = 0;
        self.recent_vulnerabilities.clear();
        
        if verbose {
            println!("=== METAMORPHOSIS COMPLETE ===");
            println!("  - Trees: {}", self.primary.trees.len());
            println!("  - Total metamorphoses: {}\n", self.metamorphosis_count);
        }
        
        Ok(())
    }
    
    fn add_incremental_trees(&mut self, n_trees: usize, lr_multiplier: f64, verbose: bool) -> Result<usize, String> {
        let buffer_x: Vec<Vec<f64>> = self.recent_x.iter().cloned().collect();
        let buffer_y: Vec<f64> = self.recent_y.iter().cloned().collect();
        
        if buffer_x.len() < 1000 {
            return Err(format!("Insufficient buffer: {} samples", buffer_x.len()));
        }
        
        let y_mean = buffer_y.iter().sum::<f64>() / buffer_y.len() as f64;
        let current_preds = self.primary.predict(&buffer_x)?;
        let pred_mean = current_preds.iter().sum::<f64>() / current_preds.len() as f64;
        let pred_error = (pred_mean - y_mean).abs();
        
        // Check if predictions are catastrophically wrong
        let mut raw_preds = if pred_error > y_mean.abs() * 10.0 || !pred_mean.is_finite() {
            if verbose {
                println!("  ⚠️  Predictions far off, resetting to mean baseline");
            }
            vec![y_mean; buffer_x.len()]
        } else {
            current_preds.clone()
        };
        
        let hb = self.primary.histogram_builder.as_ref().unwrap();
        let x_proc = hb.transform(&buffer_x);
        let transposed = TransposedData::from_rows(&x_proc);
        
        let n_features = buffer_x[0].len();
        let feature_indices: Vec<usize> = (0..n_features).collect();
        let sample_indices: Vec<usize> = (0..buffer_x.len()).collect();
        
        let params = TreeParams {
            min_samples_split: self.primary.min_samples_split,
            min_child_weight: self.primary.min_child_weight,
            reg_lambda: self.primary.reg_lambda * 5.0,  // Increase regularization
            gamma: self.primary.gamma * 2.0,
            mi_weight: 0.3,
            n_bins_per_feature: feature_indices.iter().map(|&i| hb.n_bins_per_feature[i]).collect(),
        };
        
        let adaptive_lr = (self.primary.learning_rate * lr_multiplier).min(0.05);  // Cap LR
        let mut trees_added = 0;
        
        for tree_idx in 0..n_trees {
            let mut grad = self.primary.get_gradient(&buffer_y, &raw_preds);
            let hess = self.primary.get_hessian(&buffer_y, &raw_preds);
            
            // Gradient clipping
            let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            if grad_norm > GRADIENT_CRITICAL_THRESHOLD {
                let scale = GRADIENT_CRITICAL_THRESHOLD / grad_norm;
                grad = grad.iter().map(|&g| g * scale).collect();
                if verbose && tree_idx == 0 {
                    println!("  🔧 Clipping gradients: {:.0} → {:.0}", grad_norm, GRADIENT_CRITICAL_THRESHOLD);
                }
            }
            
            if grad_norm > GRADIENT_CRITICAL_THRESHOLD * 5.0 {
                if verbose {
                    println!("  🛑 Stopping early: gradients too large ({:.0})", grad_norm);
                }
                break;
            }
            
            let mut tree = OptimizedTreeShannon::new(self.primary.max_depth);
            tree.fit_optimized(&transposed, &buffer_y, &grad, &hess, &sample_indices, &feature_indices, &params);
            
            let tree_preds: Vec<f64> = (0..buffer_x.len()).into_par_iter()
                .map(|i| tree.predict_from_transposed(&transposed, i))
                .collect();
            
            for (i, &tp) in tree_preds.iter().enumerate() {
                raw_preds[i] += adaptive_lr * tp;
                // Clip predictions
                let y_range = y_mean.abs() * 100.0;
                raw_preds[i] = raw_preds[i].clamp(y_mean - y_range, y_mean + y_range);
            }
            
            self.primary.trees.push(tree);
            trees_added += 1;
        }
        
        Ok(trees_added)
    }
    
    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Result<Vec<f64>, String> {
        self.primary.predict(x)
    }
    
    pub fn get_state(&self) -> SystemState {
        self.state
    }
    
    pub fn get_metamorphosis_count(&self) -> usize {
        self.metamorphosis_count
    }
    
    pub fn get_vulnerability_score(&self) -> f64 {
        self.vulnerability_ema
    }
    
    pub fn predict_with_uncertainty(&self, x: &Vec<Vec<f64>>) -> Result<(Vec<f64>, Vec<f64>), String> {
        self.primary.predict_with_uncertainty(x)
    }
}
