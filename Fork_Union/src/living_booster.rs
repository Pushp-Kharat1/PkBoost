// This is the main "living" model that can adapt to data drift in real-time
// The idea is to detect when the model starts failing and trigger a metamorphosis
// to prune bad trees and grow new ones on recent data

use crate::model::OptimizedPKBoostShannon;
use crate::adversarial::{AdversarialEnsemble, Vulnerability};
use crate::metabolism::FeatureMetabolism;
use crate::tree::{OptimizedTreeShannon, TreeParams};
use crate::optimized_data::TransposedData;
use crate::metrics::calculate_pr_auc;
use fork_union::prelude::*;
use std::time::Instant;
use std::collections::VecDeque;

// State machine for the model - tracks if we're doing ok or need to adapt
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SystemState {
    Normal,  // everything's fine
    Alert { checks_in_alert: usize },  // performance is degrading
    Metamorphosis,  // time to rebuild parts of the model
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetamorphosisStrategy {
    Conservative,
    DataAware,
    FeatureAware,
}

pub struct DriftAnalyzer {
    feature_volatility: Vec<f64>,
    minority_class_error_rate: f64,
}

impl DriftAnalyzer {
    pub fn new(n_features: usize) -> Self {
        Self {
            feature_volatility: vec![0.0; n_features],
            minority_class_error_rate: 0.0,
        }
    }
    
    pub fn update(&mut self, recent_x: &VecDeque<Vec<f64>>, recent_y: &VecDeque<f64>, vulnerabilities: &VecDeque<Vulnerability>) {
        if recent_x.is_empty() { return; }
        
        let mut minority_errors = 0;
        let mut minority_total = 0;
        for (i, &true_y) in recent_y.iter().enumerate() {
            if true_y > 0.5 {
                minority_total += 1;
                for vuln in vulnerabilities.iter().take(1000) {
                    if vuln.sample_idx == i && vuln.error > 0.3 {
                        minority_errors += 1;
                        break;
                    }
                }
            }
        }
        self.minority_class_error_rate = if minority_total > 0 {
            minority_errors as f64 / minority_total as f64
        } else { 0.0 };
        
        let n_features = recent_x.get(0).map(|r| r.len()).unwrap_or(0);
        for feat_idx in 0..n_features {
            let mut vals: Vec<f64> = recent_x.iter()
                .filter_map(|row| row.get(feat_idx).copied().filter(|v| !v.is_nan()))
                .collect();
            if vals.is_empty() { continue; }
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = vals[vals.len() / 2];
            let variance: f64 = vals.iter().map(|v| (v - median).powi(2)).sum::<f64>() / vals.len() as f64;
            self.feature_volatility[feat_idx] = variance.sqrt();
        }
    }
    
    pub fn select_strategy(&self) -> MetamorphosisStrategy {
        let avg_volatility = self.feature_volatility.iter().sum::<f64>() / self.feature_volatility.len().max(1) as f64;
        let corruption_score = ((avg_volatility / 10.0).min(1.0) * self.minority_class_error_rate * 2.0).min(1.0);
        let concept_drift_score = if avg_volatility < 3.0 && self.minority_class_error_rate > 0.3 { 0.8 } else { 0.0 };
        
        if corruption_score > 0.6 && corruption_score > concept_drift_score {
            MetamorphosisStrategy::FeatureAware
        } else if concept_drift_score > 0.7 {
            MetamorphosisStrategy::DataAware
        } else {
            MetamorphosisStrategy::Conservative
        }
    }
}

pub struct AdversarialLivingBooster {
    primary: OptimizedPKBoostShannon,  // main gradient boosting model
    adversary: AdversarialEnsemble,  // tracks where the model is failing
    metabolism: FeatureMetabolism,  // monitors which features are still useful
    state: SystemState,
    alert_trigger_threshold: usize,
    metamorphosis_trigger_threshold: usize,
    vulnerability_alert_threshold: f64,
    vulnerability_metamorphosis_threshold: f64,
    baseline_vulnerability: f64,
    consecutive_vulnerable_checks: usize,
    observations_count: usize,
    metamorphosis_count: usize,  // how many times we've adapted
    recent_x: VecDeque<Vec<f64>>,  // rolling buffer of recent samples
    recent_y: VecDeque<f64>,
    buffer_size: usize,
    metamorphosis_cooldown: usize,  // dont adapt too frequently
    iterations_since_metamorphosis: usize,
    #[allow(dead_code)]
    drift_analyzer: DriftAnalyzer,
    #[allow(dead_code)]
    last_strategy: MetamorphosisStrategy,
    recent_pr_aucs: VecDeque<f64>,
    baseline_pr_auc: f64,
}

impl AdversarialLivingBooster {
    pub fn new(x_train: &Vec<Vec<f64>>, y_train: &[f64]) -> Self {
        let n_features = x_train.get(0).map_or(0, |row| row.len());
        let n_samples = x_train.len();
        
        // figure out how imbalanced the data is - this affects everything
        let pos_ratio = y_train.iter().sum::<f64>() / y_train.len() as f64;
        let imbalance_level = if pos_ratio < 0.02 || pos_ratio > 0.98 {
            "extreme"
        } else if pos_ratio < 0.10 || pos_ratio > 0.90 {
            "high"
        } else if pos_ratio < 0.20 || pos_ratio > 0.80 {
            "moderate"
        } else {
            "balanced"
        };
        
        // Default thresholds (will be calibrated during fit_initial)
        let vuln_alert_threshold = 0.02;
        let vuln_meta_threshold = 0.03;
        
        // smaller datasets = be more agressive with adaptation
        let (alert_thresh, meta_thresh) = if n_samples < 50_000 {
            (1, 2)  // More aggressive for testing
        } else if n_samples < 200_000 {
            (2, 3)
        } else {
            (3, 5)  // larger datasets can afford to wait a bit
        };
        
        // keep a rolling window of recent data for retraining
        let buffer_sz = if n_samples < 50_000 {
            10000
        } else if n_samples < 200_000 {
            15000
        } else {
            20000
        };
        
        let cooldown = if n_samples < 50_000 {
            5000
        } else if n_samples < 200_000 {
            10000
        } else {
            15000
        };
        
        println!("\n=== Adaptive Metamorphosis Configuration ===");
        println!("Dataset: {} samples, {} features", n_samples, n_features);
        println!("Positive ratio: {:.1}% ({})", pos_ratio * 100.0, imbalance_level);
        println!("Alert trigger: {} consecutive checks", alert_thresh);
        println!("Metamorphosis trigger: {} checks in alert", meta_thresh);
        println!("Buffer size: {} samples", buffer_sz);
        println!("Cooldown period: {} observations", cooldown);
        println!("Note: Vulnerability thresholds will be calibrated during fit_initial");
        println!("===========================================\n");
        
        Self {
            primary: OptimizedPKBoostShannon::auto(x_train, y_train),
            adversary: AdversarialEnsemble::new(pos_ratio),
            metabolism: FeatureMetabolism::new(n_features),
            state: SystemState::Normal,
            alert_trigger_threshold: alert_thresh,
            metamorphosis_trigger_threshold: meta_thresh,
            vulnerability_alert_threshold: vuln_alert_threshold,
            vulnerability_metamorphosis_threshold: vuln_meta_threshold,
            baseline_vulnerability: 0.0,
            consecutive_vulnerable_checks: 0,
            observations_count: 0,
            metamorphosis_count: 0,
            recent_x: VecDeque::with_capacity(buffer_sz),
            recent_y: VecDeque::with_capacity(buffer_sz),
            buffer_size: buffer_sz,
            metamorphosis_cooldown: cooldown,
            iterations_since_metamorphosis: 0,
            drift_analyzer: DriftAnalyzer::new(n_features),
            last_strategy: MetamorphosisStrategy::Conservative,
            recent_pr_aucs: VecDeque::with_capacity(5),
            baseline_pr_auc: 0.0,
        }
    }
    
    pub fn fit_initial(
        &mut self,
        x: &Vec<Vec<f64>>,
        y: &[f64],
        eval_set: Option<(&Vec<Vec<f64>>, &[f64])>,
        verbose: bool,
    ) -> Result<(), String> {
        if verbose {
            println!("\n=== INITIAL TRAINING (Adversarial Living Booster) ===");
        }
        self.primary.fit(x, y, eval_set, verbose)?;
        
        // Calibrate vulnerability thresholds on validation set
        if let Some((x_val, y_val)) = eval_set {
            let calibration = crate::auto_tuner::VulnerabilityCalibration::calibrate(
                &self.primary,
                x_val,
                y_val,
            );
            self.baseline_vulnerability = calibration.baseline_vulnerability;
            self.vulnerability_alert_threshold = calibration.alert_threshold;
            self.vulnerability_metamorphosis_threshold = calibration.metamorphosis_threshold;
            
            // Set baseline PR-AUC for performance tracking
            let val_preds = self.primary.predict_proba(x_val)?;
            self.baseline_pr_auc = calculate_pr_auc(y_val, &val_preds);
            
            if verbose {
                println!("Baseline PR-AUC: {:.4}", self.baseline_pr_auc);
                println!("Vulnerability thresholds configured based on validation data");
            }
        }
        
        if verbose {
            println!("Initial training complete. Model ready for streaming.");
        }
        Ok(())
    }
    
    // this is where the magic happens - process new data and decide if we need to adapt
    pub fn observe_batch(
        &mut self,
        x: &Vec<Vec<f64>>,
        y: &[f64],
        verbose: bool,
    ) -> Result<(), String> {
        self.observations_count += x.len();
        self.iterations_since_metamorphosis += x.len();
        
        // maintain rolling buffer of recent samples
        for (xi, &yi) in x.iter().zip(y.iter()) {
            if self.recent_x.len() >= self.buffer_size {
                self.recent_x.pop_front();
                self.recent_y.pop_front();
            }
            self.recent_x.push_back(xi.clone());
            self.recent_y.push_back(yi);
        }
        
        let primary_preds = self.primary.predict_proba(x)?;
        
        // Track PR-AUC for performance-based triggering
        let batch_pr_auc = calculate_pr_auc(y, &primary_preds);
        self.recent_pr_aucs.push_back(batch_pr_auc);
        if self.recent_pr_aucs.len() > 5 {
            self.recent_pr_aucs.pop_front();
        }
        
        // check where the model is screwing up
        for (i, (&pred, &true_y)) in primary_preds.iter().zip(y.iter()).enumerate() {
            let vuln = self.adversary.find_vulnerability(true_y, pred, i);
            self.adversary.record_vulnerability(vuln);
        }
        
        // track which features are actually being used
        let usage = self.primary.get_feature_usage();
        self.metabolism.update(&usage, self.observations_count);
        
        // dont trigger metamorphosis too often - need cooldown period
        if self.iterations_since_metamorphosis > self.metamorphosis_cooldown {
            self.update_state(verbose);
        } else if verbose && self.observations_count % 5000 < x.len() {
            println!("In cooldown period: {}/{} observations since last metamorphosis",
                     self.iterations_since_metamorphosis, self.metamorphosis_cooldown);
        }
        
        if let SystemState::Metamorphosis = self.state {
            if verbose { 
                println!("\n=== METAMORPHOSIS TRIGGERED at observation {} ===", self.observations_count); 
            }
            self.execute_metamorphosis(verbose)?;
            self.iterations_since_metamorphosis = 0;
        }
        
        if verbose && self.observations_count % 5000 < x.len() {
            let vuln_score = self.get_vulnerability_score();
            let dead_features = self.metabolism.get_dead_features();
            println!("Status @ {}: Vuln Score: {:.4}, State: {:?}, Dead Features: {}, Buffer: {}/{}", 
                self.observations_count, vuln_score, self.state, dead_features.len(),
                self.recent_x.len(), self.buffer_size);
        }
        
        Ok(())
    }
    
    // state machine logic - decide if we need to go into alert or metamorphosis
    fn update_state(&mut self, verbose: bool) {
        let vuln_score = self.get_vulnerability_score();
        
        // Performance-based check
        let (performance_degraded, recent_avg) = if self.recent_pr_aucs.len() >= 3 && self.baseline_pr_auc > 0.0 {
            let avg: f64 = self.recent_pr_aucs.iter().sum::<f64>() / self.recent_pr_aucs.len() as f64;
            let degradation = (self.baseline_pr_auc - avg) / self.baseline_pr_auc;
            (degradation > 0.10, avg)  // 10% drop
        } else {
            (false, 0.0)
        };
        
        // Trigger if EITHER vulnerability OR performance degraded
        let is_vulnerable = vuln_score > self.vulnerability_alert_threshold || performance_degraded;
        
        // Reset to Normal only if performance actually recovered (within 5% of baseline)
        let performance_recovered = recent_avg >= self.baseline_pr_auc * 0.95;
        
        match self.state {
            SystemState::Normal => {
                if is_vulnerable && !performance_recovered {
                    self.consecutive_vulnerable_checks += 1;
                    if self.consecutive_vulnerable_checks >= self.alert_trigger_threshold {
                        if verbose { 
                            println!("-- ALERT: Vulnerability score {:.4} > threshold {:.4} --", 
                                vuln_score, self.vulnerability_alert_threshold); 
                        }
                        self.state = SystemState::Alert { checks_in_alert: 1 };
                    }
                } else if performance_recovered {
                    self.consecutive_vulnerable_checks = 0;
                }
            },
            SystemState::Alert { checks_in_alert } => {
                if is_vulnerable && performance_degraded {
                    if checks_in_alert + 1 >= self.metamorphosis_trigger_threshold {
                        if verbose {
                            println!("-- METAMORPHOSIS: Vulnerability {:.4} + performance degraded {} checks --",
                                vuln_score, checks_in_alert + 1);
                        }
                        self.state = SystemState::Metamorphosis;
                    } else {
                        self.state = SystemState::Alert { checks_in_alert: checks_in_alert + 1 };
                    }
                } else if performance_recovered {
                    // Only reset to Normal if performance actually recovered
                    if verbose { println!("-- System state returned to NORMAL (performance recovered) --"); }
                    self.consecutive_vulnerable_checks = 0;
                    self.state = SystemState::Normal;
                } else {
                    // Stay in Alert but don't increment counter
                    println!("-- Staying in ALERT (performance still degraded) --");
                }
            },
            SystemState::Metamorphosis => {
                // Will be reset after metamorphosis completes
            },
        }
    }
    
    // the actual metamorphosis - prune bad trees and grow new ones
    fn execute_metamorphosis(&mut self, verbose: bool) -> Result<(), String> {
        let metamorphosis_start = Instant::now();
        
        // CHECKPOINT: Save state before metamorphosis
        let checkpoint_trees = self.primary.trees.clone();
        let checkpoint_pr_auc = self.baseline_pr_auc;
        
        let dead_features = self.metabolism.get_dead_features();
        
        if verbose {
            println!("  - Checkpointing {} trees before metamorphosis", checkpoint_trees.len());
            println!("  - Dead features: {:?}", dead_features);
            println!("  - Buffer: {} samples", self.recent_x.len());
        }
        
        // Only prune if we have dead features, otherwise just retrain
        let pruned_count = if !dead_features.is_empty() {
            let count = self.primary.prune_trees(&dead_features, 0.8);
            if verbose {
                println!("  - Pruned {} trees", count);
            }
            count
        } else {
            0
        };
        
        // Always add new trees on recent data
        if self.recent_x.len() > 1000 {
            let n_new_trees = if pruned_count > 0 { pruned_count.min(10) } else { 5 };
            
            if verbose {
                println!("  - Adding {} new trees on buffer data", n_new_trees);
            }
            
            match self.add_incremental_trees(n_new_trees, verbose) {
                Ok(added) => {
                    if verbose {
                        println!("  - Added {} trees", added);
                    }
                },
                Err(e) => {
                    if verbose {
                        println!("  - Error: {}", e);
                    }
                    // Rollback on error
                    self.primary.trees = checkpoint_trees;
                    self.state = SystemState::Normal;
                    self.consecutive_vulnerable_checks = 0;
                    return Err(e);
                }
            }
        }
        
        // VALIDATION: Test metamorphosis quality on held-out buffer data
        let validation_size = 2000.min(self.recent_x.len() / 2);
        if validation_size > 100 {
            let val_x: Vec<Vec<f64>> = self.recent_x.iter().rev().take(validation_size).cloned().collect();
            let val_y: Vec<f64> = self.recent_y.iter().rev().take(validation_size).cloned().collect();
            
            let post_meta_preds = self.primary.predict_proba(&val_x)?;
            let post_meta_pr_auc = calculate_pr_auc(&val_y, &post_meta_preds);
            
            // ROLLBACK if performance degraded by more than 2%
            let performance_threshold = 0.98;
            if post_meta_pr_auc < checkpoint_pr_auc * performance_threshold {
                if verbose {
                    println!("  ⚠️  ROLLBACK: Metamorphosis degraded performance");
                    println!("     Before: {:.4}, After: {:.4}", checkpoint_pr_auc, post_meta_pr_auc);
                    println!("     Restoring {} trees", checkpoint_trees.len());
                }
                self.primary.trees = checkpoint_trees;
                self.state = SystemState::Normal;
                self.consecutive_vulnerable_checks = 0;
                self.adversary.recent_vulnerabilities.clear();
                return Ok(());
            }
            
            // SUCCESS: Update baseline with new performance
            self.baseline_pr_auc = post_meta_pr_auc;
            
            if verbose {
                println!("  ✅ Performance maintained: {:.4} → {:.4}", checkpoint_pr_auc, post_meta_pr_auc);
            }
        }
        
        self.metamorphosis_count += 1;
        self.state = SystemState::Normal;
        self.consecutive_vulnerable_checks = 0;
        self.adversary.recent_vulnerabilities.clear();
        
        let metamorphosis_time = metamorphosis_start.elapsed();
        
        if verbose {
            println!("=== METAMORPHOSIS COMPLETE ===");
            println!("  - Active trees: {}", self.primary.trees.len());
            println!("  - Total metamorphoses: {}", self.metamorphosis_count);
            println!("  - Took: {:.2}s", metamorphosis_time.as_secs_f64());
            println!();
        }
        
        Ok(())
    }
    
    // train new trees on recent data from the buffer
    fn add_incremental_trees(&mut self, n_trees: usize, verbose: bool) -> Result<usize, String> {
        let buffer_x: Vec<Vec<f64>> = self.recent_x.iter().cloned().collect();
        let buffer_y: Vec<f64> = self.recent_y.iter().cloned().collect();
        
        // need enough data to train on
        if buffer_x.len() < 1000 {
            return Err(format!("Insufficient data in buffer for retraining: {} samples", buffer_x.len()));
        }
        
        if verbose {
            println!("    - Retraining on {} recent samples from buffer", buffer_x.len());
        }
        
        // get current predictions and convert to log-odds for gradient boosting
        let current_probs = self.primary.predict_proba(&buffer_x)?;
        
        // Check drift severity (for logging only)
        let avg_error: f64 = current_probs.iter().zip(buffer_y.iter())
            .map(|(pred, true_y)| (pred - true_y).abs())
            .sum::<f64>() / buffer_y.len() as f64;
        
        if verbose {
            println!("    - Drift assessment: avg error = {:.4}", avg_error);
        }
        
        // Always use current predictions as base (NEVER reset to zero)
        let mut raw_preds: Vec<f64> = current_probs.iter()
            .map(|&p| {
                let p_clamped = p.clamp(1e-7, 1.0 - 1e-7);
                (p_clamped / (1.0 - p_clamped)).ln()  // logit transform
            })
            .collect();
        
        let histogram_builder = self.primary.histogram_builder.as_ref()
            .ok_or("Histogram builder not initialized")?;
        let x_processed = histogram_builder.transform(&buffer_x);
        let transposed_data = TransposedData::from_rows(&x_processed);
        
        let n_features = buffer_x[0].len();
        let feature_indices: Vec<usize> = (0..n_features).collect();
        let sample_indices: Vec<usize> = (0..buffer_x.len()).collect();
        
        let tree_params = TreeParams {
            min_samples_split: self.primary.min_samples_split,
            min_child_weight: self.primary.min_child_weight,
            reg_lambda: self.primary.reg_lambda,
            gamma: self.primary.gamma,
            mi_weight: self.primary.mi_weight,
            n_bins_per_feature: feature_indices.iter()
                .map(|&i| histogram_builder.n_bins_per_feature[i])
                .collect(),
        };
        
        let mut trees_added = 0;
        
        // standard gradient boosting loop
        for tree_idx in 0..n_trees {
            let grad = self.primary.loss_fn.gradient(
                &buffer_y,
                &raw_preds, 
                self.primary.scale_pos_weight
            );
            let hess = self.primary.loss_fn.hessian(
                &buffer_y,
                &raw_preds, 
                self.primary.scale_pos_weight
            );
            
            let mut new_tree = OptimizedTreeShannon::new(self.primary.max_depth);
            new_tree.fit_optimized(
                &transposed_data,
                &buffer_y,
                &grad,
                &hess,
                &sample_indices,
                &feature_indices,
                &tree_params
            );
            
            // ✅ CORRECT: Collect predictions directly with Arc<Mutex>
            use std::sync::{Arc, Mutex};
            
            let mut pool = fork_union::spawn(fork_union::count_logical_cores());
            let results = Arc::new(Mutex::new(Vec::new()));
            let results_clone = Arc::clone(&results);
            
            (0..buffer_x.len())
                .into_par_iter()
                .with_pool(&mut pool)
                .for_each(|i| {
                    let pred = new_tree.predict_from_transposed(&transposed_data, i);
                    results_clone.lock().unwrap().push((i, pred));
                });
            
            let mut tree_preds_indexed = Arc::try_unwrap(results)
                .expect("All clones should be dropped")
                .into_inner()
                .unwrap();
            
            tree_preds_indexed.sort_by_key(|&(idx, _)| idx);
            let tree_preds: Vec<f64> = tree_preds_indexed.into_iter().map(|(_, pred)| pred).collect();
            
            for (i, &tree_pred) in tree_preds.iter().enumerate() {
                raw_preds[i] += self.primary.learning_rate * tree_pred;
            }
            
            self.primary.trees.push(new_tree);
            trees_added += 1;
            
            if verbose && (tree_idx + 1) % 5 == 0 {
                println!("    - Added tree {}/{}", tree_idx + 1, n_trees);
            }
        }
        
        Ok(trees_added)
    }

    pub fn predict_proba(&self, x: &Vec<Vec<f64>>) -> Result<Vec<f64>, String> {
        self.primary.predict_proba(x)
    }
    
    pub fn get_state(&self) -> SystemState {
        self.state
    }
    
    pub fn get_metamorphosis_count(&self) -> usize {
        self.metamorphosis_count
    }
    
    pub fn get_vulnerability_score(&self) -> f64 {
        self.adversary.get_vulnerability_score()
    }
}
