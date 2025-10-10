// This is the main "living" model that can adapt to data drift in real-time
// The idea is to detect when the model starts failing and trigger a metamorphosis
// to prune bad trees and grow new ones on recent data

use crate::model::OptimizedPKBoostShannon;
use crate::adversarial::AdversarialEnsemble;
use crate::metabolism::FeatureMetabolism;
use crate::tree::{OptimizedTreeShannon, TreeParams};
use crate::optimized_data::TransposedData;
use rayon::prelude::*;
use std::time::Instant;
use std::collections::VecDeque;

// State machine for the model - tracks if we're doing ok or need to adapt
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SystemState {
    Normal,  // everything's fine
    Alert { checks_in_alert: usize },  // performance is degrading
    Metamorphosis,  // time to rebuild parts of the model
}

pub struct AdversarialLivingBooster {
    primary: OptimizedPKBoostShannon,  // main gradient boosting model
    adversary: AdversarialEnsemble,  // tracks where the model is failing
    metabolism: FeatureMetabolism,  // monitors which features are still useful
    state: SystemState,
    alert_trigger_threshold: usize,
    metamorphosis_trigger_threshold: usize,
    vulnerability_threshold: f64,  // how bad does performance need to get?
    consecutive_vulnerable_checks: usize,
    observations_count: usize,
    metamorphosis_count: usize,  // how many times we've adapted
    recent_x: VecDeque<Vec<f64>>,  // rolling buffer of recent samples
    recent_y: VecDeque<f64>,
    buffer_size: usize,
    metamorphosis_cooldown: usize,  // dont adapt too frequently
    iterations_since_metamorphosis: usize,
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
        
        // more imbalanced data = lower threshold for triggering adaptation
        // because even small changes can be catastrophic
        let vuln_threshold = match imbalance_level {
            "extreme" => 0.01,
            "high" => 0.0145,
            "moderate" => 0.02,
            _ => 0.025
        };
        
        // smaller datasets = be more agressive with adaptation
        let (alert_thresh, meta_thresh) = if n_samples < 50_000 {
            (2, 3)
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
        println!("Vulnerability threshold: {:.4}", vuln_threshold);
        println!("Alert trigger: {} consecutive checks", alert_thresh);
        println!("Metamorphosis trigger: {} checks in alert", meta_thresh);
        println!("Buffer size: {} samples", buffer_sz);
        println!("Cooldown period: {} observations", cooldown);
        println!("===========================================\n");
        
        Self {
            primary: OptimizedPKBoostShannon::auto(x_train, y_train),
            adversary: AdversarialEnsemble::new(pos_ratio),
            metabolism: FeatureMetabolism::new(n_features),
            state: SystemState::Normal,
            alert_trigger_threshold: alert_thresh,
            metamorphosis_trigger_threshold: meta_thresh,
            vulnerability_threshold: vuln_threshold,
            consecutive_vulnerable_checks: 0,
            observations_count: 0,
            metamorphosis_count: 0,
            recent_x: VecDeque::with_capacity(buffer_sz),
            recent_y: VecDeque::with_capacity(buffer_sz),
            buffer_size: buffer_sz,
            metamorphosis_cooldown: cooldown,
            iterations_since_metamorphosis: 0,
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
        let is_vulnerable = vuln_score > self.vulnerability_threshold;
        
        match self.state {
            SystemState::Normal => {
                if is_vulnerable {
                    self.consecutive_vulnerable_checks += 1;
                    if self.consecutive_vulnerable_checks >= self.alert_trigger_threshold {
                        if verbose { 
                            println!("-- System state changed to ALERT after {} consecutive vulnerable checks --", 
                                self.consecutive_vulnerable_checks); 
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
                        self.state = SystemState::Metamorphosis;
                    } else {
                        self.state = SystemState::Alert { checks_in_alert: checks_in_alert + 1 };
                    }
                } else {
                    if verbose { println!("-- System state returned to NORMAL --"); }
                    self.consecutive_vulnerable_checks = 0;
                    self.state = SystemState::Normal;
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
        
        // find features that havent been used in a while
        let dead_features = self.metabolism.get_dead_features();
        
        if verbose {
            println!("  - Analyzing feature health...");
            println!("    Dead features: {:?}", dead_features);
            println!("    Buffer contains {} recent samples", self.recent_x.len());
        }
        
        // Step 1: Prune trees dependent on dead features (less aggressive)
        // remove trees that rely heavily on dead features
        let pruned_count = if !dead_features.is_empty() {
            let count = self.primary.prune_trees(&dead_features, 0.8);
            if verbose {
                println!("  - Pruned {} trees dependent on dead features.", count);
            }
            count
        } else {
            0
        };
        
        // grow new trees to replace the ones we pruned
        if pruned_count > 0 {
            let n_new_trees = pruned_count.min(10);  // dont go crazy
            
            if verbose {
                println!("  - Adding {} replacement trees using recent buffer data...", n_new_trees);
            }
            
            match self.add_incremental_trees(n_new_trees, verbose) {
                Ok(added) => {
                    if verbose {
                        println!("  - Successfully added {} new trees.", added);
                    }
                },
                Err(e) => {
                    if verbose {
                        println!("  - Warning: Failed to add new trees: {}", e);
                        println!("  - Continuing with pruned model.");
                    }
                }
            }
        }
        
        self.metamorphosis_count += 1;
        self.state = SystemState::Normal;
        self.consecutive_vulnerable_checks = 0;
        self.adversary.recent_vulnerabilities.clear();
        
        let metamorphosis_time = metamorphosis_start.elapsed();
        
        if verbose {
            println!("=== METAMORPHOSIS COMPLETE ===");
            println!("  - Pruned: {} trees", pruned_count);
            println!("  - Active trees: {}", self.primary.trees.len());
            println!("  - Total metamorphoses: {}", self.metamorphosis_count);
            println!("  - Metamorphosis took: {:.2}s", metamorphosis_time.as_secs_f64());
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
            
            // get predictions from new tree and update ensemble
            let tree_preds: Vec<f64> = (0..buffer_x.len())
                .into_par_iter()
                .map(|i| new_tree.predict_from_transposed(&transposed_data, i))
                .collect();
            
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
