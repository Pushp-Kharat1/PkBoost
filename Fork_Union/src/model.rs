// Main gradient boosting model with auto-tuning and adaptive features
// This is basically xgboost but with shannon entropy guidance and better handling of imbalanced data

// use log::info;
use rand::prelude::*;
use crate::parallel::*;
use std::time::Instant;
use crate::auto_params::{ DataStats};
use crate::auto_tuner::auto_tune_principled;
use crate::{
    histogram_builder::OptimizedHistogramBuilder,
    loss::OptimizedShannonLoss,
    tree::{OptimizedTreeShannon, TreeParams},
    metrics::{calculate_roc_auc, calculate_pr_auc},
    optimized_data::TransposedData,
};

// Builder pattern for creating models with custom hyperparameters
pub struct PKBoostBuilder {
    n_estimators: Option<usize>,
    learning_rate: Option<f64>,
    max_depth: Option<usize>,
    min_samples_split: Option<usize>,
    min_child_weight: Option<f64>,
    reg_lambda: Option<f64>,
    gamma: Option<f64>,
    subsample: Option<f64>,
    colsample_bytree: Option<f64>,
    early_stopping_rounds: Option<usize>,
    histogram_bins: Option<usize>,
    mi_weight: Option<f64>,
    scale_pos_weight: Option<f64>,
    use_auto_params: bool,
}

impl Default for PKBoostBuilder {
    fn default() -> Self {
        Self {
            n_estimators: None,
            learning_rate: None,
            max_depth: None,
            min_samples_split: None,
            min_child_weight: None,
            reg_lambda: None,
            gamma: None,
            subsample: None,
            colsample_bytree: None,
            early_stopping_rounds: None,
            histogram_bins: None,
            mi_weight: None,
            scale_pos_weight: None,
            use_auto_params: false,
        }
    }
}

impl PKBoostBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn auto(mut self) -> Self {
        self.use_auto_params = true;
        self
    }

    pub fn n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = Some(n);
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = Some(lr);
        self
    }

    pub fn max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    pub fn min_samples_split(mut self, n: usize) -> Self {
        self.min_samples_split = Some(n);
        self
    }

    pub fn min_child_weight(mut self, weight: f64) -> Self {
        self.min_child_weight = Some(weight);
        self
    }

    pub fn reg_lambda(mut self, lambda: f64) -> Self {
        self.reg_lambda = Some(lambda);
        self
    }

    pub fn gamma(mut self, g: f64) -> Self {
        self.gamma = Some(g);
        self
    }

    pub fn subsample(mut self, ratio: f64) -> Self {
        self.subsample = Some(ratio);
        self
    }

    pub fn colsample_bytree(mut self, ratio: f64) -> Self {
        self.colsample_bytree = Some(ratio);
        self
    }

    pub fn early_stopping_rounds(mut self, rounds: usize) -> Self {
        self.early_stopping_rounds = Some(rounds);
        self
    }

    pub fn histogram_bins(mut self, bins: usize) -> Self {
        self.histogram_bins = Some(bins);
        self
    }

    pub fn mi_weight(mut self, weight: f64) -> Self {
        self.mi_weight = Some(weight);
        self
    }

    pub fn scale_pos_weight(mut self, weight: f64) -> Self {
        self.scale_pos_weight = Some(weight);
        self
    }

    pub fn build_with_data(self, x: &Vec<Vec<f64>>, y: &[f64]) -> OptimizedPKBoostShannon {
        if self.use_auto_params {
            let stats = compute_data_stats(x, y);
            let imbalance = (1.0 - stats.pos_ratio) / stats.pos_ratio.max(1e-6);  // class weight

            println!("\nAuto-Parameter Selection :");
            println!("  Dataset shape: {} × {}", stats.n_rows, stats.n_cols);
            println!("  Positive ratio: {:.3}", stats.pos_ratio);
            println!("  Missing rate: {:.3}", stats.missing_rate);

            let mut model = OptimizedPKBoostShannon {
                n_estimators: 0,
                learning_rate: 0.0,
                max_depth: 0,
                min_samples_split: 0,
                min_child_weight: 0.0,
                reg_lambda: 0.0,
                gamma: 0.0,
                subsample: 0.0,
                colsample_bytree: 0.0,
                early_stopping_rounds: 0,
                histogram_bins: 0,
                mi_weight: 0.0,
                scale_pos_weight: 0.0,
                trees: Vec::new(),
                base_score: 0.0,
                best_iteration: 0,
                best_score: f64::NEG_INFINITY,
                fitted: false,
                loss_fn: OptimizedShannonLoss::new(),
                histogram_builder: None,
                auto_tuned: true,
                metric_history: Vec::new(),
                patience_counter: 0,
            };

            // let the auto-tuner set all hyperparameters based on data characteristics
            auto_tune_principled(&mut model, stats.n_rows, stats.n_cols, stats.pos_ratio);
            model.scale_pos_weight = imbalance;

            OptimizedPKBoostShannon {
                n_estimators: self.n_estimators.unwrap_or(model.n_estimators),
                learning_rate: self.learning_rate.unwrap_or(model.learning_rate),
                max_depth: self.max_depth.unwrap_or(model.max_depth),
                min_samples_split: self.min_samples_split.unwrap_or(model.min_samples_split),
                min_child_weight: self.min_child_weight.unwrap_or(model.min_child_weight),
                reg_lambda: self.reg_lambda.unwrap_or(model.reg_lambda),
                gamma: self.gamma.unwrap_or(model.gamma),
                subsample: self.subsample.unwrap_or(model.subsample),
                colsample_bytree: self.colsample_bytree.unwrap_or(model.colsample_bytree),
                early_stopping_rounds: self.early_stopping_rounds.unwrap_or(model.early_stopping_rounds),
                histogram_bins: self.histogram_bins.unwrap_or(model.histogram_bins),
                mi_weight: self.mi_weight.unwrap_or(model.mi_weight),
                scale_pos_weight: self.scale_pos_weight.unwrap_or(model.scale_pos_weight),
                trees: model.trees,
                base_score: model.base_score,
                best_iteration: model.best_iteration,
                best_score: model.best_score,
                fitted: model.fitted,
                loss_fn: model.loss_fn,
                histogram_builder: model.histogram_builder,
                auto_tuned: model.auto_tuned,
                metric_history: model.metric_history,
                patience_counter: model.patience_counter,
            }
        } else {
            self.build()
        }
    }

    pub fn build(self) -> OptimizedPKBoostShannon {
        OptimizedPKBoostShannon {
            n_estimators: self.n_estimators.unwrap_or(1000),
            learning_rate: self.learning_rate.unwrap_or(0.05),
            max_depth: self.max_depth.unwrap_or(6),
            min_samples_split: self.min_samples_split.unwrap_or(100),
            min_child_weight: self.min_child_weight.unwrap_or(5.0),
            reg_lambda: self.reg_lambda.unwrap_or(2.0),
            gamma: self.gamma.unwrap_or(0.1),
            subsample: self.subsample.unwrap_or(0.8),
            colsample_bytree: self.colsample_bytree.unwrap_or(0.7),
            early_stopping_rounds: self.early_stopping_rounds.unwrap_or(75),
            histogram_bins: self.histogram_bins.unwrap_or(32),
            mi_weight: self.mi_weight.unwrap_or(0.3),
            scale_pos_weight: self.scale_pos_weight.unwrap_or(1.0),
            trees: Vec::new(),
            base_score: 0.0,
            best_iteration: 0,
            best_score: f64::NEG_INFINITY,
            fitted: false,
            loss_fn: OptimizedShannonLoss::new(),
            histogram_builder: None,
            auto_tuned: false,
            metric_history: Vec::new(),
            patience_counter: 0,
        }
    }
}

fn compute_data_stats(x: &Vec<Vec<f64>>, y: &[f64]) -> DataStats {
    let n_rows = x.len();
    let n_cols = if n_rows > 0 { x[0].len() } else { 0 };

    let pos_count = y.iter().filter(|&&v| v > 0.5).count();

    let mut missing_count = 0;
    let mut max_cardinality = 0;

    for col_idx in 0..n_cols {
        let mut unique_vals = std::collections::HashSet::new();
        let mut col_missing = 0;

        for row in x.iter() {
            if row[col_idx].is_nan() {
                col_missing += 1;
            } else {
                unique_vals.insert(row[col_idx].to_bits());
            }
        }

        missing_count += col_missing;
        max_cardinality = max_cardinality.max(unique_vals.len());
    }

    DataStats::from_slices(n_rows, n_cols, pos_count, missing_count, max_cardinality)
}

#[derive(Debug)]
pub struct OptimizedPKBoostShannon {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_child_weight: f64,
    pub reg_lambda: f64,  // L2 regularization
    pub gamma: f64,  // complexity penalty
    pub subsample: f64,  // row sampling ratio
    pub colsample_bytree: f64,  // column sampling ratio
    pub early_stopping_rounds: usize,
    pub histogram_bins: usize,
    pub mi_weight: f64,  // weight for mutual information (entropy) term
    pub scale_pos_weight: f64,  // upweight minority class
    pub trees: Vec<OptimizedTreeShannon>,
    base_score: f64,
    best_iteration: usize,
    best_score: f64,
    fitted: bool,
    pub loss_fn: OptimizedShannonLoss,
    pub histogram_builder: Option<OptimizedHistogramBuilder>,
    auto_tuned: bool,
    pub metric_history: Vec<f64>,  // for smoothed early stopping
    pub patience_counter: usize,
}

impl OptimizedPKBoostShannon {
    pub fn auto(x: &Vec<Vec<f64>>, y: &[f64]) -> Self {
        Self::builder().auto().build_with_data(x, y)
    }

    pub fn new() -> Self {
        Self::builder().build()
    }

    pub fn builder() -> PKBoostBuilder {
        PKBoostBuilder::new()
    }

    pub fn is_auto_tuned(&self) -> bool {
        self.auto_tuned
    }

    // main training loop - standard gradient boosting with some tricks
    pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &[f64], eval_set: Option<(&Vec<Vec<f64>>, &[f64])>, verbose: bool) -> Result<(), String> {
        let fit_start_time = Instant::now();
        let n_samples = x.len();
        if n_samples == 0 { return Err("Input data is empty".to_string()); }
        let n_features = x[0].len();

        let pos_ratio = y.iter().sum::<f64>() / y.len() as f64;

        if verbose {
            println!("=== PKBoost Training Started ===");
            println!("Dataset: {} samples, {} features", n_samples, n_features);
            println!("Positive ratio: {:.3}", pos_ratio);
            println!("Hyperparams: lr={:.3}, depth={}, trees={}, scale_pos_weight={:.2}",
                     self.learning_rate, self.max_depth, self.n_estimators, self.scale_pos_weight);
            println!("Validation set: {}", if eval_set.is_some() { "Yes" } else { "No" });
            println!();
        }

        self.base_score = self.loss_fn.init_score(y);
        let mut train_preds = vec![self.base_score; n_samples];  // raw predictions (log-odds)

        // bin continuous features into histograms for faster splitting
        if self.histogram_builder.is_none() {
            if verbose { println!("Building histograms (one-time)..."); }
            let mut histogram_builder = OptimizedHistogramBuilder::new(self.histogram_bins);
            histogram_builder.fit(x);
            self.histogram_builder = Some(histogram_builder);
        }
        
        let histogram_builder = self.histogram_builder.as_ref().unwrap();
        let x_processed = histogram_builder.transform(x);
        let transposed_data = TransposedData::from_rows(&x_processed);

        let (x_val_processed, mut val_preds) = if let Some((x_val, y_val)) = eval_set {
            (Some(histogram_builder.transform(x_val)), Some(vec![self.base_score; y_val.len()]))
        } else { (None, None) };

        let val_transposed = if let Some(ref x_val_proc) = x_val_processed {
            Some(TransposedData::from_rows(x_val_proc))
        } else {
            None
        };

        if verbose {
            println!("Histogram building complete. Starting boosting iterations...");
            if eval_set.is_some() {
                println!("{:<8} {:<10} {:<10} {:<8} {:<12} {:<10}",
                         "Iter", "Val-ROC", "Val-PR", "Time(s)", "Samples/sec", "Status");
                println!("{}", "-".repeat(65));
            }
        }

        // boosting iterations
        for iteration in 0..self.n_estimators {
            let mut rng = rand::thread_rng();

            // stochastic gradient boosting - subsample rows
            let sample_size = (self.subsample * n_samples as f64) as usize;
            let mut sample_indices: Vec<usize> = (0..n_samples).collect();
            sample_indices.shuffle(&mut rng);
            sample_indices.truncate(sample_size);

            // subsample columns too
            let feature_size = ((self.colsample_bytree * n_features as f64) as usize).max(1);
            let mut feature_indices: Vec<usize> = (0..n_features).collect();
            feature_indices.shuffle(&mut rng);
            feature_indices.truncate(feature_size);

            // compute gradients and hessians for newton boosting
            let grad = self.loss_fn.gradient(y, &train_preds, self.scale_pos_weight);
            let hess = self.loss_fn.hessian(y, &train_preds, self.scale_pos_weight);

            let mut tree = OptimizedTreeShannon::new(self.max_depth);

            let tree_params = TreeParams {
                min_samples_split: self.min_samples_split,
                min_child_weight: self.min_child_weight,
                reg_lambda: self.reg_lambda,
                gamma: self.gamma,
                mi_weight: self.mi_weight,
                n_bins_per_feature: feature_indices.iter().map(|&i|
                    histogram_builder.n_bins_per_feature[i]
                ).collect()
            };

            tree.fit_optimized(
                &transposed_data,
                &y,
                &grad,
                &hess,
                &sample_indices,
                &feature_indices,
                &tree_params
            );

            if (iteration + 1) % 25 == 0 && verbose {
    let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
    let grad_mean: f64 = grad.iter().sum::<f64>() / grad.len() as f64;
    let pred_min = train_preds.iter().cloned().fold(f64::INFINITY, f64::min);
    let pred_max = train_preds.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let pred_mean = train_preds.iter().sum::<f64>() / train_preds.len() as f64;

    eprintln!("DEBUG Iter {}: grad_norm={:.4}, grad_mean={:.6}, pred_range=[{:.2}, {:.2}], pred_mean={:.4}",
             iteration + 1, grad_norm, grad_mean, pred_min, pred_max, pred_mean);

    if grad_norm > 10000.0 {
        eprintln!("  WARNING: Gradient explosion detected!");
    }
    if pred_max > 50.0 || pred_min < -50.0 {
        eprintln!("  WARNING: Prediction values out of safe range!");
    }
    
    if grad_norm < 0.001 {
        eprintln!("  WARNING: Gradients nearly zero - stopping early");
        return Ok(());
    }
}

            let tree_preds: Vec<f64> = par_map_range(0..n_samples, |sample_idx| {
                tree.predict_from_transposed(&transposed_data, sample_idx)
            });

            // update predictions with new tree
            for (current_pred, tree_pred) in train_preds.iter_mut().zip(tree_preds.iter()) {
                *current_pred += self.learning_rate * tree_pred;
            }

            // clamp to prevent numerical instability
            for pred in train_preds.iter_mut() {
                *pred = pred.clamp(-10.0, 10.0);
            }

            if let (Some(ref val_transposed_data), Some(ref mut val_preds_ref), Some((_, y_val))) =
                (val_transposed.as_ref(), val_preds.as_mut(), eval_set)
            {
                let val_tree_preds: Vec<f64> = par_map_range(0..val_transposed_data.n_samples, |sample_idx| {
                    tree.predict_from_transposed(val_transposed_data, sample_idx)
                });

                for (current_pred, tree_pred) in val_preds_ref.iter_mut().zip(val_tree_preds.iter()) {
                    *current_pred += self.learning_rate * tree_pred;
                }
                
                // evaluate on validation set periodically
if (iteration + 1) % 10 == 0 || iteration == 0 {
    let val_probs = self.loss_fn.sigmoid(val_preds_ref);
    let val_roc = calculate_roc_auc(y_val, &val_probs);
    let val_pr = calculate_pr_auc(y_val, &val_probs);  // PR-AUC better for imbalanced data
    
    // smooth metrics over last 3 evaluations to reduce noise
    if self.metric_history.len() >= 3 {
        self.metric_history.remove(0);
    }
    self.metric_history.push(val_pr);
    
    let smoothed_metric = self.metric_history.iter().sum::<f64>() / self.metric_history.len() as f64;
    
    if verbose && ((iteration + 1) % 10 == 0 || iteration == 0) {
        let total_time = fit_start_time.elapsed().as_secs_f64();
        let samples_per_sec = (n_samples as f64 * (iteration + 1) as f64) / total_time;
        println!("{:<8} {:<10.4} {:<10.4} {:<8.1} {:<12.0} {:<10}",
                 iteration + 1, val_roc, val_pr, total_time, samples_per_sec, "Training");
        println!("  ↳ Smoothed PR-AUC: {:.4} (best: {:.4} @ iter {})", 
                 smoothed_metric, self.best_score, self.best_iteration + 1);
        use std::io::{self, Write};
        let _ = io::stdout().flush();
    }
    
    // update best score if we improved
    if smoothed_metric > self.best_score + 1e-5 {
        self.best_score = smoothed_metric;
        self.best_iteration = iteration;
    }
    
    // early stopping if no improvement
    if iteration - self.best_iteration >= self.early_stopping_rounds {
        if verbose { 
            println!("\nEarly stopping at iteration {} (no improvement for {} rounds)", 
                     iteration + 1, self.early_stopping_rounds); 
        }
        break;
    }
}

            } else {
                self.best_iteration = iteration;
            }

            self.trees.push(tree);
        }

        self.fitted = true;
        self.trees.truncate(self.best_iteration + 1);

        if verbose {
            let final_time = fit_start_time.elapsed().as_secs_f64();
            println!("\n=== Training Complete ===");
            println!("Best iteration: {}", self.best_iteration + 1);
            println!("Best score: {:.6}", self.best_score);
            println!("Total time: {:.2}s", final_time);
            println!("Final trees: {}", self.trees.len());
        }

        Ok(())
    }

    pub fn predict_proba(&self, x: &Vec<Vec<f64>>) -> Result<Vec<f64>, String> {
        if !self.fitted { return Err("Model not fitted".to_string()); }

        let histogram_builder = self.histogram_builder.as_ref().unwrap();
        let x_proc = histogram_builder.transform(x);
        let transposed_data = TransposedData::from_rows(&x_proc);

        let mut predictions = vec![self.base_score; x_proc.len()];

        for tree in &self.trees {
            for (idx, current_pred) in predictions.iter_mut().enumerate() {
                *current_pred += self.learning_rate * tree.predict_from_transposed(&transposed_data, idx);
            }
        }
        
        Ok(self.loss_fn.sigmoid(&predictions))
    }

    pub fn predict_proba_batch(&self, x: &Vec<Vec<f64>>, batch_size: usize) -> Result<Vec<f64>, String> {
        if !self.fitted { return Err("Model not fitted".to_string()); }

        let histogram_builder = self.histogram_builder.as_ref().unwrap();
        let x_proc = histogram_builder.transform(x);
        let transposed_data = TransposedData::from_rows(&x_proc);
        let mut all_predictions = Vec::with_capacity(x_proc.len());

        for batch_start in (0..transposed_data.n_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(transposed_data.n_samples);
            let mut batch_preds = vec![self.base_score; batch_end - batch_start];

            for tree in &self.trees {
                let tree_preds: Vec<f64> = par_map_range(batch_start..batch_end, |sample_idx| {
                    tree.predict_from_transposed(&transposed_data, sample_idx)
                });

                for (current_pred, tree_pred) in batch_preds.iter_mut().zip(tree_preds.iter()) {
                    *current_pred += self.learning_rate * tree_pred;
                }
            }

            all_predictions.extend(batch_preds);
        }

        Ok(self.loss_fn.sigmoid(&all_predictions))
    }

    // remove trees that depend heavily on dead features
    pub fn prune_trees(&mut self, dead_features: &[usize], threshold: f64) -> usize {
        let initial = self.trees.len();
        self.trees.retain(|tree| {
            tree.feature_dependency_score(dead_features) < threshold
        });
        initial - self.trees.len()
    }
    
    pub fn get_feature_usage(&self) -> Vec<usize> {
        let n_features = self.histogram_builder.as_ref()
            .map(|h| h.n_bins_per_feature.len())
            .unwrap_or(0);
        let mut usage = vec![0; n_features];
        
        for tree in &self.trees {
            for &feat in &tree.get_used_features() {
                if feat < usage.len() {
                    usage[feat] += 1;
                }
            }
        }
        usage
    }
}

impl Default for OptimizedPKBoostShannon {
    fn default() -> Self {
        Self::new()
    }
}

pub fn quick_train(
    x_train: &Vec<Vec<f64>>,
    y_train: &[f64],
    x_val: Option<(&Vec<Vec<f64>>, &[f64])>,
    verbose: bool
) -> Result<OptimizedPKBoostShannon, String> {
    let mut model = OptimizedPKBoostShannon::auto(x_train, y_train);
    model.fit(x_train, y_train, x_val, verbose)?;
    Ok(model)
}

pub fn train_with_overrides(
    x_train: &Vec<Vec<f64>>,
    y_train: &[f64],
    max_depth: Option<usize>,
    learning_rate: Option<f64>,
    verbose: bool
) -> Result<OptimizedPKBoostShannon, String> {
    let mut builder = OptimizedPKBoostShannon::builder().auto();

    if let Some(depth) = max_depth {
        builder = builder.max_depth(depth);
    }

    if let Some(lr) = learning_rate {
        builder = builder.learning_rate(lr);
    }

    let mut model = builder.build_with_data(x_train, y_train);
    model.fit(x_train, y_train, None, verbose)?;
    Ok(model)
}
