// Main gradient boosting model with auto-tuning and adaptive features
// This is basically xgboost but with shannon entropy guidance and better handling of imbalanced data

use crate::auto_params::DataStats;
use crate::auto_tuner::auto_tune_principled;
use crate::{
    histogram_builder::OptimizedHistogramBuilder,
    loss::OptimizedShannonLoss,
    metrics::{calculate_pr_auc, calculate_roc_auc},
    optimized_data::TransposedData,
    tree::{OptimizedTreeShannon, TreeParams},
};
use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

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

    pub fn build_with_data(
        self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
    ) -> OptimizedPKBoostShannon {
        if self.use_auto_params {
            let stats = compute_data_stats(x, y.as_slice().unwrap());
            let imbalance = (1.0 - stats.pos_ratio) / stats.pos_ratio.max(1e-6); // class weight

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
                binned_data_cache: None,
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
                early_stopping_rounds: self
                    .early_stopping_rounds
                    .unwrap_or(model.early_stopping_rounds),
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
                binned_data_cache: None,
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
            binned_data_cache: None,
        }
    }
}

fn compute_data_stats(x: ArrayView2<'_, f64>, y: &[f64]) -> DataStats {
    let n_rows = x.nrows();
    let n_cols = x.ncols();

    let pos_count = y.iter().filter(|&&v| v > 0.5).count();

    let mut missing_count = 0;
    let mut max_cardinality = 0;

    for col_idx in 0..n_cols {
        let mut unique_vals = std::collections::HashSet::new();
        let mut col_missing = 0;

        for row_idx in 0..n_rows {
            let val = x[[row_idx, col_idx]];
            if val.is_nan() {
                col_missing += 1;
            } else {
                unique_vals.insert(val.to_bits());
            }
        }

        missing_count += col_missing;
        max_cardinality = max_cardinality.max(unique_vals.len());
    }

    DataStats::from_slices(n_rows, n_cols, pos_count, missing_count, max_cardinality)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OptimizedPKBoostShannon {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_child_weight: f64,
    pub reg_lambda: f64,       // L2 regularization
    pub gamma: f64,            // complexity penalty
    pub subsample: f64,        // row sampling ratio
    pub colsample_bytree: f64, // column sampling ratio
    pub early_stopping_rounds: usize,
    pub histogram_bins: usize,
    pub mi_weight: f64,        // weight for mutual information (entropy) term
    pub scale_pos_weight: f64, // upweight minority class
    pub trees: Vec<OptimizedTreeShannon>,
    base_score: f64,
    best_iteration: usize,
    #[serde(
        default = "default_best_score",
        skip_serializing_if = "is_neg_infinity"
    )]
    best_score: f64,
    fitted: bool,
    #[serde(default)]
    pub loss_fn: OptimizedShannonLoss,
    pub histogram_builder: Option<OptimizedHistogramBuilder>,
    auto_tuned: bool,
    pub metric_history: Vec<f64>, // for smoothed early stopping
    pub patience_counter: usize,
    #[serde(skip)] // Cache can be rebuilt, no need to serialize
    pub binned_data_cache: Option<TransposedData>,
}

fn default_best_score() -> f64 {
    f64::NEG_INFINITY
}

fn is_neg_infinity(value: &f64) -> bool {
    value.is_infinite() && value.is_sign_negative()
}

impl OptimizedPKBoostShannon {
    /// Create auto-tuned model from ArrayView2 (zero-copy from Python)
    pub fn auto(x: ArrayView2<'_, f64>, y: ArrayView1<'_, f64>) -> Self {
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

    /// Main training loop - standard gradient boosting with performance optimizations
    /// Accepts ArrayView2 for zero-copy data transfer from Python
    pub fn fit(
        &mut self,
        x: ArrayView2<'_, f64>,
        y: ArrayView1<'_, f64>,
        eval_set: Option<(ArrayView2<'_, f64>, ArrayView1<'_, f64>)>,
        verbose: bool,
    ) -> Result<(), String> {
        let fit_start_time = Instant::now();
        let n_samples = x.nrows();
        if n_samples == 0 {
            return Err("Input data is empty".to_string());
        }
        let n_features = x.ncols();
        let y_slice = y.as_slice().ok_or("y must be contiguous")?;

        let pos_ratio = y_slice.iter().sum::<f64>() / y_slice.len() as f64;

        if verbose {
            println!("=== PKBoost Training Started ===");
            println!("Dataset: {} samples, {} features", n_samples, n_features);
            println!("Positive ratio: {:.3}", pos_ratio);
            println!(
                "Hyperparams: lr={:.3}, depth={}, trees={}, scale_pos_weight={:.2}",
                self.learning_rate, self.max_depth, self.n_estimators, self.scale_pos_weight
            );
            if self.auto_tuned {
                println!("Mode: Auto-tuned");
                println!("  Learning rate  : {:.4}", self.learning_rate);
                println!("  Max depth      : {}", self.max_depth);
                println!(
                    "  Estimators     : {} (early stop @ {})",
                    self.n_estimators, self.early_stopping_rounds
                );
                println!("  Col sample     : {:.2}", self.colsample_bytree);
                println!("  Reg lambda     : {:.4}", self.reg_lambda);
                println!("  Min child wt   : {:.2}", self.min_child_weight);
                println!("  Gamma          : {:.2}", self.gamma);
                println!("  MI weight      : {:.2}", self.mi_weight);
            }
            println!(
                "Validation set: {}",
                if eval_set.is_some() { "Yes" } else { "No" }
            );
            println!();
        }

        self.base_score = self.loss_fn.init_score(y_slice);
        let mut train_preds = vec![self.base_score; n_samples]; // raw predictions (log-odds)

        // bin continuous features into histograms for faster splitting
        if self.histogram_builder.is_none() {
            if verbose {
                println!("Building histograms (one-time)...");
            }
            let mut histogram_builder = OptimizedHistogramBuilder::new(self.histogram_bins);
            histogram_builder.fit(x);
            self.histogram_builder = Some(histogram_builder);
        }

        let histogram_builder = self.histogram_builder.as_ref().unwrap();

        // 🔥 OPTIMIZATION: Transform directly to columnar u8 TransposedData
        // Eliminates intermediate row-major buffer, skips transposition, and pins bin edges in L1 cache
        self.binned_data_cache = Some(histogram_builder.transform_transposed_u8(x));
        let transposed_data = self.binned_data_cache.as_ref().unwrap();

        let (val_transposed, mut val_preds) = if let Some((x_val, y_val)) = eval_set {
            (
                Some(histogram_builder.transform_transposed_u8(x_val)),
                Some(vec![self.base_score; y_val.len()]),
            )
        } else {
            (None, None)
        };

        if verbose {
            println!("Histogram building complete. Starting boosting iterations...");
            if eval_set.is_some() {
                println!(
                    "{:<8} {:<10} {:<10} {:<8} {:<12} {:<10}",
                    "Iter", "Val-ROC", "Val-PR", "Time(s)", "Samples/sec", "Status"
                );
                println!("{}", "-".repeat(65));
            }
        }

        // Pre-allocate persistent buffers to avoid allocations per iteration
        let mut sample_indices: Vec<usize> = (0..n_samples).collect();
        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        let mut tree_preds_buf = vec![0.0f64; n_samples];
        let mut grad = vec![0.0f64; n_samples];
        let mut hess = vec![0.0f64; n_samples];

        // boosting iterations
        for iteration in 0..self.n_estimators {
            let mut rng = rand::thread_rng();

            // stochastic gradient boosting - subsample rows
            let sample_size = (self.subsample * n_samples as f64) as usize;
            if sample_size < n_samples {
                sample_indices.clear();
                // Fast sorted subsampling: coin flip / Bernoulli selection
                // Generates samples directly in increasing order without shuffle + sort!
                let p = self.subsample;
                for i in 0..n_samples {
                    if rng.gen::<f64>() < p {
                        sample_indices.push(i);
                    }
                }
                if sample_indices.is_empty() {
                    sample_indices.extend(0..n_samples);
                }
            } else if sample_indices.len() != n_samples {
                sample_indices.clear();
                sample_indices.extend(0..n_samples);
            }

            let feature_size = ((self.colsample_bytree * n_features as f64) as usize).max(1);
            if feature_size < n_features {
                feature_indices.clear();
                feature_indices.extend(0..n_features);
                feature_indices.shuffle(&mut rng);
                feature_indices.truncate(feature_size);
                feature_indices.sort_unstable();
            } else if feature_indices.len() != n_features {
                feature_indices.clear();
                feature_indices.extend(0..n_features);
            }

            // PHASE 1: Compute gradients and hessians in-place (zero allocation)
            self.loss_fn.compute_gradient_hessian_into(
                y_slice,
                &train_preds,
                self.scale_pos_weight,
                &mut grad,
                &mut hess,
            );

            let mut tree = OptimizedTreeShannon::new(self.max_depth);

            let tree_params = TreeParams {
                min_samples_split: self.min_samples_split,
                min_child_weight: self.min_child_weight,
                reg_lambda: self.reg_lambda,
                gamma: self.gamma,
                mi_weight: self.mi_weight,
                n_bins_per_feature: feature_indices
                    .iter()
                    .map(|&i| histogram_builder.n_bins_per_feature[i])
                    .collect(),
                feature_elimination_threshold: 0.01,
            };

            // PHASE 2 & 3: Tree fitting + direct leaf prediction reuse
            tree.fit_optimized_with_predictions(
                &transposed_data,
                y_slice,
                &grad,
                &hess,
                &sample_indices,
                &feature_indices,
                &tree_params,
                &mut tree_preds_buf,
            );

            // Guard against gradient collapse
            if (iteration + 1) % 100 == 0 {
                let grad_norm: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
                if grad_norm < 0.001 {
                    // Gradients have vanished - continuing would waste time
                    break;
                }
            }

            let adaptive_lr = self.learning_rate;

            // PHASE 4: Fused update and clamping in a single parallel pass
            train_preds
                .par_iter_mut()
                .zip(tree_preds_buf.par_iter())
                .for_each(|(current_pred, &tree_pred)| {
                    *current_pred = (*current_pred + adaptive_lr * tree_pred).clamp(-10.0, 10.0);
                });

            if let (Some(ref val_transposed_data), Some(ref mut val_preds_ref), Some((_, y_val))) =
                (val_transposed.as_ref(), val_preds.as_mut(), eval_set)
            {
                let y_val_slice = y_val.as_slice().unwrap();
                let val_sample_indices: Vec<usize> = (0..val_transposed_data.n_samples).collect();
                let val_tree_preds = tree.predict_batch(val_transposed_data, &val_sample_indices);

                val_preds_ref
                    .par_iter_mut()
                    .zip(val_tree_preds.par_iter())
                    .for_each(|(current_pred, tree_pred)| {
                        *current_pred += adaptive_lr * tree_pred;
                    });

                // evaluate on validation set periodically (less frequent = faster)
                if (iteration + 1) % 20 == 0 || iteration == 0 {
                    let val_probs = self.loss_fn.sigmoid(val_preds_ref);
                    let val_roc = calculate_roc_auc(y_val_slice, &val_probs);
                    let val_pr = calculate_pr_auc(y_val_slice, &val_probs); // PR-AUC better for imbalanced data

                    // smooth metrics over last 3 evaluations to reduce noise
                    if self.metric_history.len() >= 3 {
                        self.metric_history.remove(0);
                    }
                    self.metric_history.push(val_pr);

                    let smoothed_metric =
                        self.metric_history.iter().sum::<f64>() / self.metric_history.len() as f64;

                    if verbose && ((iteration + 1) % 20 == 0 || iteration == 0) {
                        let total_time = fit_start_time.elapsed().as_secs_f64();
                        let samples_per_sec =
                            (n_samples as f64 * (iteration + 1) as f64) / total_time;
                        println!(
                            "{:<8} {:<10.4} {:<10.4} {:<8.1} {:<12.0} {:<10}",
                            iteration + 1,
                            val_roc,
                            val_pr,
                            total_time,
                            samples_per_sec,
                            "Training"
                        );
                        println!(
                            "  ↳ Smoothed PR-AUC: {:.4} (best: {:.4} @ iter {})",
                            smoothed_metric,
                            self.best_score,
                            self.best_iteration + 1
                        );
                        use std::io::{self, Write};
                        let _ = io::stdout().flush();
                    }

                    // update best score if we improved
                    if smoothed_metric > self.best_score + 1e-5 {
                        self.best_score = smoothed_metric;
                        self.best_iteration = iteration;
                        self.patience_counter = 0;
                    } else {
                        self.patience_counter += 1;
                    }

                    // early stopping if no improvement
                    if self.patience_counter >= self.early_stopping_rounds {
                        if verbose {
                            println!(
                                "\nEarly stopping at iteration {} (no improvement for {} rounds)",
                                iteration + 1,
                                self.early_stopping_rounds
                            );
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

        // Always print training results - these are essential output
        let final_time = fit_start_time.elapsed().as_secs_f64();
        println!("\n=== Training Complete ===");
        println!("Best iteration: {}", self.best_iteration + 1);
        println!("Best score: {:.6}", self.best_score);
        println!("Total time: {:.2}s", final_time);
        println!("Final trees: {}", self.trees.len());

        Ok(())
    }

    /// Predict probabilities from ArrayView2 (zero-copy from Python)
    pub fn predict_proba(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>, String> {
        if !self.fitted {
            return Err("Model not fitted".to_string());
        }

        let histogram_builder = self.histogram_builder.as_ref().unwrap();
        let transposed_data = histogram_builder.transform_transposed_u8(x);
        let n_samples = transposed_data.n_samples;
        let sample_indices: Vec<usize> = (0..n_samples).collect();

        let mut predictions = vec![self.base_score; n_samples];

        for tree in &self.trees {
            let tree_preds = tree.predict_batch(&transposed_data, &sample_indices);
            predictions
                .par_iter_mut()
                .zip(tree_preds.par_iter())
                .for_each(|(current_pred, tree_pred)| {
                    *current_pred += self.learning_rate * tree_pred;
                });
        }

        Ok(Array1::from(self.loss_fn.sigmoid(&predictions)))
    }

    // OPTIMIZATION 5: Memory-mapped prediction for large datasets
    pub fn predict_proba_chunked(&self, x: ArrayView2<'_, f64>) -> Result<Array1<f64>, String> {
        if !self.fitted {
            return Err("Model not fitted".to_string());
        }

        let histogram_builder = self.histogram_builder.as_ref().unwrap();
        let transposed_data = histogram_builder.transform_transposed_u8(x);

        let n_samples = transposed_data.n_samples;
        let mut predictions = vec![self.base_score; n_samples];

        // OPTIMIZATION: Process trees in chunks for better cache utilization
        const TREE_CHUNK_SIZE: usize = 8;

        for chunk in self.trees.chunks(TREE_CHUNK_SIZE) {
            for tree in chunk {
                // Process all samples for this tree chunk
                for i in 0..n_samples {
                    predictions[i] +=
                        self.learning_rate * tree.predict_from_transposed(&transposed_data, i);
                }
            }
        }

        Ok(Array1::from(self.loss_fn.sigmoid(&predictions)))
    }

    pub fn predict_proba_batch(
        &self,
        x: ArrayView2<'_, f64>,
        batch_size: usize,
    ) -> Result<Array1<f64>, String> {
        if !self.fitted {
            return Err("Model not fitted".to_string());
        }

        let histogram_builder = self.histogram_builder.as_ref().unwrap();
        let transposed_data = histogram_builder.transform_transposed_u8(x);
        let mut all_predictions = Vec::with_capacity(transposed_data.n_samples);

        for batch_start in (0..transposed_data.n_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(transposed_data.n_samples);
            let mut batch_preds = vec![self.base_score; batch_end - batch_start];

            for tree in &self.trees {
                let tree_preds: Vec<f64> = (batch_start..batch_end)
                    .into_par_iter()
                    .map(|sample_idx| tree.predict_from_transposed(&transposed_data, sample_idx))
                    .collect();

                batch_preds
                    .par_iter_mut()
                    .zip(tree_preds.par_iter())
                    .for_each(|(current_pred, tree_pred)| {
                        *current_pred += self.learning_rate * tree_pred;
                    });
            }

            all_predictions.extend(batch_preds);
        }

        Ok(Array1::from(self.loss_fn.sigmoid(&all_predictions)))
    }

    // remove trees that depend heavily on dead features
    pub fn prune_trees(&mut self, dead_features: &[usize], threshold: f64) -> usize {
        let initial = self.trees.len();
        self.trees
            .retain(|tree| tree.feature_dependency_score(dead_features) < threshold);
        initial - self.trees.len()
    }

    pub fn get_feature_usage(&self) -> Vec<usize> {
        let n_features = self
            .histogram_builder
            .as_ref()
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
    x_train: ArrayView2<'_, f64>,
    y_train: ArrayView1<'_, f64>,
    x_val: Option<(ArrayView2<'_, f64>, ArrayView1<'_, f64>)>,
    verbose: bool,
) -> Result<OptimizedPKBoostShannon, String> {
    let mut model = OptimizedPKBoostShannon::auto(x_train, y_train);
    model.fit(x_train, y_train, x_val, verbose)?;
    Ok(model)
}

pub fn train_with_overrides(
    x_train: ArrayView2<'_, f64>,
    y_train: ArrayView1<'_, f64>,
    max_depth: Option<usize>,
    learning_rate: Option<f64>,
    verbose: bool,
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
