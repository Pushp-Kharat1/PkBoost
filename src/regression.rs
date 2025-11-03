// Regression support for PKBoost
// Uses MSE loss with L2 regularization

use rayon::prelude::*;
use crate::{
    histogram_builder::OptimizedHistogramBuilder,
    tree::{OptimizedTreeShannon, TreeParams},
    optimized_data::TransposedData,
    huber_loss::HuberLoss,
};

#[derive(Debug, Clone, Copy)]
pub enum RegressionLossType {
    MSE,
    Huber { delta: f64 },
}

#[derive(Debug)]
pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self { Self }
    
    pub fn gradient(&self, y_true: &[f64], y_pred: &[f64]) -> Vec<f64> {
        y_pred.par_iter().zip(y_true.par_iter())
            .map(|(&pred, &true_y)| pred - true_y)
            .collect()
    }
    
    pub fn hessian(&self, y_true: &[f64]) -> Vec<f64> {
        vec![1.0; y_true.len()]
    }
    
    pub fn init_score(&self, y_true: &[f64]) -> f64 {
        y_true.iter().sum::<f64>() / y_true.len() as f64
    }
}

pub fn detect_outliers(y: &[f64]) -> f64 {
    if y.len() < 4 { return 0.0; }
    let mut sorted = y.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q1 = sorted[sorted.len() / 4];
    let q3 = sorted[3 * sorted.len() / 4];
    let iqr = q3 - q1;
    if iqr < 1e-10 { return 0.0; }
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;
    y.iter().filter(|&&v| v < lower || v > upper).count() as f64 / y.len() as f64
}

pub fn calculate_mad(y: &[f64]) -> f64 {
    if y.is_empty() { return 1.0; }
    let median = {
        let mut sorted = y.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };
    let mut abs_devs: Vec<f64> = y.iter().map(|&v| (v - median).abs()).collect();
    abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    abs_devs[abs_devs.len() / 2]
}

#[derive(Debug)]
pub struct PKBoostRegressor {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_child_weight: f64,
    pub reg_lambda: f64,
    pub gamma: f64,
    pub subsample: f64,
    pub colsample_bytree: f64,
    pub early_stopping_rounds: usize,
    pub histogram_bins: usize,
    pub trees: Vec<OptimizedTreeShannon>,
    base_score: f64,
    best_iteration: usize,
    best_score: f64,
    fitted: bool,
    pub loss_type: RegressionLossType,
    pub loss_fn: MSELoss,
    pub huber_loss: Option<HuberLoss>,
    pub histogram_builder: Option<OptimizedHistogramBuilder>,
}

impl PKBoostRegressor {
    pub fn new() -> Self {
        Self {
            n_estimators: 1000,
            learning_rate: 0.05,
            max_depth: 6,
            min_samples_split: 100,
            min_child_weight: 1.0,
            reg_lambda: 1.0,
            gamma: 0.0,
            subsample: 0.8,
            colsample_bytree: 0.8,
            early_stopping_rounds: 50,
            histogram_bins: 32,
            trees: Vec::new(),
            base_score: 0.0,
            best_iteration: 0,
            best_score: f64::INFINITY,
            fitted: false,
            loss_type: RegressionLossType::MSE,
            loss_fn: MSELoss::new(),
            huber_loss: None,
            histogram_builder: None,
        }
    }
    
    pub fn auto(x: &Vec<Vec<f64>>, y: &[f64]) -> Self {
        let mut model = Self::new();
        let n_samples = x.len();
        
        // Auto-select loss based on outliers
        let outlier_ratio = detect_outliers(y);
        if outlier_ratio > 0.05 {
            let delta = calculate_mad(y) * 1.35;
            model.loss_type = RegressionLossType::Huber { delta };
            model.huber_loss = Some(HuberLoss::new(delta));
        }
        
        model.learning_rate = if n_samples < 5000 { 0.1 } else { 0.05 };
        model.max_depth = ((x[0].len() as f64).ln() as usize + 3).clamp(4, 8);
        model.reg_lambda = (x[0].len() as f64).sqrt() * 0.1;
        model.n_estimators = ((n_samples as f64).ln() as usize * 100).clamp(200, 2000);
        
        model
    }
    
    pub fn get_gradient(&self, y_true: &[f64], y_pred: &[f64]) -> Vec<f64> {
        match self.loss_type {
            RegressionLossType::MSE => self.loss_fn.gradient(y_true, y_pred),
            RegressionLossType::Huber { .. } => {
                self.huber_loss.as_ref().unwrap().gradient(y_true, y_pred)
            }
        }
    }
    
    pub fn get_hessian(&self, y_true: &[f64], y_pred: &[f64]) -> Vec<f64> {
        match self.loss_type {
            RegressionLossType::MSE => self.loss_fn.hessian(y_true),
            RegressionLossType::Huber { .. } => {
                self.huber_loss.as_ref().unwrap().hessian(y_true, y_pred)
            }
        }
    }
    
    pub fn fit(&mut self, x: &Vec<Vec<f64>>, y: &[f64], eval_set: Option<(&Vec<Vec<f64>>, &[f64])>, verbose: bool) -> Result<(), String> {
        let n_samples = x.len();
        if n_samples == 0 { return Err("Empty data".to_string()); }
        let n_features = x[0].len();
        
        self.base_score = self.loss_fn.init_score(y);
        let mut train_preds = vec![self.base_score; n_samples];
        
        if self.histogram_builder.is_none() {
            let mut hb = OptimizedHistogramBuilder::new(self.histogram_bins);
            hb.fit(x);
            self.histogram_builder = Some(hb);
        }
        
        let hb = self.histogram_builder.as_ref().unwrap();
        let x_proc = hb.transform(x);
        let transposed = TransposedData::from_rows(&x_proc);
        
        let (x_val_proc, mut val_preds) = if let Some((xv, yv)) = eval_set {
            (Some(hb.transform(xv)), Some(vec![self.base_score; yv.len()]))
        } else { (None, None) };
        
        let val_trans = x_val_proc.as_ref().map(|xv| TransposedData::from_rows(xv));
        
        if verbose {
            println!("=== PKBoost Regressor Training ===");
            println!("Samples: {}, Features: {}", n_samples, n_features);
        }
        
        for iter in 0..self.n_estimators {
            let mut rng = rand::thread_rng();
            let sample_size = (self.subsample * n_samples as f64) as usize;
            let mut sample_indices: Vec<usize> = (0..n_samples).collect();
            use rand::seq::SliceRandom;
            sample_indices.shuffle(&mut rng);
            sample_indices.truncate(sample_size);
            
            let feature_size = ((self.colsample_bytree * n_features as f64) as usize).max(1);
            let mut feature_indices: Vec<usize> = (0..n_features).collect();
            feature_indices.shuffle(&mut rng);
            feature_indices.truncate(feature_size);
            
            let grad = self.get_gradient(y, &train_preds);
            let hess = self.get_hessian(y, &train_preds);
            
            let mut tree = OptimizedTreeShannon::new(self.max_depth);
            let params = TreeParams {
                min_samples_split: self.min_samples_split,
                min_child_weight: self.min_child_weight,
                reg_lambda: self.reg_lambda,
                gamma: self.gamma,
                mi_weight: 0.3,  // Use variance reduction for regression
                n_bins_per_feature: feature_indices.iter().map(|&i| hb.n_bins_per_feature[i]).collect()
            };
            
            tree.fit_optimized(&transposed, y, &grad, &hess, &sample_indices, &feature_indices, &params);
            
            let tree_preds: Vec<f64> = (0..n_samples).into_par_iter()
                .map(|i| tree.predict_from_transposed(&transposed, i))
                .collect();
            
            train_preds.par_iter_mut().zip(tree_preds.par_iter())
                .for_each(|(p, &tp)| *p += self.learning_rate * tp);
            
            if let (Some(ref vt), Some(ref mut vp), Some((_, yv))) = (val_trans.as_ref(), val_preds.as_mut(), eval_set) {
                let val_tree_preds: Vec<f64> = (0..vt.n_samples).into_par_iter()
                    .map(|i| tree.predict_from_transposed(vt, i))
                    .collect();
                
                vp.par_iter_mut().zip(val_tree_preds.par_iter())
                    .for_each(|(p, &tp)| *p += self.learning_rate * tp);
                
                if (iter + 1) % 10 == 0 {
                    let mse: f64 = vp.iter().zip(yv.iter())
                        .map(|(p, y)| (p - y).powi(2))
                        .sum::<f64>() / yv.len() as f64;
                    let rmse = mse.sqrt();
                    
                    if verbose && (iter + 1) % 50 == 0 {
                        println!("Iter {}: RMSE = {:.4}", iter + 1, rmse);
                    }
                    
                    if rmse < self.best_score {
                        self.best_score = rmse;
                        self.best_iteration = iter;
                    }
                    
                    if iter - self.best_iteration >= self.early_stopping_rounds {
                        if verbose { println!("Early stopping at iter {}", iter + 1); }
                        break;
                    }
                }
            } else {
                self.best_iteration = iter;
            }
            
            self.trees.push(tree);
        }
        
        self.fitted = true;
        
        if eval_set.is_some() {
            self.trees.truncate(self.best_iteration + 1);
        }
        
        let final_train_rmse = calculate_rmse(y, &train_preds);
        
        if verbose {
            if eval_set.is_some() {
                println!("Training complete. Trees: {}, Best Val RMSE: {:.4}, Train RMSE: {:.4}", 
                    self.trees.len(), self.best_score, final_train_rmse);
            } else {
                println!("Training complete. Trees: {}, Train RMSE: {:.4}", 
                    self.trees.len(), final_train_rmse);
            }
        }
        
        Ok(())
    }
    
    pub fn predict(&self, x: &Vec<Vec<f64>>) -> Result<Vec<f64>, String> {
        if !self.fitted { return Err("Model not fitted".to_string()); }
        
        let hb = self.histogram_builder.as_ref().unwrap();
        let x_proc = hb.transform(x);
        let transposed = TransposedData::from_rows(&x_proc);
        
        let mut preds = vec![self.base_score; x_proc.len()];
        
        for tree in &self.trees {
            preds.par_iter_mut().enumerate()
                .for_each(|(i, p)| *p += self.learning_rate * tree.predict_from_transposed(&transposed, i));
        }
        
        Ok(preds)
    }
    
    pub fn predict_with_uncertainty(&self, x: &Vec<Vec<f64>>) -> Result<(Vec<f64>, Vec<f64>), String> {
        if !self.fitted { return Err("Model not fitted".to_string()); }
        
        let hb = self.histogram_builder.as_ref().unwrap();
        let x_proc = hb.transform(x);
        let transposed = TransposedData::from_rows(&x_proc);
        let n_samples = x_proc.len();
        
        let mut cumulative_preds: Vec<Vec<f64>> = vec![Vec::with_capacity(self.trees.len()); n_samples];
        
        for preds in &mut cumulative_preds {
            preds.push(self.base_score);
        }
        
        for tree in &self.trees {
            for sample_idx in 0..n_samples {
                let tree_pred = tree.predict_from_transposed(&transposed, sample_idx);
                let last_pred = *cumulative_preds[sample_idx].last().unwrap();
                cumulative_preds[sample_idx].push(last_pred + self.learning_rate * tree_pred);
            }
        }
        
        let mut predictions = Vec::with_capacity(n_samples);
        let mut uncertainties = Vec::with_capacity(n_samples);
        
        for sample_preds in cumulative_preds {
            let final_pred = *sample_preds.last().unwrap();
            predictions.push(final_pred);
            
            let mean = sample_preds.iter().sum::<f64>() / sample_preds.len() as f64;
            let variance = sample_preds.iter()
                .map(|&p| (p - mean).powi(2))
                .sum::<f64>() / sample_preds.len() as f64;
            uncertainties.push(variance.sqrt());
        }
        
        Ok((predictions, uncertainties))
    }
    
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

impl Default for PKBoostRegressor {
    fn default() -> Self {
        Self::new()
    }
}

pub fn calculate_rmse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let mse: f64 = y_true.iter().zip(y_pred.iter())
        .map(|(yt, yp)| (yt - yp).powi(2))
        .sum::<f64>() / y_true.len() as f64;
    mse.sqrt()
}

pub fn calculate_mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    y_true.iter().zip(y_pred.iter())
        .map(|(yt, yp)| (yt - yp).abs())
        .sum::<f64>() / y_true.len() as f64
}

pub fn calculate_r2(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let ss_tot: f64 = y_true.iter().map(|y| (y - mean).powi(2)).sum();
    let ss_res: f64 = y_true.iter().zip(y_pred.iter()).map(|(yt, yp)| (yt - yp).powi(2)).sum();
    1.0 - (ss_res / ss_tot)
}
