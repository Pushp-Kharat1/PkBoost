// Shannon entropy loss for gradient boosting
// Uses information theory to guide splits - works better for imbalanced data

use rayon::prelude::*;
use crate::adaptive_parallel::{adaptive_par_map, ParallelComplexity};

#[derive(Debug)]
pub struct OptimizedShannonLoss;

impl OptimizedShannonLoss {
    pub fn new() -> Self { Self }

    // convert raw predictions to probabilities
    pub fn sigmoid(&self, x: &[f64]) -> Vec<f64> {
        const PARALLEL_THRESHOLD: usize = 1000;
        
        if x.len() < PARALLEL_THRESHOLD {  // small arrays - just do it sequentially
            x.iter().map(|&val| {
                let clamped = val.clamp(-50.0, 50.0);
                1.0 / (1.0 + (-clamped).exp())
            }).collect()
        } else {
            adaptive_par_map(x, ParallelComplexity::Simple, |&val| {
                let clamped = val.clamp(-50.0, 50.0);
                1.0 / (1.0 + (-clamped).exp())
            })
        }
    }

// compute gradients for gradient boosting
// we upweight the minority class to handle imbalance
pub fn gradient(&self, y_true: &[f64], y_pred: &[f64], scale_pos_weight: f64) -> Vec<f64> {
    let p = self.sigmoid(y_pred);
    let pos_multiplier = scale_pos_weight.sqrt().min(8.0);  // sqrt to avoid explosion
    
    p.par_iter().zip(y_true.par_iter()).map(|(&p_val, &y_val)| {
        let grad = p_val - y_val;
        if y_val > 0.5 {  // positive class
            grad * pos_multiplier  // upweight gradient
        } else { 
            grad 
        }
    }).collect()
}

// second derivatives for newton boosting
pub fn hessian(&self, y_true: &[f64], y_pred: &[f64], scale_pos_weight: f64) -> Vec<f64> {
    let p = self.sigmoid(y_pred);
    let pos_multiplier = scale_pos_weight.sqrt().min(8.0);
    
    p.par_iter().zip(y_true.par_iter()).map(|(&p_val, &y_val)| {
        let base_hess = (p_val * (1.0 - p_val)).max(1e-7);  // clamp to avoid division by zero
        if y_val > 0.5 { 
            base_hess * pos_multiplier  // same upweighting as gradient
        } else { 
            base_hess 
        }
    }).collect()
}

    pub fn init_score(&self, _y_true: &[f64]) -> f64 {
        0.0
    }
}
