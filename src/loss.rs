/// Loss functions for PKBoost
/// Provides gradient and hessian calculations for Newton-Raphson optimization
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Parallel threshold - avoid Rayon overhead for small arrays
const PARALLEL_THRESHOLD: usize = 10000;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizedShannonLoss;

/// Compute focal loss gradient and hessian for a single sample.
///
/// Focal loss: FL = -[(1-pt)^γ · log(pt)]  where pt = p if y=1, (1-p) if y=0.
/// At focal_gamma=0 this reduces exactly to standard weighted log-loss.
///
/// Gradient derivation (w.r.t. raw log-odds f, p = sigmoid(f)):
///   y=1: g = (1-p)^γ · (γ·p·ln(p) − (1-p))
///   y=0: g = p^γ    · (p − γ·(1-p)·ln(1-p))
///
/// Hessian: approximate as focal_weight · p · (1-p).  The exact hessian has
/// log terms that are numerically unstable at extremes; this approximation is
/// standard practice and exact at γ=0.
#[inline(always)]
fn focal_grad_hess(y: f64, p_raw: f64, weight: f64, focal_gamma: f64) -> (f64, f64) {
    let p = p_raw.clamp(1e-7, 1.0 - 1e-7);
    let q = 1.0 - p;

    if focal_gamma == 0.0 {
        // Fast path: standard weighted log-loss
        return (weight * (p - y), weight * p * q.max(1e-6));
    }

    if y > 0.5 {
        // Positive class: focal weight = (1-p)^γ
        let fw = q.powf(focal_gamma);
        // Exact gradient: (1-p)^γ · (γ·p·ln(p) − (1-p))
        let grad = weight * fw * (focal_gamma * p * p.ln() - q);
        let hess = (weight * fw * p * q).max(1e-6);
        (grad, hess)
    } else {
        // Negative class: focal weight = p^γ
        let fw = p.powf(focal_gamma);
        // Exact gradient: p^γ · (p − γ·(1-p)·ln(1-p))
        let grad = weight * fw * (p - focal_gamma * q * q.ln());
        let hess = (weight * fw * p * q).max(1e-6);
        (grad, hess)
    }
}

impl OptimizedShannonLoss {
    pub fn new() -> Self {
        Self
    }

    pub fn init_score(&self, y: &[f64]) -> f64 {
        let pos = y.iter().filter(|&&v| v > 0.5).count() as f64;
        let neg = y.len() as f64 - pos;
        (pos / neg.max(1.0)).ln()
    }

    /// Fused gradient and hessian calculation.
    /// focal_gamma=0.0 is the standard weighted log-loss (no overhead).
    #[inline]
    pub fn gradient_hessian(
        &self,
        y_true: &[f64],
        y_pred: &[f64],
        scale_pos_weight: f64,
        focal_gamma: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = y_true.len();

        if n >= PARALLEL_THRESHOLD {
            let results: Vec<(f64, f64)> = y_pred
                .par_iter()
                .zip(y_true.par_iter())
                .map(|(&pred, &true_y)| {
                    let prob = 1.0 / (1.0 + (-pred).exp());
                    let weight = if true_y > 0.5 { scale_pos_weight } else { 1.0 };
                    focal_grad_hess(true_y, prob, weight, focal_gamma)
                })
                .collect();

            let mut grad = Vec::with_capacity(n);
            let mut hess = Vec::with_capacity(n);
            for (g, h) in results {
                grad.push(g);
                hess.push(h);
            }
            (grad, hess)
        } else {
            let mut grad = Vec::with_capacity(n);
            let mut hess = Vec::with_capacity(n);
            for (&pred, &true_y) in y_pred.iter().zip(y_true.iter()) {
                let prob = 1.0 / (1.0 + (-pred).exp());
                let weight = if true_y > 0.5 { scale_pos_weight } else { 1.0 };
                let (g, h) = focal_grad_hess(true_y, prob, weight, focal_gamma);
                grad.push(g);
                hess.push(h);
            }
            (grad, hess)
        }
    }

    /// Legacy: separate gradient (for backward compatibility)
    pub fn gradient(&self, y_true: &[f64], y_pred: &[f64], scale_pos_weight: f64) -> Vec<f64> {
        self.gradient_hessian(y_true, y_pred, scale_pos_weight, 0.0).0
    }

    /// Legacy: separate hessian (for backward compatibility)
    pub fn hessian(&self, y_true: &[f64], y_pred: &[f64], scale_pos_weight: f64) -> Vec<f64> {
        self.gradient_hessian(y_true, y_pred, scale_pos_weight, 0.0).1
    }

    pub fn sigmoid(&self, preds: &[f64]) -> Vec<f64> {
        if preds.len() >= PARALLEL_THRESHOLD {
            preds
                .par_iter()
                .map(|&p| 1.0 / (1.0 + (-p).exp()))
                .collect()
        } else {
            preds.iter().map(|&p| 1.0 / (1.0 + (-p).exp())).collect()
        }
    }
}

pub struct PoissonLoss;

impl PoissonLoss {
    /// Compute gradient and hessian for Poisson regression
    /// Gradient: exp(f) - y
    /// Hessian: exp(f)
    pub fn gradient_hessian(y_true: &[f64], y_pred: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut grad = Vec::with_capacity(y_true.len());
        let mut hess = Vec::with_capacity(y_true.len());

        for (&y, &f) in y_true.iter().zip(y_pred.iter()) {
            let exp_f = f.exp().min(1e15); // Prevent overflow
            grad.push(exp_f - y);
            hess.push(exp_f.max(1e-6)); // Prevent zero hessian
        }
        (grad, hess)
    }

    /// Compute Poisson deviance loss
    pub fn loss(y_true: &[f64], y_pred: &[f64]) -> f64 {
        y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y, &f)| {
                let exp_f = f.exp().min(1e15);
                exp_f - y * f
            })
            .sum::<f64>()
            / y_true.len() as f64
    }
}

pub struct MSELoss;

impl MSELoss {
    pub fn gradient_hessian(y_true: &[f64], y_pred: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let grad: Vec<f64> = y_pred
            .iter()
            .zip(y_true.iter())
            .map(|(&pred, &true_y)| pred - true_y)
            .collect();
        let hess = vec![1.0; y_true.len()];
        (grad, hess)
    }

    pub fn loss(y_true: &[f64], y_pred: &[f64]) -> f64 {
        y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(&y, &pred)| (y - pred).powi(2))
            .sum::<f64>()
            / y_true.len() as f64
    }
}

pub struct HuberLoss {
    pub delta: f64,
}

impl HuberLoss {
    pub fn new(delta: f64) -> Self {
        Self { delta }
    }

    pub fn gradient_hessian(&self, y_true: &[f64], y_pred: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut grad = Vec::with_capacity(y_true.len());
        let mut hess = Vec::with_capacity(y_true.len());

        for (&y, &pred) in y_true.iter().zip(y_pred.iter()) {
            let residual = pred - y;
            if residual.abs() <= self.delta {
                grad.push(residual);
                hess.push(1.0);
            } else {
                grad.push(self.delta * residual.signum());
                hess.push(1e-6); // Small constant for outliers
            }
        }
        (grad, hess)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum LossType {
    MSE,
    Huber,
    Poisson,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_gradient() {
        let y_true = vec![0.0, 1.0, 3.0, 2.0];
        let y_pred = vec![0.1, 0.5, 1.2, 0.8];
        let (grad, hess) = PoissonLoss::gradient_hessian(&y_true, &y_pred);

        // exp(0.1) - 0 ≈ 1.105
        assert!((grad[0] - 1.105).abs() < 0.01);
        // exp(0.5) - 1 ≈ 0.649
        assert!((grad[1] - 0.649).abs() < 0.01);

        // Hessian should be exp(f)
        assert!((hess[0] - 0.1_f64.exp()).abs() < 0.01);
    }

    #[test]
    fn test_poisson_loss() {
        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![0.0, 0.69, 1.10]; // log(1), log(2), log(3)
        let loss = PoissonLoss::loss(&y_true, &y_pred);
        assert!(loss < 0.5); // Should be small for good predictions
    }
}
