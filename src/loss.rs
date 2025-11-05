/// Loss functions for PKBoost
/// Provides gradient and hessian calculations for Newton-Raphson optimization

use rayon::prelude::*;

#[derive(Debug)]
pub struct OptimizedShannonLoss;

impl OptimizedShannonLoss {
    pub fn new() -> Self { Self }
    
    pub fn init_score(&self, y: &[f64]) -> f64 {
        let pos = y.iter().filter(|&&v| v > 0.5).count() as f64;
        let neg = y.len() as f64 - pos;
        (pos / neg.max(1.0)).ln()
    }
    
    pub fn gradient(&self, y_true: &[f64], y_pred: &[f64], scale_pos_weight: f64) -> Vec<f64> {
        y_pred.par_iter().zip(y_true.par_iter())
            .map(|(&pred, &true_y)| {
                let prob = 1.0 / (1.0 + (-pred).exp());
                let weight = if true_y > 0.5 { scale_pos_weight } else { 1.0 };
                weight * (prob - true_y)
            })
            .collect()
    }
    
    pub fn hessian(&self, y_true: &[f64], y_pred: &[f64], scale_pos_weight: f64) -> Vec<f64> {
        y_pred.par_iter().zip(y_true.par_iter())
            .map(|(&pred, &true_y)| {
                let prob = 1.0 / (1.0 + (-pred).exp());
                let weight = if true_y > 0.5 { scale_pos_weight } else { 1.0 };
                weight * prob * (1.0 - prob).max(1e-6)
            })
            .collect()
    }
    
    pub fn sigmoid(&self, preds: &[f64]) -> Vec<f64> {
        preds.par_iter().map(|&p| 1.0 / (1.0 + (-p).exp())).collect()
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
        y_true.iter().zip(y_pred.iter())
            .map(|(&y, &f)| {
                let exp_f = f.exp().min(1e15);
                exp_f - y * f
            })
            .sum::<f64>() / y_true.len() as f64
    }
}

pub struct MSELoss;

impl MSELoss {
    pub fn gradient_hessian(y_true: &[f64], y_pred: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let grad: Vec<f64> = y_pred.iter().zip(y_true.iter())
            .map(|(&pred, &true_y)| pred - true_y)
            .collect();
        let hess = vec![1.0; y_true.len()];
        (grad, hess)
    }

    pub fn loss(y_true: &[f64], y_pred: &[f64]) -> f64 {
        y_true.iter().zip(y_pred.iter())
            .map(|(&y, &pred)| (y - pred).powi(2))
            .sum::<f64>() / y_true.len() as f64
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
