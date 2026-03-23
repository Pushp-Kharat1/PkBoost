// Huber loss for robust regression (less sensitive to outliers than MSE)

use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct HuberLoss {
    pub delta: f64,  // Threshold for switching from squared to linear
}

impl HuberLoss {
    pub fn new(delta: f64) -> Self {
        Self { delta: delta.max(0.1) }
    }
    
    pub fn auto(y: &[f64]) -> Self {
        // Auto-set delta to 1.35 * MAD (median absolute deviation)
        let mut abs_devs: Vec<f64> = y.iter().map(|&v| v.abs()).collect();
        abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = abs_devs[abs_devs.len() / 2];
        Self::new(1.35 * mad)
    }
    
    pub fn gradient(&self, y_true: &[f64], y_pred: &[f64]) -> Vec<f64> {
        y_pred.par_iter().zip(y_true.par_iter())
            .map(|(&pred, &true_y)| {
                let error = pred - true_y;
                if error.abs() <= self.delta {
                    error  // Quadratic region (like MSE)
                } else {
                    self.delta * error.signum()  // Linear region (robust to outliers)
                }
            })
            .collect()
    }
    
    pub fn hessian(&self, y_true: &[f64], y_pred: &[f64]) -> Vec<f64> {
        y_pred.par_iter().zip(y_true.par_iter())
            .map(|(&pred, &true_y)| {
                let error = (pred - true_y).abs();
                if error <= self.delta { 1.0 } else { 0.0 }
            })
            .collect()
    }
    
    // For compatibility with existing code
    pub fn hessian_const(&self, y_true: &[f64]) -> Vec<f64> {
        vec![1.0; y_true.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_huber_gradient() {
        let loss = HuberLoss::new(1.0);
        let y_true = vec![0.0, 0.0, 0.0];
        let y_pred = vec![0.5, 1.5, 3.0];  // Small, medium, large error
        
        let grad = loss.gradient(&y_true, &y_pred);
        
        assert!((grad[0] - 0.5).abs() < 1e-6);  // Quadratic: error
        assert!((grad[1] - 1.0).abs() < 1e-6);  // Linear: delta * sign
        assert!((grad[2] - 1.0).abs() < 1e-6);  // Linear: delta * sign
    }
}
