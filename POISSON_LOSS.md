# Poisson Loss for Count-Based Regression

## Overview

PKBoost v2.0.2+ now supports **Poisson regression** for modeling count data (non-negative integers). This is ideal for:
- Insurance claims frequency
- Customer purchase counts
- Event occurrence modeling
- Any scenario where Y ∈ {0, 1, 2, 3, ...}

## Mathematical Foundation

### Poisson Distribution
Assumes target follows: `Y ~ Poisson(λ)` where `λ = E[Y] = exp(f(x))`

### Loss Function (Negative Log-Likelihood)
```
L(y, f) = exp(f) - y·f + log(y!)
```

### Gradients for Newton-Raphson
```
Gradient:  ∂L/∂f = exp(f) - y
Hessian:   ∂²L/∂f² = exp(f)
```

### Log-Link Function
Predictions use `λ = exp(f)` to ensure non-negativity.

## Usage

### Basic Example
```rust
use pkboost::*;

// Generate or load count data
let (x_train, y_train) = load_count_data();
let (x_test, y_test) = load_test_data();

// Create Poisson regressor
let mut model = PKBoostRegressor::auto(&x_train, &y_train)
    .with_loss(RegressionLossType::Poisson);

// Train
model.fit(&x_train, &y_train, None, true)?;

// Predict (automatically applies exp transformation)
let predictions = model.predict(&x_test)?;

// Evaluate
let rmse = calculate_rmse(&y_test, &predictions);
let mae = calculate_mae(&y_test, &predictions);
```

### With Validation Set
```rust
let mut model = PKBoostRegressor::auto(&x_train, &y_train)
    .with_loss(RegressionLossType::Poisson);

model.n_estimators = 500;
model.learning_rate = 0.03;

model.fit(
    &x_train, 
    &y_train, 
    Some((&x_val, &y_val)),  // Early stopping on validation
    true
)?;
```

### Manual Configuration
```rust
let mut model = PKBoostRegressor::new();
model.loss_type = RegressionLossType::Poisson;
model.n_estimators = 300;
model.learning_rate = 0.05;
model.max_depth = 5;
model.reg_lambda = 2.0;

model.fit(&x_train, &y_train, None, false)?;
```

## Benchmark Results

**Synthetic Poisson Data** (5000 train, 1000 test)
- True model: `λ = exp(0.5 + 0.3·x₁ + 0.7·x₂)`

| Loss Type | RMSE  | MAE   | Improvement |
|-----------|-------|-------|-------------|
| MSE       | 1.653 | 1.202 | Baseline    |
| **Poisson**   | **1.548** | **1.143** | **+6.4%**   |

## When to Use Poisson Loss

### ✅ Good Fit
- **Count data**: Y ∈ {0, 1, 2, 3, ...}
- **Event frequency**: Number of claims, purchases, visits
- **Rare events**: Low mean counts (λ < 10)
- **Overdispersion acceptable**: Variance ≈ mean

### ❌ Not Suitable
- **Continuous targets**: Use MSE or Huber
- **Negative values**: Poisson requires Y ≥ 0
- **Severe overdispersion**: Consider Negative Binomial
- **Zero-inflated data**: May need specialized models

## Comparison with Other Losses

| Loss Type | Best For | Target Range | Robustness |
|-----------|----------|--------------|------------|
| **MSE** | Continuous regression | (-∞, +∞) | Low (sensitive to outliers) |
| **Huber** | Robust regression | (-∞, +∞) | High (outlier resistant) |
| **Poisson** | Count data | [0, ∞) integers | Medium (assumes Poisson distribution) |

## Implementation Details

### Overflow Prevention
```rust
let exp_f = f.exp().min(1e15);  // Cap at 10^15
```

### Hessian Stability
```rust
hess.push(exp_f.max(1e-6));  // Prevent zero hessian
```

### Prediction Transformation
```rust
// Raw predictions are log(λ)
let raw_preds = model.predict_raw(&x_test)?;

// Automatically transformed to λ = exp(f)
let predictions = model.predict(&x_test)?;
```

## Advanced: Custom Loss Functions

To add new loss functions (e.g., Gamma, Tweedie):

1. **Add to loss.rs**:
```rust
pub struct GammaLoss;

impl GammaLoss {
    pub fn gradient_hessian(y_true: &[f64], y_pred: &[f64]) -> (Vec<f64>, Vec<f64>) {
        // Implement gradient and hessian
    }
}
```

2. **Add to RegressionLossType enum**:
```rust
pub enum RegressionLossType {
    MSE,
    Huber { delta: f64 },
    Poisson,
    Gamma,  // New
}
```

3. **Integrate in regression.rs**:
```rust
match self.loss_type {
    RegressionLossType::Gamma => GammaLoss::gradient_hessian(y_true, y_pred),
    // ...
}
```

## Testing

Run the Poisson regression test:
```bash
cargo run --release --bin test_poisson
```

Expected output:
```
=== Poisson Regression Test ===
Train: 5000 samples, Y mean: 2.41
Test: 1000 samples

MSE Model - RMSE: 1.653, MAE: 1.202
Poisson Model - RMSE: 1.548, MAE: 1.143

RMSE Improvement: +6.4%
✅ Poisson loss performs better for count data!
```

## References

- **Poisson Regression**: McCullagh & Nelder (1989), "Generalized Linear Models"
- **Gradient Boosting**: Friedman (2001), "Greedy Function Approximation"
- **Newton-Raphson**: Used in XGBoost, LightGBM for second-order optimization

## Future Extensions

Planned loss functions:
- **Gamma Loss**: For continuous skewed data (e.g., claim severity)
- **Tweedie Loss**: Generalizes Poisson + Gamma (insurance pricing)
- **Negative Binomial**: For overdispersed count data

---

**Version**: PKBoost v2.0.2  
**Author**: Pushp Kharat  
**Date**: November 2025
