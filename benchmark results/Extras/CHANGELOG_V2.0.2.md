# PKBoost v2.0.2 Changelog

## Release Date: November 2025

## 🎯 Major Feature: Poisson Loss for Count Regression

### New Capabilities
- **Poisson Regression**: Full support for count-based targets (Y ∈ {0, 1, 2, ...})
- **Log-Link Function**: Automatic exp() transformation for non-negative predictions
- **Newton-Raphson Integration**: Seamless fit into existing optimization framework

### Performance
- **6.4% improvement** over MSE on synthetic Poisson data
- Optimized for insurance claims, purchase counts, event frequency modeling

### API
```rust
let mut model = PKBoostRegressor::auto(&x_train, &y_train)
    .with_loss(RegressionLossType::Poisson);
model.fit(&x_train, &y_train, None, true)?;
let predictions = model.predict(&x_test)?;
```

### Files Added
- `src/loss.rs` - Unified loss module with Poisson, MSE, Huber
- `src/bin/test_poisson.rs` - Benchmark test for Poisson regression
- `POISSON_LOSS.md` - Complete documentation and usage guide

### Technical Details
- Gradient: `exp(f) - y`
- Hessian: `exp(f)`
- Overflow prevention: Cap at 10^15
- Hessian stability: Min 1e-6

## 🔧 Improvements

### Loss Module Refactoring
- Consolidated loss functions into single module
- Added `OptimizedShannonLoss` for backward compatibility
- Unified gradient/hessian interface

### Regression Enhancements
- `RegressionLossType` enum now includes Poisson
- `.with_loss()` builder method for easy loss selection
- Automatic prediction transformation based on loss type

## 📊 Benchmark Results

**Synthetic Poisson Data** (5000 train, 1000 test):
```
True model: λ = exp(0.5 + 0.3·x₁ + 0.7·x₂)

MSE Loss:     RMSE 1.653, MAE 1.202
Poisson Loss: RMSE 1.548, MAE 1.143 (+6.4% improvement)
```

## 🐛 Bug Fixes
- None (new feature release)

## 📚 Documentation
- Added comprehensive Poisson loss guide
- Mathematical foundation and derivations
- Usage examples and best practices
- When to use Poisson vs MSE vs Huber

## 🔮 Future Roadmap
- Gamma Loss (continuous skewed data)
- Tweedie Loss (insurance pricing)
- Negative Binomial (overdispersed counts)

## Breaking Changes
- None (fully backward compatible)

## Migration Guide
No migration needed. Existing code continues to work. To use Poisson:
```rust
// Old (still works)
let model = PKBoostRegressor::auto(&x, &y);

// New (Poisson)
let model = PKBoostRegressor::auto(&x, &y)
    .with_loss(RegressionLossType::Poisson);
```

---

**Full Changelog**: v2.0.1...v2.0.2
