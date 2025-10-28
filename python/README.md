# PKBoost Python Bindings

Python interface for PKBoost - gradient boosting with adaptive drift detection for imbalanced data.

## Installation

```bash
# From local wheel (development)
pip install target/wheels/pkboost-0.1.0-cp313-cp313-win_amd64.whl --force-reinstall

# From PyPI 
pip install pkboost
```

## Quick Start

### Basic Usage (Static Model)

```python
import numpy as np
from pkboost import PKBoostClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

# Generate imbalanced dataset
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    weights=[0.98, 0.02],  # 2% minority class
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to contiguous arrays (required)
X_train = np.ascontiguousarray(X_train, dtype=np.float64)
y_train = np.ascontiguousarray(y_train, dtype=np.float64)
X_test = np.ascontiguousarray(X_test, dtype=np.float64)
y_test = np.ascontiguousarray(y_test, dtype=np.float64)

# Auto-tuned model (recommended)
model = PKBoostClassifier.auto()
model.fit(X_train, y_train, x_val=X_test[:500], y_val=y_test[:500], verbose=True)

# Predict
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test, threshold=0.5)

# Evaluate
pr_auc = average_precision_score(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"PR-AUC: {pr_auc:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Feature importance
importance = model.get_feature_importance()
print(f"Top features: {importance.argsort()[-5:][::-1]}")
```

### Adaptive Model with Drift Detection

```python
from pkboost import PKBoostAdaptive

# Initialize adaptive model
model = PKBoostAdaptive()

# Initial training
model.fit_initial(X_train, y_train, x_val=X_test[:500], y_val=y_test[:500], verbose=True)

# Baseline evaluation
y_pred = model.predict_proba(X_test)
baseline_pr_auc = average_precision_score(y_test, y_pred)

print(f"Baseline PR-AUC: {baseline_pr_auc:.4f}")
print(f"State: {model.get_state()}")
print(f"Vulnerability: {model.get_vulnerability_score():.4f}")

# Simulate streaming data
for batch_idx in range(10):
    # Get new batch of data
    X_batch, y_batch = get_streaming_batch()  # Your data source
    
    X_batch = np.ascontiguousarray(X_batch, dtype=np.float64)
    y_batch = np.ascontiguousarray(y_batch, dtype=np.float64)
    
    # Observe batch (triggers drift detection & adaptation)
    model.observe_batch(X_batch, y_batch, verbose=True)
    
    # Check adaptation status
    print(f"State: {model.get_state()}")
    print(f"Vulnerability: {model.get_vulnerability_score():.4f}")
    print(f"Metamorphoses: {model.get_metamorphosis_count()}")
    
    # Evaluate current performance
    y_pred = model.predict_proba(X_test)
    current_pr_auc = average_precision_score(y_test, y_pred)
    degradation = (baseline_pr_auc - current_pr_auc) / baseline_pr_auc * 100
    print(f"Performance degradation: {degradation:.1f}%")
```

## API Reference

### PKBoostClassifier

Standard gradient boosting model for static datasets.

**Methods:**

- `PKBoostClassifier.auto()` - Create auto-tuned model
- `PKBoostClassifier(n_estimators=1000, learning_rate=0.05, ...)` - Manual configuration
- `fit(X, y, x_val=None, y_val=None, verbose=False)` - Train model
- `predict_proba(X)` - Predict probabilities
- `predict(X, threshold=0.5)` - Predict classes
- `get_feature_importance()` - Get feature importance scores
- `get_n_trees()` - Get number of trees in ensemble
- `is_fitted` - Check if model is trained

**Parameters:**

- `n_estimators` (int): Number of boosting rounds (default: 1000)
- `learning_rate` (float): Learning rate (default: 0.05)
- `max_depth` (int): Maximum tree depth (default: 6)
- `min_samples_split` (int): Minimum samples to split (default: 20)
- `min_child_weight` (float): Minimum sum of instance weight in child (default: 1.0)
- `reg_lambda` (float): L2 regularization (default: 1.0)
- `gamma` (float): Minimum loss reduction for split (default: 0.0)
- `subsample` (float): Row sampling ratio (default: 0.8)
- `colsample_bytree` (float): Column sampling ratio (default: 0.8)
- `scale_pos_weight` (float): Weight for positive class (default: 1.0)

### PKBoostAdaptive

Adaptive model with real-time drift detection and metamorphosis.

**Methods:**

- `PKBoostAdaptive()` - Create adaptive model
- `fit_initial(X, y, x_val=None, y_val=None, verbose=False)` - Initial training
- `observe_batch(X, y, verbose=False)` - Process streaming batch
- `predict_proba(X)` - Predict probabilities
- `predict(X, threshold=0.5)` - Predict classes
- `get_vulnerability_score()` - Get current vulnerability score
- `get_state()` - Get system state ("Normal", "Alert(n)", "Metamorphosis")
- `get_metamorphosis_count()` - Get number of adaptations triggered
- `is_fitted` - Check if model is trained

**States:**

- `Normal` - Model performing well
- `Alert(n)` - Performance degrading (n consecutive checks)
- `Metamorphosis` - Actively adapting to drift

**Drift Detection:**

The model automatically:
1. Monitors vulnerability scores on streaming data
2. Detects performance degradation
3. Triggers metamorphosis when thresholds exceeded
4. Prunes outdated trees and adds new ones
5. Validates adaptation quality (rollback if degraded)

## Data Requirements

- **Format**: NumPy arrays (2D for features, 1D for labels)
- **Type**: `float64` (use `np.ascontiguousarray(X, dtype=np.float64)`)
- **Labels**: Binary (0.0 or 1.0)
- **Missing values**: Supported (median imputation)
- **Categorical features**: Not supported (encode first)

## Performance Tips

1. **Use auto-tuning**: `PKBoostClassifier.auto()` optimizes for your data
2. **Provide validation set**: Enables early stopping and better tuning
3. **Contiguous arrays**: Always use `np.ascontiguousarray()` for speed
4. **Batch size**: For adaptive model, use batches of 500-2000 samples
5. **Verbose mode**: Set `verbose=True` to monitor training progress

## Examples

See:
- `example.py` - Basic usage with static model
- `example_drift.py` - Adaptive model with drift simulation

## Comparison with XGBoost/LightGBM

**When to use PKBoost:**
- Extreme class imbalance (<5% minority)
- Streaming data with concept drift
- Need automatic hyperparameter tuning
- Prioritize PR-AUC over speed

**When to use XGBoost/LightGBM:**
- Balanced datasets
- Static data (no drift)
- Need fastest training speed
- Multi-class or regression tasks

## Benchmarks

| Dataset | Imbalance | PKBoost PR-AUC | XGBoost PR-AUC | Improvement |
|---------|-----------|----------------|----------------|-------------|
| Credit Card | 0.2% | 87.8% | 74.5% | +17.9% |
| Pima Diabetes | 35% | 98.0% | 68.0% | +44.0% |
| Ionosphere | 36% | 98.0% | 97.2% | +0.8% |

**Drift Resilience (Credit Card):**
- PKBoost: 1.8% degradation under drift
- XGBoost: 31.8% degradation
- LightGBM: 42.5% degradation

## Troubleshooting

**"TypeError: 'ndarray' object cannot be converted"**
```python
# Fix: Use contiguous arrays
X = np.ascontiguousarray(X, dtype=np.float64)
y = np.ascontiguousarray(y, dtype=np.float64)
```

**"Model not fitted"**
```python
# Fix: Call fit() or fit_initial() first
model.fit(X_train, y_train)
```

**Slow training**
```python
# Use smaller validation set for faster early stopping
model.fit(X_train, y_train, x_val=X_val[:1000], y_val=y_val[:1000])
```

## License

MIT License - Free for commercial use

## Citation

```bibtex
@software{kharat2025pkboost,
  author = {Kharat, Pushp},
  title = {PKBoost: Shannon-Guided Gradient Boosting for Extreme Imbalance},
  year = {2025},
  url = {https://github.com/Pushp-Kharat1/pkboost}
}
```

## PkBoost has several meanings:
1) Performance-based Knowledge Boosting 
2) Pushp Kharat's booster
3) Pain-in-the-ass Kafkaesque Booster (for those moments when debugging feels like a bureaucratic fever dream, google the meaning, GitHub will ban me if i say it here).
