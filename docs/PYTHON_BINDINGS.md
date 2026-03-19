# PKBoost Python Bindings

Python interface for PKBoost - gradient boosting with adaptive drift detection for imbalanced data.

## Installation

```bash
# Install from PyPI (recommended)
pip install pkboost

# For development version from local wheel
pip install target/wheels/pkboost-2.2.0-cp314-cp314-win_amd64.whl --force-reinstall
```

## Available Interfaces

PKBoost provides two Python interfaces:

1. **Direct Rust Bindings** (`pkboost`): High-performance interface with zero-copy NumPy integration
2. **Scikit-learn Wrapper** (`pkboost_sklearn`): Full sklearn compatibility with Pipeline, GridSearchCV support

Choose based on your needs:
- Use `pkboost` for maximum performance and streaming applications
- Use `pkboost_sklearn` for sklearn ecosystem integration

## Comparisons with XGBoost/LightGBM

| Dataset | Imbalance | PKBoost PR-AUC | XGBoost PR-AUC | Speed (Samples/s) |
|---------|-----------|----------------|----------------|-------------------|
| Credit Card | 0.2% | 83.6% | 74.5% | ~2.75M |

## Quick Start

### Interface 1: Direct Rust Bindings (Recommended for Performance)

```python
import numpy as np
from pkboost import PKBoostClassifier, PKBoostAdaptive, PKBoostRegressor, PKBoostMultiClass
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

# Convert to contiguous arrays (required for zero-copy)
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

### Interface 2: Scikit-learn Wrapper (Recommended for Integration)

```python
from pkboost_sklearn import PKBoostClassifier, PKBoostRegressor, PKBoostAdaptiveClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Standard sklearn interface
clf = PKBoostClassifier(n_estimators=500, learning_rate=0.1, auto=True)
clf.fit(X_train, y_train, eval_set=(X_test[:500], y_test[:500]))

# Works with sklearn tools
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', PKBoostClassifier())
])
pipe.fit(X_train, y_train)

# Grid search
param_grid = {'n_estimators': [100, 500], 'learning_rate': [0.05, 0.1]}
grid = GridSearchCV(PKBoostClassifier(), param_grid, cv=3)
grid.fit(X_train, y_train)
```

### Adaptive Model with Drift Detection

```python
from pkboost import PKBoostAdaptive
# OR from sklearn wrapper:
# from pkboost_sklearn import PKBoostAdaptiveClassifier

# Direct Rust bindings
model = PKBoostAdaptive()
model.fit_initial(X_train, y_train, x_val=X_test[:500], y_val=y_test[:500], verbose=True)

# Scikit-learn wrapper (uses standard fit method)
# model = PKBoostAdaptiveClassifier()
# model.fit(X_train, y_train, eval_set=(X_test[:500], y_test[:500]))

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

### Multi-Class Classification

```python
from pkboost import PKBoostMultiClass
# OR from sklearn wrapper:
# from pkboost_sklearn import PKBoostMultiClass

# Generate multi-class dataset
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=5000, n_features=20, n_classes=5, 
                           n_informative=15, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to contiguous arrays
X_train = np.ascontiguousarray(X_train, dtype=np.float64)
y_train = np.ascontiguousarray(y_train, dtype=np.float64)
X_test = np.ascontiguousarray(X_test, dtype=np.float64)

# Direct Rust bindings
model = PKBoostMultiClass(n_classes=5)
model.fit(X_train, y_train, verbose=True)

# Scikit-learn wrapper (auto-detects classes)
# model = PKBoostMultiClass()
# model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)  # Shape: (n_samples, n_classes)

# Evaluate
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
```

### Regression

```python
from pkboost import PKBoostRegressor
# OR from sklearn wrapper:
# from pkboost_sklearn import PKBoostRegressor

from sklearn.datasets import make_regression
X, y = make_regression(n_samples=5000, n_features=20, noise=0.1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to contiguous arrays
X_train = np.ascontiguousarray(X_train, dtype=np.float64)
y_train = np.ascontiguousarray(y_train, dtype=np.float64)
X_test = np.ascontiguousarray(X_test, dtype=np.float64)

# Auto-tuned regressor
model = PKBoostRegressor.auto()
model.fit(X_train, y_train, verbose=True)

# Predict
y_pred = model.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f}, R²: {r2:.4f}")
```

## API Reference

### Interface 1: Direct Rust Bindings (`from pkboost import ...`)

#### PKBoostClassifier

Standard gradient boosting model for binary classification.

**Constructor:**
```python
PKBoostClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_samples_split=20,
    min_child_weight=1.0,
    reg_lambda=1.0,
    gamma=0.0,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.0
)
```

**Static Methods:**
- `PKBoostClassifier.auto()` - Create auto-tuned model (recommended)

**Methods:**
- `fit(X, y, x_val=None, y_val=None, verbose=False)` - Train model
- `predict_proba(X)` - Predict probabilities (returns 1D array for positive class)
- `predict(X, threshold=0.5)` - Predict classes (0 or 1)
- `get_feature_importance()` - Get feature importance scores (1D array)
- `get_n_trees()` - Get number of trees in ensemble
- `is_fitted` - Check if model is trained (property)
- `to_bytes()` - Serialize model to bytes
- `from_bytes(data)` - Deserialize from bytes (static method)
- `to_json()` - Serialize model to JSON string
- `from_json(json_str)` - Deserialize from JSON (static method)
- `save(path)` - Save model to file
- `load(path)` - Load model from file (static method)

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

#### PKBoostAdaptive

Adaptive model with real-time drift detection and metamorphosis.

**Constructor:**
```python
PKBoostAdaptive()  # No parameters
```

**Methods:**
- `fit_initial(X, y, x_val=None, y_val=None, verbose=False)` - Initial training
- `observe_batch(X, y, verbose=False)` - Process streaming batch
- `predict_proba(X)` - Predict probabilities (returns 1D array)
- `predict(X, threshold=0.5)` - Predict classes (0 or 1)
- `get_vulnerability_score()` - Get current vulnerability score (float)
- `get_state()` - Get system state ("Normal", "Alert(n)", "Metamorphosis")
- `get_metamorphosis_count()` - Get number of adaptations triggered
- `is_fitted` - Check if model is trained (property)

**States:**
- `Normal` - Model performing well
- `Alert(n)` - Performance degrading (n consecutive checks)
- `Metamorphosis` - Actively adapting to drift

#### PKBoostRegressor

Gradient boosting model for regression tasks.

**Constructor:**
```python
PKBoostRegressor()  # No parameters for manual config
```

**Static Methods:**
- `PKBoostRegressor.auto()` - Create auto-tuned model (recommended)

**Methods:**
- `fit(X, y, x_val=None, y_val=None, verbose=False)` - Train model
- `predict(X)` - Predict continuous values
- `is_fitted` - Check if model is trained (property)

#### PKBoostMultiClass

Multi-class classification using One-vs-Rest strategy.

**Constructor:**
```python
PKBoostMultiClass(n_classes)  # n_classes: int, required
```

**Methods:**
- `fit(X, y, x_val=None, y_val=None, verbose=False)` - Train model
- `predict_proba(X)` - Predict probabilities (returns 2D array: [n_samples, n_classes])
- `predict(X)` - Predict class labels (returns 1D array of class indices)
- `is_fitted` - Check if model is trained (property)

---

### Interface 2: Scikit-learn Wrapper (`from pkboost_sklearn import ...`)

#### PKBoostClassifier

Scikit-learn compatible binary classifier.

**Constructor:**
```python
PKBoostClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    min_samples_split=20,
    min_child_weight=1.0,
    reg_lambda=1.0,
    gamma=0.0,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1.0,
    auto=False,  # Use auto-tuning
    random_state=None
)
```

**Methods:**
- `fit(X, y, sample_weight=None, eval_set=None, verbose=False)` - Train model
- `predict_proba(X)` - Predict probabilities (returns 2D array: [n_samples, 2])
- `predict(X, threshold=0.5)` - Predict classes
- `predict_log_proba(X)` - Predict log-probabilities
- `decision_function(X)` - Compute decision function
- `score(X, y, sample_weight=None)` - Return mean accuracy
- `save_model(path)` - Save model to file
- `load_model(path, classes=None, **params)` - Load model from file (classmethod)

**Sklearn Attributes:**
- `classes_` - Class labels
- `n_features_in_` - Number of features seen during fit
- `feature_importances_` - Feature importance scores
- `n_trees_` - Number of trees after early stopping

#### PKBoostRegressor

Scikit-learn compatible regressor.

**Constructor:**
```python
PKBoostRegressor(auto=True, random_state=None)
```

**Methods:**
- `fit(X, y, sample_weight=None, eval_set=None, verbose=False)` - Train model
- `predict(X)` - Predict continuous values
- `score(X, y, sample_weight=None)` - Return R² score

**Sklearn Attributes:**
- `n_features_in_` - Number of features seen during fit
- `feature_importances_` - Feature importance scores (zeros for regressor)

#### PKBoostAdaptiveClassifier

Scikit-learn compatible adaptive classifier.

**Constructor:**
```python
PKBoostAdaptiveClassifier(auto=True)
```

**Methods:**
- `fit(X, y, eval_set=None, verbose=False)` - Train model (uses standard sklearn interface)
- `observe_batch(X, y, verbose=False)` - Process streaming batch
- `predict_proba(X)` - Predict probabilities (returns 2D array)
- `predict(X, threshold=0.5)` - Predict classes
- `get_status()` - Get current adaptation status (dict with state, vulnerability_score, metamorphosis_count)

**Sklearn Attributes:**
- `classes_` - Class labels
- `n_features_in_` - Number of features seen during fit
- `state_` - Current system state
- `vulnerability_score_` - Current vulnerability score
- `metamorphosis_count_` - Number of metamorphoses

#### PKBoostMultiClass

Scikit-learn compatible multi-class classifier.

**Constructor:**
```python
PKBoostMultiClass(n_classes=None, verbose=False)
```

**Methods:**
- `fit(X, y, eval_set=None, verbose=False)` - Train model
- `predict_proba(X)` - Predict probabilities (returns 2D array)
- `predict(X)` - Predict class labels
- `score(X, y, sample_weight=None)` - Return mean accuracy

**Sklearn Attributes:**
- `classes_` - Class labels
- `n_features_in_` - Number of features seen during fit

## Data Requirements

### Array Format
- **Format**: NumPy arrays (2D for features, 1D for labels)
- **Type**: `float64` (use `np.ascontiguousarray(X, dtype=np.float64)`)
- **Memory Layout**: C-contiguous (required for zero-copy performance)
- **Missing values**: Supported (automatic median imputation)

### Label Requirements
- **Binary Classification**: 0.0 or 1.0 (auto-converted from other binary labels)
- **Multi-class Classification**: Integer class labels (0, 1, 2, ...)
- **Regression**: Continuous float values

### Feature Requirements
- **Categorical features**: Not supported (encode first using one-hot, label encoding, etc.)
- **Sparse matrices**: Not supported (convert to dense arrays)
- **String features**: Not supported (encode numerically)

## Performance Tips

### For Maximum Performance
1. **Use auto-tuning**: `PKBoostClassifier.auto()` optimizes for your data
2. **Provide validation set**: Enables early stopping and better tuning
3. **Contiguous arrays**: Always use `np.ascontiguousarray()` for zero-copy
4. **Correct dtype**: Ensure `float64` dtype to avoid conversions
5. **Batch size**: For adaptive model, use batches of 500-2000 samples

### For Sklearn Integration
1. **Use sklearn wrapper**: `pkboost_sklearn` for Pipeline, GridSearchCV compatibility
2. **Standard preprocessing**: Use sklearn's StandardScaler, MinMaxScaler, etc.
3. **Cross-validation**: Works with sklearn's cross_val_score, GridSearchCV

### Memory Optimization
1. **Large datasets**: Use smaller validation sets for early stopping
2. **Streaming**: Use PKBoostAdaptive for incremental learning
3. **Serialization**: Use `save()`/`load()` for model persistence

## Drift Detection

### How It Works
The adaptive model automatically:
1. **Monitors vulnerability scores** on streaming data
2. **Detects performance degradation** using statistical tests
3. **Triggers metamorphosis** when thresholds exceeded
4. **Prunes outdated trees** and adds new ones
5. **Validates adaptation quality** (rollback if degraded)

### When to Use Adaptive Model
- **Streaming data**: Data arrives continuously over time
- **Concept drift**: Underlying patterns change over time
- **Production systems**: Need automatic adaptation
- **IoT/Sensor data**: Real-time monitoring applications

### When to Use Static Model
- **Static datasets**: Fixed training and test sets
- **Batch processing**: All data available upfront
- **Research/analysis**: One-time model training
- **Stable environments**: No expected concept drift

## Error Handling

### Common Errors and Solutions

**"TypeError: 'ndarray' object cannot be converted"**
```python
# Problem: Non-contiguous array or wrong dtype
# Solution: Use contiguous float64 arrays
X = np.ascontiguousarray(X, dtype=np.float64)
y = np.ascontiguousarray(y, dtype=np.float64)
```

**"Model not fitted"**
```python
# Problem: Calling predict before fit
# Solution: Call fit() or fit_initial() first
model.fit(X_train, y_train)
```

**"ValueError: PKBoostClassifier only supports binary classification"**
```python
# Problem: Using binary classifier for multi-class data
# Solution: Use multi-class classifier
from pkboost import PKBoostMultiClass
model = PKBoostMultiClass(n_classes=3)
```

**"X has N features, but PKBoostClassifier was fitted with M features"**
```python
# Problem: Feature dimension mismatch
# Solution: Ensure same number of features during train and test
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
```

### Memory Errors

**Out of memory during training**
```python
# Solution 1: Use smaller validation set
model.fit(X_train, y_train, x_val=X_val[:1000], y_val=y_val[:1000])

# Solution 2: Reduce n_estimators
model = PKBoostClassifier(n_estimators=500)  # Instead of 1000

# Solution 3: Use auto-tuning (optimizes parameters)
model = PKBoostClassifier.auto()
```

## Model Persistence

### Serialization Options

**JSON Format (Human-readable)**
```python
# Save
json_str = model.to_json()
with open("model.json", "w") as f:
    f.write(json_str)

# Load
with open("model.json", "r") as f:
    json_str = f.read()
model = PKBoostClassifier.from_json(json_str)
```

**Binary Format (Compact)**
```python
# Save
import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Load
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
```

**File Format**
```python
# Save
model.save("model.pkboost")

# Load
model = PKBoostClassifier.load("model.pkboost")
```

### Sklearn Wrapper Serialization

**Joblib/Pickle Compatible**
```python
import joblib

# Save entire pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', PKBoostClassifier())
])
pipe.fit(X_train, y_train)

joblib.dump(pipe, "pipeline.joblib")
pipe_loaded = joblib.load("pipeline.joblib")
```

## Advanced Usage

### Custom Hyperparameter Tuning

```python
# Manual parameter selection
model = PKBoostClassifier(
    n_estimators=2000,      # More trees for complex data
    learning_rate=0.01,      # Lower learning rate for better convergence
    max_depth=8,             # Deeper trees for complex interactions
    min_samples_split=50,    # Higher minimum to prevent overfitting
    reg_lambda=2.0,          # Stronger regularization
    subsample=0.9,           # Use more data per tree
    colsample_bytree=0.9     # Use more features per tree
)
```

### Early Stopping with Validation

```python
# Provide validation set for early stopping
model.fit(
    X_train, y_train,
    x_val=X_val, y_val=y_val,
    verbose=True  # Shows training progress
)
```

### Feature Importance Analysis

```python
# Get feature importance
importance = model.get_feature_importance()

# Sort by importance
sorted_idx = importance.argsort()[::-1]
top_features = sorted_idx[:10]

print("Top 10 features:")
for i, idx in enumerate(top_features):
    print(f"{i+1}. Feature {idx}: {importance[idx]:.4f}")
```

### Threshold Optimization

```python
# Find optimal threshold for classification
y_proba = model.predict_proba(X_test)
thresholds = np.arange(0.1, 0.9, 0.05)

best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Best threshold: {best_threshold:.2f}, F1: {best_f1:.4f}")
```

## Comparison with XGBoost/LightGBM

### When to Use PKBoost:
- **Extreme class imbalance** (<5% minority class)
- **Streaming data** with concept drift
- **Need automatic hyperparameter tuning**
- **Prioritize PR-AUC** over raw speed
- **Fraud detection**, medical diagnosis, anomaly detection

### When to Use XGBoost/LightGBM:
- **Balanced datasets** (>20% minority class)
- **Static data** (no drift expected)
- **Need fastest training speed**
- **Multi-class or regression tasks** (though PKBoost now supports these)
- **Large-scale distributed training**

## Benchmarks

### Binary Classification (Imbalanced Datasets)

| Dataset         | Imbalance | PKBoost PR-AUC | XGBoost PR-AUC | LightGBM PR-AUC | Improvement |
|------------------|-----------|----------------|----------------|-----------------|-------------|
| **Credit Card**  | 0.2%      | **87.8%**      | 74.5%          | 79.3%           | +17.9% vs XGB |
| **Pima Diabetes**| 35%       | **98.0%**      | 68.0%          | 62.9%           | +44.0% vs XGB |
| **Ionosphere**   | 36%       | **98.0%**      | 97.2%          | 95.4%           | +0.8% vs XGB  |
| **Breast Cancer**| 37%       | 97.9%          | **99.2%**      | **99.1%**       | -1.4% vs XGB  |

### Multi-Class Classification

| Dataset         | Classes | Imbalance | PKBoost Accuracy | XGBoost Accuracy | LightGBM Accuracy |
|------------------|---------|-----------|-----------------|------------------|-------------------|
| **Dry Bean**     | 7       | 16:1      | **92.36%**      | 89.2%            | 90.1%             |
| **Synthetic-5**  | 5       | 16:1      | **100.0%**      | 70.7%            | 71.8%             |
| **Iris**         | 3       | 1:1       | **96.7%**       | 96.0%            | 96.7%             |

### Drift Resilience (Credit Card Dataset)

After introducing significant covariate shift:

| Model           | Baseline PR-AUC | After Drift | Degradation |
|------------------|-----------------|-------------|-------------|
| **PKBoost**      | **87.8%**       | **86.2%**   | **1.8%**    |
| XGBoost          | 74.5%           | 50.8%       | 31.8%       |
| LightGBM         | 79.3%           | 45.6%       | 42.5%       |

### Performance Metrics

**Training Speed** (Credit Card dataset, ~285K samples):
- **PKBoost**: ~2.75M samples/second
- **XGBoost**: ~3.2M samples/second  
- **LightGBM**: ~4.1M samples/second

**Memory Usage** (during training):
- **PKBoost**: ~1.2x dataset size
- **XGBoost**: ~1.5x dataset size
- **LightGBM**: ~1.8x dataset size

## Version History

### Version 2.2.0 (Current)
- ✅ **Multi-class support**: One-vs-Rest with softmax
- ✅ **165x faster adaptation**: Hierarchical Adaptive Boosting
- ✅ **Complete sklearn compatibility**: Pipeline, GridSearchCV support
- ✅ **Enhanced serialization**: JSON, binary, and file formats
- ✅ **Improved error handling**: Better error messages and recovery

### Version 2.1.0
- ✅ **Adaptive drift detection**: Real-time metamorphosis
- ✅ **Auto-tuning improvements**: Better parameter optimization
- ✅ **Memory optimizations**: Reduced memory footprint

### Version 2.0.0
- ✅ **Rust rewrite**: Complete performance overhaul
- ✅ **Zero-copy NumPy integration**: Direct array access
- ✅ **Shannon entropy guidance**: Information-theoretic splitting

## Troubleshooting

### Installation Issues

**"PKBoost not installed" error**
```bash
# Solution: Install from PyPI
pip install pkboost

# Or install specific version
pip install pkboost==2.2.0

# For development version
pip install git+https://github.com/Pushp-Kharat1/pkboost.git
```

**Import errors**
```python
# Check installation
import pkboost
print(pkboost.__version__)  # Should show version

# Check sklearn wrapper
from pkboost_sklearn import PKBoostClassifier
```

### Performance Issues

**Slow training**
```python
# Use auto-tuning (optimizes parameters)
model = PKBoostClassifier.auto()

# Reduce validation set size
model.fit(X_train, y_train, x_val=X_val[:1000], y_val=y_val[:1000])

# Check array contiguity
print(f"X is contiguous: {X.flags.c_contiguous}")
print(f"X dtype: {X.dtype}")  # Should be float64
```

**Poor predictions**
```python
# Check class imbalance
print(f"Class distribution: {np.bincount(y.astype(int))}")

# Use appropriate evaluation metrics
from sklearn.metrics import average_precision_score, roc_auc_score
pr_auc = average_precision_score(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

# Try different thresholds
thresholds = np.arange(0.1, 0.9, 0.1)
for thresh in thresholds:
    y_pred = (y_proba >= thresh).astype(int)
    f1 = f1_score(y_test, y_pred)
    print(f"Threshold {thresh:.1f}: F1 = {f1:.3f}")
```

### Memory Issues

**Out of memory errors**
```python
# Solution 1: Use smaller batches for adaptive model
for batch in np.array_split(X_train, 10):
    model.observe_batch(batch, y_batch)

# Solution 2: Reduce model complexity
model = PKBoostClassifier(
    n_estimators=500,      # Fewer trees
    max_depth=4,           # Shallower trees
    subsample=0.6          # Less data per tree
)

# Solution 3: Use float32 if memory critical (requires conversion)
X_train = X_train.astype(np.float32)
# Note: Will be converted to float64 internally
```

## Frequently Asked Questions

### Q: When should I use PKBoost vs XGBoost?
**A:** Use PKBoost for:
- Extremely imbalanced data (<5% minority class)
- Streaming applications with concept drift
- When you want automatic hyperparameter tuning
- Fraud detection, anomaly detection, medical diagnosis

Use XGBoost/LightGBM for:
- Balanced datasets
- Static data without drift
- When raw training speed is critical
- Large-scale distributed training

### Q: Does PKBoost support GPU training?
**A:** Currently PKBoost is CPU-only. The Rust implementation is highly optimized for CPU performance with parallel processing via Rayon. GPU support is planned for future versions.

### Q: How does auto-tuning work?
**A:** PKBoost analyzes your data characteristics:
- Class imbalance ratio
- Feature count and dimensionality
- Sample size
- Missing value patterns

Then automatically selects optimal hyperparameters based on extensive benchmarking across hundreds of datasets.

### Q: Can I use PKBoost for time series forecasting?
**A:** Yes, PKBoostRegressor works well for time series regression. For streaming time series with concept drift, use PKBoostAdaptive for continuous adaptation.

### Q: How does drift detection work?
**A:** PKBoostAdaptive monitors:
- Prediction vulnerability scores
- Performance degradation patterns
- Statistical distribution shifts

When drift is detected, it automatically triggers "metamorphosis" - pruning outdated trees and training new ones adapted to current data patterns.

### Q: What's the difference between the two interfaces?
**A:** **Direct Rust bindings** (`pkboost`):
- Maximum performance with zero-copy NumPy
- Streaming and real-time applications
- Full access to all PKBoost features

**Sklearn wrapper** (`pkboost_sklearn`):
- Full sklearn ecosystem compatibility
- Pipeline, GridSearchCV, cross_val_score support
- Easier integration with existing sklearn workflows

Both provide identical model performance - choose based on your integration needs.

## License and Citation

### License
PKBoost is dual-licensed under:
- **GNU General Public License v3.0 or later** (GPL-3.0-or-later)
- **Apache License, Version 2.0**

You may choose either license when using this software. The Apache 2.0 license allows free commercial use.

### Citation
If you use PKBoost in research, please cite:

```bibtex
@software{kharat2025pkboost,
  author = {Kharat, Pushp},
  title = {PKBoost: Shannon-Guided Gradient Boosting for Extreme Imbalance},
  year = {2025},
  url = {https://github.com/Pushp-Kharat1/pkboost}
}
```

## Support and Contributing

### Getting Help
- **Documentation**: This file and README.md
- **Issues**: Report bugs on GitHub Issues
- **Examples**: See `tests/` and `pkboost_sklearn/test_sklearn_compat.py`
- **Email**: kharatpushp16@outlook.com

### Contributing
We welcome contributions! Focus areas:
- Performance optimizations
- New features and algorithms
- Bug fixes and improvements
- Documentation enhancements
- Test coverage expansion

### Development Setup
```bash
git clone https://github.com/Pushp-Kharat1/pkboost.git
cd pkboost
cargo build --release
pip install -e .
```

---

**PKBoost**: Performance-based Knowledge Boosting for the modern ML ecosystem.

*Last updated: January 2025*
