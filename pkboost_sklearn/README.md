# PKBoost Scikit-learn Wrappers

Scikit-learn compatible wrappers for PKBoost models.

## Installation

```bash
pip install pkboost scikit-learn
```

## Usage

### Binary Classification

```python
from pkboost_sklearn import PKBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=1000, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Basic usage
model = PKBoostClassifier(auto_tune=True, verbose=True)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"Accuracy: {model.score(X_test, y_test):.4f}")

# With GridSearchCV
param_grid = {
    'n_estimators': [500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [4, 6, 8]
}
grid = GridSearchCV(PKBoostClassifier(auto_tune=False), param_grid, cv=3, scoring='roc_auc')
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
```

### Regression

```python
from pkboost_sklearn import PKBoostRegressor
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = PKBoostRegressor(auto_tune=True)
model.fit(X_train, y_train)
print(f"R² Score: {model.score(X_test, y_test):.4f}")
```

### Multi-Class Classification

```python
from pkboost_sklearn import PKBoostMultiClass
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = PKBoostMultiClass(n_classes=3)
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.4f}")
```

### Pipeline Integration

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pkboost_sklearn import PKBoostClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', PKBoostClassifier(auto_tune=True))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

## API Compatibility

All wrappers implement standard scikit-learn interfaces:

- `fit(X, y, eval_set=None)` - Train the model
- `predict(X)` - Make predictions
- `predict_proba(X)` - Get probability estimates (classifiers only)
- `score(X, y)` - Evaluate model performance
- `get_params()` / `set_params()` - Get/set hyperparameters

Compatible with:
- `GridSearchCV` / `RandomizedSearchCV`
- `cross_val_score` / `cross_validate`
- `Pipeline`
- All sklearn metrics and utilities

## Differences from Direct API

**Direct API** (from `pkboost` package):
- More control over validation sets
- Access to internal methods like `get_feature_importance()`
- Adaptive models with drift detection

**Sklearn Wrappers** (from `pkboost_sklearn` package):
- Standard sklearn interface
- Works with sklearn tools out of the box
- Simpler for most use cases

## License

MIT License - Same as PKBoost
