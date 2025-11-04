"""
PKBoost - High-performance gradient boosting with adaptive features

PKBoost is a Rust-accelerated gradient boosting library with:
- Shannon entropy-guided splitting for better minority class detection
- Adaptive learning with automatic drift detection and model evolution
- Full scikit-learn compatibility for easy integration
- Optimized for imbalanced datasets and streaming data

Classes
-------
PKBoostClassifier
    Binary classification with auto-tuning
PKBoostRegressor
    Regression with outlier robustness
PKBoostAdaptiveClassifier
    Adaptive classifier with drift detection

Examples
--------
Basic Classification:
    >>> from pkboost_sklearn import PKBoostClassifier
    >>> clf = PKBoostClassifier(auto=True)
    >>> clf.fit(X_train, y_train, eval_set=(X_val, y_val))
    >>> y_pred = clf.predict(X_test)

With sklearn Pipeline:
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> pipe = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('clf', PKBoostClassifier())
    ... ])
    >>> pipe.fit(X_train, y_train)

Adaptive Learning (streaming):
    >>> clf = PKBoostAdaptiveClassifier()
    >>> clf.fit(X_train, y_train, eval_set=(X_val, y_val))
    >>> for X_batch, y_batch in stream:
    ...     clf.observe_batch(X_batch, y_batch)
    ...     preds = clf.predict(X_batch)
"""

__version__ = "2.0.1"
__author__ = "Pushp Kharat"

from .sklearn_interface import (
    PKBoostClassifier,
    PKBoostRegressor,
    PKBoostAdaptiveClassifier
)

__all__ = [
    'PKBoostClassifier',
    'PKBoostRegressor',
    'PKBoostAdaptiveClassifier',
    '__version__',
]


def get_config():
    """Get PKBoost configuration and build information."""
    import platform
    return {
        'version': __version__,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
    }


def show_versions():
    """Print version information for debugging."""
    import sys
    try:
        import numpy as np
        import sklearn
        
        config = get_config()
        
        print("PKBoost Configuration")
        print("=" * 60)
        print(f"PKBoost version:      {config['version']}")
        print(f"Python version:       {config['python_version']}")
        print(f"NumPy version:        {np.__version__}")
        print(f"Scikit-learn version: {sklearn.__version__}")
        print(f"Platform:             {config['platform']}")
        print("=" * 60)
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
