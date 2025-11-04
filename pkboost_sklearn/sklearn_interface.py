"""
Scikit-learn compatible interface for PKBoost

This module provides full sklearn compatibility including:
- BaseEstimator and ClassifierMixin/RegressorMixin inheritance
- fit(), predict(), predict_proba() methods
- get_params() and set_params() for hyperparameter tuning
- Feature importance via feature_importances_ property
- Support for sklearn's Pipeline, GridSearchCV, cross_val_score, etc.

Usage:
    from pkboost_sklearn.sklearn_interface import PKBoostClassifier, PKBoostRegressor
    
    # Classification
    clf = PKBoostClassifier(n_estimators=500, learning_rate=0.05)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # With GridSearchCV
    from sklearn.model_selection import GridSearchCV
    params = {'n_estimators': [100, 500], 'max_depth': [4, 6, 8]}
    grid = GridSearchCV(PKBoostClassifier(), params, cv=5)
    grid.fit(X_train, y_train)
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from typing import Optional, Union

try:
    import pkboost
except ImportError:
    raise ImportError("PKBoost not installed. Run: pip install pkboost")


class PKBoostClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible PKBoost binary classifier.
    
    Parameters
    ----------
    n_estimators : int, default=1000
        Number of boosting iterations (trees).
    
    learning_rate : float, default=0.05
        Step size shrinkage to prevent overfitting.
    
    max_depth : int, default=6
        Maximum depth of each tree.
    
    min_samples_split : int, default=20
        Minimum samples required to split a node.
    
    min_child_weight : float, default=1.0
        Minimum sum of instance weight (hessian) in a child.
    
    reg_lambda : float, default=1.0
        L2 regularization term on weights.
    
    gamma : float, default=0.0
        Minimum loss reduction required to make a split.
    
    subsample : float, default=0.8
        Fraction of samples to use for each tree.
    
    colsample_bytree : float, default=0.8
        Fraction of features to use for each tree.
    
    scale_pos_weight : float, default=1.0
        Balancing weight for positive class (useful for imbalanced data).
    
    auto : bool, default=False
        If True, automatically tune hyperparameters based on data characteristics.
    
    random_state : int, optional
        Random seed for reproducibility (not yet implemented in Rust backend).
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    
    n_features_in_ : int
        Number of features seen during fit.
    
    feature_importances_ : ndarray of shape (n_features,)
        Feature importance scores based on usage in trees.
    
    n_trees_ : int
        Actual number of trees after early stopping.
    
    Examples
    --------
    >>> from pkboost_sklearn.sklearn_interface import PKBoostClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> 
    >>> X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> 
    >>> clf = PKBoostClassifier(n_estimators=500, learning_rate=0.05)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    >>> print(f"Accuracy: {clf.score(X_test, y_test):.3f}")
    """
    
    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        min_samples_split: int = 20,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        gamma: float = 0.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: float = 1.0,
        auto: bool = False,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.auto = auto
        self.random_state = random_state
    
    def fit(
        self,
        X,
        y,
        sample_weight=None,
        eval_set: Optional[tuple] = None,
        verbose: bool = False,
    ):
        """
        Fit the PKBoost classifier.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        y : array-like of shape (n_samples,)
            Target values (0 or 1 for binary classification).
        
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights (not yet implemented).
        
        eval_set : tuple (X_val, y_val), optional
            Validation set for early stopping.
        
        verbose : bool, default=False
            Whether to print training progress.
        
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        if len(self.classes_) != 2:
            raise ValueError(
                f"PKBoostClassifier only supports binary classification. "
                f"Got {len(self.classes_)} classes. "
                f"Use PKBoostMultiClass for multi-class problems."
            )
        
        y = y.astype(np.float64)
        if not np.all(np.isin(y, [0, 1])):
            y = (y == self.classes_[1]).astype(np.float64)
        
        X_val, y_val = None, None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val = check_array(X_val, accept_sparse=False, dtype=np.float64)
            y_val = y_val.astype(np.float64)
            if not np.all(np.isin(y_val, [0, 1])):
                y_val = (y_val == self.classes_[1]).astype(np.float64)
        
        if self.auto:
            self._model = pkboost.PKBoostClassifier.auto()
        else:
            self._model = pkboost.PKBoostClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                scale_pos_weight=self.scale_pos_weight,
            )
        
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        if X_val is not None:
            X_val = np.ascontiguousarray(X_val, dtype=np.float64)
            y_val = np.ascontiguousarray(y_val, dtype=np.float64)
        
        self._model.fit(X, y, x_val=X_val, y_val=y_val, verbose=verbose)
        
        self.feature_importances_ = self._model.get_feature_importance()
        self.n_trees_ = self._model.get_n_trees()
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but PKBoostClassifier "
                f"was fitted with {self.n_features_in_} features."
            )
        
        X = np.ascontiguousarray(X, dtype=np.float64)
        proba_class_1 = self._model.predict_proba(X)
        proba_class_0 = 1.0 - proba_class_1
        
        return np.column_stack([proba_class_0, proba_class_1])
    
    def predict(self, X, threshold: float = 0.5):
        """Predict class labels."""
        proba = self.predict_proba(X)[:, 1]
        predictions = (proba >= threshold).astype(int)
        return self.classes_[predictions]
    
    def predict_log_proba(self, X):
        """Predict class log-probabilities."""
        proba = self.predict_proba(X)
        return np.log(np.clip(proba, 1e-15, 1.0))
    
    def decision_function(self, X):
        """Compute the decision function."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        X = np.ascontiguousarray(X, dtype=np.float64)
        return self._model.predict_proba(X)
    
    def score(self, X, y, sample_weight=None):
        """Return the mean accuracy."""
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)


class PKBoostRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible PKBoost regressor."""
    
    def __init__(self, auto: bool = True, random_state: Optional[int] = None):
        self.auto = auto
        self.random_state = random_state
    
    def fit(self, X, y, sample_weight=None, eval_set: Optional[tuple] = None, verbose: bool = False):
        """Fit the regressor."""
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, y_numeric=True)
        
        self.n_features_in_ = X.shape[1]
        y = y.astype(np.float64)
        
        X_val, y_val = None, None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val = check_array(X_val, accept_sparse=False, dtype=np.float64)
            y_val = y_val.astype(np.float64)
        
        if self.auto:
            self._model = pkboost.PKBoostRegressorPy.auto()
        else:
            self._model = pkboost.PKBoostRegressorPy()
        
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        if X_val is not None:
            X_val = np.ascontiguousarray(X_val, dtype=np.float64)
            y_val = np.ascontiguousarray(y_val, dtype=np.float64)
        
        self._model.fit(X, y, x_val=X_val, y_val=y_val, verbose=verbose)
        self.feature_importances_ = np.zeros(self.n_features_in_)
        
        return self
    
    def predict(self, X):
        """Predict regression targets."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but was fitted with "
                f"{self.n_features_in_} features."
            )
        
        X = np.ascontiguousarray(X, dtype=np.float64)
        return self._model.predict(X)
    
    def score(self, X, y, sample_weight=None):
        """Return R² score."""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)


class PKBoostAdaptiveClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible adaptive PKBoost classifier with drift detection."""
    
    def __init__(self, auto: bool = True):
        self.auto = auto
    
    def fit(self, X, y, eval_set: Optional[tuple] = None, verbose: bool = False):
        """Fit the adaptive classifier."""
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
        
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        if len(self.classes_) != 2:
            raise ValueError("PKBoostAdaptiveClassifier only supports binary classification.")
        
        y = (y == self.classes_[1]).astype(np.float64)
        
        X_val, y_val = None, None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val = check_array(X_val, accept_sparse=False, dtype=np.float64)
            y_val = (y_val == self.classes_[1]).astype(np.float64)
        
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        if X_val is not None:
            X_val = np.ascontiguousarray(X_val, dtype=np.float64)
            y_val = np.ascontiguousarray(y_val, dtype=np.float64)
        
        self._model = pkboost.PKBoostAdaptive()
        self._model.fit_initial(X, y, x_val=X_val, y_val=y_val, verbose=verbose)
        
        return self
    
    def observe_batch(self, X, y, verbose: bool = False):
        """Observe a new batch of data and potentially trigger adaptation."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        y = (y == self.classes_[1]).astype(np.float64)
        
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        
        self._model.observe_batch(X, y, verbose=verbose)
        
        self.vulnerability_score_ = self._model.get_vulnerability_score()
        self.metamorphosis_count_ = self._model.get_metamorphosis_count()
        self.state_ = self._model.get_state()
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        check_is_fitted(self)
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        X = np.ascontiguousarray(X, dtype=np.float64)
        
        proba_class_1 = self._model.predict_proba(X)
        proba_class_0 = 1.0 - proba_class_1
        
        return np.column_stack([proba_class_0, proba_class_1])
    
    def predict(self, X, threshold: float = 0.5):
        """Predict class labels."""
        proba = self.predict_proba(X)[:, 1]
        predictions = (proba >= threshold).astype(int)
        return self.classes_[predictions]
    
    def get_status(self):
        """Get current adaptation status."""
        check_is_fitted(self)
        return {
            'state': self.state_,
            'vulnerability_score': self.vulnerability_score_,
            'metamorphosis_count': self.metamorphosis_count_,
        }


__all__ = ['PKBoostClassifier', 'PKBoostRegressor', 'PKBoostAdaptiveClassifier']
