"""Scikit-learn compatible classifier wrapper for PKBoost."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

try:
    from pkboost import PKBoostClassifier as _PKBoostClassifier
except ImportError:
    raise ImportError("PKBoost not installed. Run: pip install pkboost")


class PKBoostClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible PKBoost binary classifier.
    
    Parameters
    ----------
    n_estimators : int, default=1000
        Number of boosting rounds.
    learning_rate : float, default=0.05
        Learning rate shrinks the contribution of each tree.
    max_depth : int, default=6
        Maximum depth of a tree.
    min_samples_split : int, default=20
        Minimum number of samples required to split an internal node.
    min_child_weight : float, default=1.0
        Minimum sum of instance weight needed in a child.
    reg_lambda : float, default=1.0
        L2 regularization term on weights.
    gamma : float, default=0.0
        Minimum loss reduction required to make a split.
    subsample : float, default=0.8
        Subsample ratio of the training instances.
    colsample_bytree : float, default=0.8
        Subsample ratio of columns when constructing each tree.
    scale_pos_weight : float, default=1.0
        Balancing of positive and negative weights.
    auto_tune : bool, default=True
        Automatically tune hyperparameters based on data.
    verbose : bool, default=False
        Enable verbose output during training.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    n_features_in_ : int
        Number of features seen during fit.
    """
    
    def __init__(self, n_estimators=1000, learning_rate=0.05, max_depth=6,
                 min_samples_split=20, min_child_weight=1.0, reg_lambda=1.0,
                 gamma=0.0, subsample=0.8, colsample_bytree=0.8,
                 scale_pos_weight=1.0, auto_tune=True, verbose=False):
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
        self.auto_tune = auto_tune
        self.verbose = verbose
        
    def fit(self, X, y, eval_set=None):
        """Fit the PKBoost model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (0 or 1).
        eval_set : tuple of (X_val, y_val), optional
            Validation set for early stopping.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y, dtype=np.float64, order='C')
        
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("PKBoostClassifier only supports binary classification")
        
        self.n_features_in_ = X.shape[1]
        
        # Create model
        if self.auto_tune:
            self._model = _PKBoostClassifier.auto()
        else:
            self._model = _PKBoostClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_child_weight=self.min_child_weight,
                reg_lambda=self.reg_lambda,
                gamma=self.gamma,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                scale_pos_weight=self.scale_pos_weight
            )
        
        # Prepare eval_set
        x_val, y_val = None, None
        if eval_set is not None:
            x_val, y_val = eval_set
            x_val = check_array(x_val, dtype=np.float64, order='C')
            y_val = np.ascontiguousarray(y_val, dtype=np.float64)
        
        # Fit
        X = np.ascontiguousarray(X, dtype=np.float64)
        y = np.ascontiguousarray(y, dtype=np.float64)
        self._model.fit(X, y, x_val=x_val, y_val=y_val, verbose=self.verbose)
        
        return self
    
    def predict(self, X):
        """Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ['_model', 'classes_'])
        X = check_array(X, dtype=np.float64, order='C')
        X = np.ascontiguousarray(X, dtype=np.float64)
        return self._model.predict(X, threshold=0.5)
    
    def predict_proba(self, X):
        """Predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Class probabilities.
        """
        check_is_fitted(self, ['_model', 'classes_'])
        X = check_array(X, dtype=np.float64, order='C')
        X = np.ascontiguousarray(X, dtype=np.float64)
        
        proba_pos = self._model.predict_proba(X)
        proba = np.column_stack([1 - proba_pos, proba_pos])
        return proba
    
    def score(self, X, y):
        """Return the mean accuracy.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
            
        Returns
        -------
        score : float
            Mean accuracy.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))
    
    def get_feature_importance(self):
        """Get feature importance scores.
        
        Returns
        -------
        importance : ndarray of shape (n_features,)
            Feature importance scores.
        """
        check_is_fitted(self, ['_model'])
        return self._model.get_feature_importance()
