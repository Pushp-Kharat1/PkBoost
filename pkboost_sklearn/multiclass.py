"""Scikit-learn compatible multi-class classifier wrapper for PKBoost."""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

try:
    from pkboost import PKBoostMultiClassPy as _PKBoostMultiClass
except ImportError:
    raise ImportError("PKBoost not installed. Run: pip install pkboost")


class PKBoostMultiClass(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible PKBoost multi-class classifier.
    
    Parameters
    ----------
    n_classes : int, optional
        Number of classes. If None, inferred from y during fit.
    verbose : bool, default=False
        Enable verbose output during training.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes labels.
    n_features_in_ : int
        Number of features seen during fit.
    """
    
    def __init__(self, n_classes=None, verbose=False):
        self.n_classes = n_classes
        self.verbose = verbose
        
    def fit(self, X, y, eval_set=None):
        """Fit the PKBoost multi-class model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (class labels).
        eval_set : tuple of (X_val, y_val), optional
            Validation set for early stopping.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y, dtype=np.float64, order='C')
        
        self.classes_ = unique_labels(y)
        n_classes = len(self.classes_)
        
        if self.n_classes is None:
            self.n_classes = n_classes
        elif self.n_classes != n_classes:
            raise ValueError(f"n_classes={self.n_classes} but y has {n_classes} classes")
        
        self.n_features_in_ = X.shape[1]
        
        # Create model
        self._model = _PKBoostMultiClass(n_classes=self.n_classes)
        
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
        
        class_indices = self._model.predict(X)
        return self.classes_[class_indices]
    
    def predict_proba(self, X):
        """Predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ['_model', 'classes_'])
        X = check_array(X, dtype=np.float64, order='C')
        X = np.ascontiguousarray(X, dtype=np.float64)
        return self._model.predict_proba(X)
    
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
