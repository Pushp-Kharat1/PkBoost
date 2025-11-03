"""Scikit-learn compatible regressor wrapper for PKBoost."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

try:
    from pkboost import PKBoostRegressorPy as _PKBoostRegressor
except ImportError:
    raise ImportError("PKBoost not installed. Run: pip install pkboost")


class PKBoostRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible PKBoost regressor.
    
    Parameters
    ----------
    auto_tune : bool, default=True
        Automatically tune hyperparameters based on data.
    verbose : bool, default=False
        Enable verbose output during training.
    
    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    """
    
    def __init__(self, auto_tune=True, verbose=False):
        self.auto_tune = auto_tune
        self.verbose = verbose
        
    def fit(self, X, y, eval_set=None):
        """Fit the PKBoost regressor.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        eval_set : tuple of (X_val, y_val), optional
            Validation set for early stopping.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y, dtype=np.float64, order='C')
        self.n_features_in_ = X.shape[1]
        
        # Create model
        if self.auto_tune:
            self._model = _PKBoostRegressor.auto()
        else:
            self._model = _PKBoostRegressor()
        
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
        """Predict target values.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, ['_model'])
        X = check_array(X, dtype=np.float64, order='C')
        X = np.ascontiguousarray(X, dtype=np.float64)
        return self._model.predict(X)
    
    def score(self, X, y):
        """Return the R² score.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values.
            
        Returns
        -------
        score : float
            R² score.
        """
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))
