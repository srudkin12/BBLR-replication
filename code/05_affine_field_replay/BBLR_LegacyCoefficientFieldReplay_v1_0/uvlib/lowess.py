import numpy as np

from scipy.spatial import distance
from scipy.optimize import minimize_scalar

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import sys
sys.path.append("../")
from bblr.ci import *



def tricube_weight(x, x0, frac):
    """Compute the tricube weight for each data point."""
    if len(x.shape) == 1:
        d = np.array([distance.euclidean(np.array([xi]), np.array([x0])) for xi in x])
    else:
        d = np.array([distance.euclidean(np.array(xi), np.array(x0)) for xi in x])
    bandwidth = np.percentile(d, frac * 100)
    u = d / bandwidth
    w = np.where(u < 1, (1 - u**3)**3, 0)
    return w

class LOWESSRegressor(BaseEstimator, RegressorMixin):
    # TODO: return error if frac too large or too small
    """LOWESS (Locally Weighted Scatterplot Smoothing) Regressor."""

    def __init__(self, frac=0.5):
        """
        Initialize LOWESS regressor.

        Parameters:
        -----------
        frac : float, default=0.5
            Fraction of data points to use for local regression (between 0 and 1)
        """
        self.frac = frac
        self.X_train = None
        self.y_train = None
        self.is_fitted = False

    def fit(self, X_train, y_train):
        """
        Fit the LOWESS model on training data.

        Parameters:
        -----------
        X_train : array-like of shape (n_samples, n_features)
            Training input samples
        y_train : array-like of shape (n_samples,)
            Training target values
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.is_fitted = True
        return self

    def predict(self, X_test, confidence=0.95, return_ci=False):
        """
        Predict using the LOWESS model.

        Parameters:
        -----------
        X_test : array-like of shape (n_samples, n_features)
            Test input samples
        confidence : float, default=0.95
            Confidence level for confidence intervals
        return_ci : bool, default=False
            Whether to return confidence intervals

        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted values
        conf_intervals : array of shape (n_samples, n_features+1, 2), optional
            Confidence intervals for coefficients (returned if return_ci=True)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_test = np.array(X_test)
        model = LinearRegression()
        y_pred = []
        coefs = []
        intercepts = []
        conf_intervals = [] # if return_ci else None
        std_errors_all = []

        for x0 in X_test:
            # Calculate weights based on distance from test point to training points
            weights = tricube_weight(self.X_train, x0, self.frac)

            # Fit local regression model
            model.fit(self.X_train, self.y_train, sample_weight=weights)

            # Make prediction for this test point
            y_hat = model.predict(x0.reshape(1, -1))[0]
            y_pred.append(y_hat)
            coefs.append(model.coef_)
            intercepts.append(model.intercept_)

            if return_ci:
                y_fitted = model.predict(self.X_train)
                conf_int, std_errors = compute_se_construct_ci(self.X_train, self.y_train,
                                                               y_fitted, model.intercept_, model.coef_,
                                                               weights, confidence = confidence)
            else:
                # conf_int = np.array([0, 0]) * (self.X_train.shape[1] + 1)
                conf_int = [(0, 0)] * (self.X_train.shape[1] + 1)
                std_errors = 0
            conf_intervals.append(conf_int)
            std_errors_all.append(std_errors)

        # return np.array(y_pred), np.array(coefs), np.array(intercepts), np.array(conf_intervals), np.array(std_errors_all)
        self.coefs_ = np.array(coefs)
        self.intercepts_ = np.array(intercepts)
        self.conf_intervals_ = np.array(conf_intervals)
        self.std_errors_ = np.array(std_errors_all)
        self.y_pred_ = np.array(y_pred)
        return np.array(y_pred)
    
    def predict_all(self):
        return self.y_pred_, self.coefs_, self.intercepts_, self.conf_intervals_, self.std_errors_


# Define the frac parameter optimising function
def lowess_optimal_frac(X_train, y_train, frac_range, k=10, maxiter=15, xatol=0.001):
    # k - number of splits in k-fold CV

    def lowess_mse(frac):
        # divide row indices into k parts
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        # store values predicted by LOESS in each fold
        y_pred_lowess = np.array([])

        # store true values of y
        y_true = np.array([])

        for fold, (train_index, test_index) in enumerate(kf.split(X_train)):
            # divide the whole training set into a training and test set for a certain fold 
            X_train_fold, X_test_fold = X_train[train_index, :], X_train[test_index, :]
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

            lowess_fold = LOWESSRegressor(frac=frac)
            
            lowess_fold.fit(X_train_fold, y_train_fold)
            y_pred_fold = lowess_fold.predict(X_test_fold)

            y_true = np.append(y_true, y_test_fold)
            y_pred_lowess = np.append(y_pred_lowess, y_pred_fold)

        return mean_squared_error(y_pred_lowess, y_true)

    # Perform the optimization
    result = minimize_scalar(lowess_mse, bounds=frac_range, method='bounded', options={'maxiter': maxiter, 'xatol': xatol})

    # return the value of the optimised parameter
    return result.x







