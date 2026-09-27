import numpy as np
from scipy.optimize import minimize_scalar

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold

from bblr.lowess import tricube_weight
from bblr.ci import *


class LPRRegressor(BaseEstimator, RegressorMixin):
    """LPR (Local Polynomial Regression) Regressor.

    Fits a degree-p polynomial locally around each test point by centering
    the training data at that point before computing polynomial features.
    This avoids the near-collinearity of global polynomial features inside
    a local neighbourhood.
    """
    # TODO: return error if frac too large or too small

    def __init__(self, frac=0.5, degree=2):
        self.frac = frac
        self.degree = degree
        self.X_train = None
        self.y_train = None
        self.is_fitted = False

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.is_fitted = True
        return self

    def predict(self, X_test, confidence=0.95, return_ci=False):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        X_test = np.array(X_test)

        # fit once — only records n_features_in_, not data values
        poly = PolynomialFeatures(degree=self.degree, include_bias=False)
        poly.fit(self.X_train)

        model = LinearRegression()
        y_pred = []
        coefs = []
        intercepts = []
        conf_intervals = []
        std_errors_all = []

        for x0 in X_test:
            # Shift training data so x0 maps to the origin.
            # Near x0 the centered features are small and near-orthogonal,
            # avoiding the multicollinearity of global polynomial features.
            X_centered = self.X_train - x0
            X_centered_poly = poly.transform(X_centered)

            weights = tricube_weight(self.X_train, x0, self.frac)
            model.fit(X_centered_poly, self.y_train, sample_weight=weights)

            # At x0 all centered features are 0 → ŷ(x0) = intercept only
            y_pred.append(model.intercept_)
            coefs.append(model.coef_)
            intercepts.append(model.intercept_)

            if return_ci:
                y_fitted = model.predict(X_centered_poly)
                conf_int, std_errors = compute_se_construct_ci(
                    X_centered_poly, self.y_train,
                    y_fitted, model.intercept_, model.coef_,
                    weights, confidence=confidence)
            else:
                conf_int = [(0, 0)] * (X_centered_poly.shape[1] + 1)
                std_errors = 0
            conf_intervals.append(conf_int)
            std_errors_all.append(std_errors)

        self.coefs_ = np.array(coefs)
        self.intercepts_ = np.array(intercepts)
        self.conf_intervals_ = np.array(conf_intervals)
        self.std_errors_ = np.array(std_errors_all)
        self.y_pred_ = np.array(y_pred)
        return np.array(y_pred)

    def predict_all(self):
        return self.y_pred_, self.coefs_, self.intercepts_, self.conf_intervals_, self.std_errors_


class LocalPolynomialRegression:
    """
    Local polynomial regression with derivative estimation.

    Parameters:
    -----------
    degree : int
        Degree of polynomial (2 or 3)
    frac : float or array-like
        Fraction parameter for LOWESS
    """

    def __init__(self, degree=2, frac=0.3, ci=False):
        if degree not in [2, 3]:
            raise ValueError("Only degree 2 and 3 are supported")
        self.degree = degree
        self.frac = frac
        self.X_train = None
        self.y_train = None
        self.ci = ci

    def fit(self, X, y):
        """Store training data"""
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.d = self.X_train.shape[1]
        if self.d == 1:
            self.X_train = self.X_train.reshape(-1, 1)
        return self

    def predict(self, X_test, return_derivatives=True):
        """
        Predict at new points with optional derivative estimation.

        Parameters:
        -----------
        X_test : array-like, shape (n_pred, d)
            Points at which to predict
        return_derivatives : bool
            If True, also return derivative estimates

        Returns:
        --------
        y_pred : array, shape (n_pred,)
            Predicted values
        coefs : array, shape (n_pred, d)
            Partial derivatives ∂ŷ/∂x_i at each test point.
            With centering these are directly the first d coefficients of
            the locally fitted polynomial — no further formula needed.
        intercepts : array, shape (n_pred,)
            Local model intercepts (equal to y_pred with centering)
        cis_raw : confidence intervals (tbd)
        std_errors : standard errors (tbd)
        """
        self.X_test = np.asarray(X_test)

        if self.X_train.ndim == 1:
            self.X_train = self.X_train.reshape(-1, 1)

        if self.X_test.ndim == 1:
            self.X_test = self.X_test.reshape(-1, 1)

        poly_reg = LPRRegressor(frac=self.frac, degree=self.degree)
        poly_reg.fit(self.X_train, self.y_train)
        poly_reg.predict(self.X_test)
        y_pred, raw_coefs, intercepts, cis_raw, std_errors = poly_reg.predict_all()

        # PolynomialFeatures (include_bias=False) always places degree-1 terms
        # first: [x1, x2, ..., xd, x1², x1·x2, ...].
        # After centering at x0, the coefficient of (xi − x0i) is exactly
        # ∂ŷ/∂xi, so the first d columns of raw_coefs are the partial derivatives.
        coefs = raw_coefs[:, :self.d]

        # TODO: compute confidence intervals and standard errors if necessary

        return y_pred, coefs, intercepts, cis_raw, std_errors

# Define the frac parameter optimising function
def lpr_optimal_frac(X_train, y_train, frac_range, degree=2, k=10, maxiter=15, xatol=0.001):
    # k - number of splits in k-fold CV

    def lpr_mse(frac):
        # divide row indices into k parts
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        # store values predicted by LPR in each fold
        y_pred_lpr = np.array([])

        # store true values of y
        y_true = np.array([])

        for fold, (train_index, test_index) in enumerate(kf.split(X_train)):
            # divide the whole training set into a training and test set for a certain fold 
            X_train_fold, X_test_fold = X_train[train_index, :], X_train[test_index, :]
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

            lpr_fold = LocalPolynomialRegression(frac=frac, degree=degree)
            
            lpr_fold.fit(X_train_fold, y_train_fold)
            y_pred_fold, _, _, _, _ = lpr_fold.predict(X_test_fold)

            y_true = np.append(y_true, y_test_fold)
            y_pred_lpr = np.append(y_pred_lpr, y_pred_fold)

        return mean_squared_error(y_pred_lpr, y_true)

    # Perform the optimization
    result = minimize_scalar(lpr_mse, bounds=frac_range, method='bounded', options={'maxiter': maxiter, 'xatol': xatol})

    # return the value of the optimised parameter
    return result.x



