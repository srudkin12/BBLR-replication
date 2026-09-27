import numpy as np
from scipy.stats import t


def compute_se_construct_ci(X, y, y_fitted, intercept, coefs, weights, confidence=0.95):
    """
    Compute standard errors and construct confidence intervals for linear regression coefficients.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Design matrix (predictor variables)
    y : array-like, shape (n_samples,)
        Response variable
    y_fitted : array-like, shape (n_samples,)
        Fitted values from the regression
    intercept : float
        Intercept term from the regression
    coefs : array-like, shape (n_features,)
        Regression coefficients (excluding intercept)
    weights : array-like, shape (n_samples,) or None
        Sample weights for weighted least squares. If None, uses ordinary least squares.
    confidence : float, default=0.95
        Confidence level for the intervals (between 0 and 1)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'standard_errors': array of standard errors for [intercept, *coefs]
        - 'confidence_intervals': array of shape (n_params, 2) with [lower, upper] bounds
        - 'residual_se': residual standard error
        - 'dof': degrees of freedom
    """
    X = np.asarray(X)
    y = np.asarray(y)
    y_fitted = np.asarray(y_fitted)
    coefs = np.asarray(coefs)
    
    n_samples, n_features = X.shape
    n_params = n_features + 1  # Including intercept
    
    # Degrees of freedom
    dof = n_samples - n_params
    
    if dof <= 0:
        raise ValueError(f"Insufficient degrees of freedom: {dof}. Need more samples than parameters to compute confidence intervals.")
    
    # Compute residuals
    residuals = y - y_fitted
    
    # Handle weights
    if weights is None:
        weights = np.ones(n_samples)
    else:
        weights = np.asarray(weights)
        if weights.shape[0] != n_samples:
            raise ValueError("weights must have the same length as y")
    
    # Compute residual standard error (sigma_hat)
    # For weighted least squares: sigma^2 = RSS / dof where RSS = sum(w_i * e_i^2)
    weighted_rss = np.sum(weights * residuals**2)
    residual_se = np.sqrt(weighted_rss / dof)
    
    # Construct design matrix with intercept column
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    
    # Weight the design matrix for WLS
    W_sqrt = np.sqrt(weights)
    X_weighted = X_with_intercept * W_sqrt[:, np.newaxis]
    
    # Compute (X^T W X)^{-1} for variance-covariance matrix
    try:
        XtWX = X_weighted.T @ X_weighted
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        raise ValueError("Design matrix is singular. Cannot compute standard errors.")
    
    # Variance-covariance matrix: Var(beta_hat) = sigma^2 * (X^T W X)^{-1}
    var_covar_matrix = (residual_se ** 2) * XtWX_inv
    
    # Standard errors are the square root of diagonal elements
    standard_errors = np.sqrt(np.diag(var_covar_matrix))
    
    # All parameters: [intercept, coef_1, ..., coef_p]
    all_params = np.concatenate([[intercept], coefs])
    
    # Compute confidence intervals using t-distribution
    alpha = 1 - confidence
    t_critical = t.ppf(1 - alpha/2, dof)
    
    margin_of_error = t_critical * standard_errors
    lower_bounds = all_params - margin_of_error
    upper_bounds = all_params + margin_of_error
    
    confidence_intervals = np.column_stack([lower_bounds, upper_bounds])

    return confidence_intervals, standard_errors
    
