import pandas as pd
import numpy as np

import statsmodels.api as sm

from scipy.spatial import distance
from scipy.stats import uniform
from scipy.optimize import minimize_scalar

import sys

from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, root_mean_squared_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

import warnings
warnings.filterwarnings('ignore')

sys.path.append("../")
from bblr.bblr import *

method_list=['BBLR', 'LOESS', 'OLS', 'SVR']
# for BBLR and LOESS methods the parameter value 0 means that the optimal parameter is found for each dataset
params_dict={'BBLR':[0, 0.5, 0.7, 1.0], 'LOESS':[0.1, 0.3, 0.5], 'OLS':[0], 'SVR':[0]}
# for methods OLS and SVR we don't have additional parameters, thus they are set to 0
# for SVR optimal values of the parameters are found

def svr_hypertune_and_predict(X, y, n_iter=20, random_state=None):
    """
    Perform hyperparameter tuning for SVR using randomized search, make predictions,
    and return coefficients.

    Parameters:
    X (pd.DataFrame or np.array): Feature matrix
    y (pd.Series or np.array): Target variable
    n_iter (int): Number of parameter settings sampled
    random_state (int): Random seed for reproducibility

    Returns:
    predictions
    """

    # Convert to numpy arrays if they're DataFrames/Series
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X = X.values
    else:
        feature_names = None

    if isinstance(y, pd.Series):
        y = y.values

    # Create a pipeline with SVR
    pipeline = Pipeline([
        ('svr', SVR(kernel='rbf'))  # Using linear kernel to get coefficients
    ])

    # Define parameter distributions for random sampling
    param_dist = {
        'svr__C': uniform(1, 1e2),  # Logarithmic distribution between 0.001 and 1000
        'svr__epsilon': uniform(0.005, 0.2),  # Uniform distribution between 0.01 and 1.01
    }

    # Use negative RMSE as scoring (higher is better)
    scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        verbose=0,
        random_state=random_state
    )

    # Fit the model
    random_search.fit(X, y)

    # Get best model
    best_model = random_search.best_estimator_

    return best_model

# Define the optimising function
def loess_optimal_frac(X_train, y_train, X_test, y_test, frac_range):
    def loess_mse(frac):
        # define loess model and make predictions
        loess = LOESSRegressor(frac=frac)
        loess.fit(X_train, y_train)
        yp, coefs, cis, stds = loess.predict(X_test)
        return mean_squared_error(yp, y_test)

    # Perform the optimization
    result = minimize_scalar(loess_mse, bounds=frac_range, method='bounded', options={'maxiter': 20, 'xatol': 0.01})

    return result.x

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

class LOESSRegressor:
    """LOESS (Locally Weighted Scatterplot Smoothing) Regressor."""

    def __init__(self, frac=0.5):
        """
        Initialize LOESS regressor.

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
        Fit the LOESS model on training data.

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
        Predict using the LOESS model.

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

            # if return_ci:
                # Compute confidence intervals for coefficients
                # y_fitted = model.predict(self.X_train)
                # residuals = self.y_train - y_fitted
                # dof = len(self.y_train) - (self.X_train.shape[1] + 1)  # Degrees of freedom
                #
                # if dof > 0:
                #     mse = np.sum((residuals ** 2) * weights) / dof
                #     X_design = np.column_stack((np.ones(len(self.y_train)), self.X_train))
                #
                #     try:
                #         XtWX_inv = np.linalg.inv(X_design.T @ np.diag(weights) @ X_design)
                #         std_errors = np.sqrt(np.diag(mse * XtWX_inv))
                #         t_val = t.ppf((1 + confidence) / 2, dof)
                #
                #         # Confidence intervals for intercept and coefficients
                #         all_coefs = np.hstack([model.intercept_, model.coef_])
                #         conf_int = [(coef - t_val * se, coef + t_val * se)
                #                    for coef, se in zip(all_coefs, std_errors)]
                #         conf_intervals.append(conf_int)
                #         std_errors_all.append(std_errors)
                #     except np.linalg.LinAlgError:
                #         # Handle singular matrix case
                #         conf_intervals.append([(np.nan, np.nan)] * (self.X_train.shape[1] + 1))
                # else:
                #     conf_intervals.append([(np.nan, np.nan)] * (self.X_train.shape[1] + 1))
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

        return np.array(y_pred), np.array(coefs), np.array(conf_intervals), np.array(std_errors_all)



def loess_regression(X_train, y_train, X_test, frac=0.5, confidence=0.95, return_ci=False):
    """
    Convenience function for LOESS regression with train/test split.

    Parameters:
    -----------
    X_train : array-like of shape (n_train_samples, n_features)
        Training input samples
    y_train : array-like of shape (n_train_samples,)
        Training target values
    X_test : array-like of shape (n_test_samples, n_features)
        Test input samples
    frac : float, default=0.5
        Fraction of data points to use for local regression
    confidence : float, default=0.95
        Confidence level for confidence intervals
    return_ci : bool, default=False
        Whether to return confidence intervals

    Returns:
    --------
    y_pred : array of shape (n_test_samples,)
        Predicted values for test set
    conf_intervals : array, optional
        Confidence intervals (returned if return_ci=True)
    """
    regressor = LOESSRegressor(frac=frac)
    regressor.fit(X_train, y_train)

    if return_ci:
        return regressor.predict(X_test, confidence=confidence, return_ci=True)
    else:
        return regressor.predict(X_test)

# Define all DGP functions
def generate_data(n, dgp_name, d, noise_var = 0.2, seed=42, x_distribution='normal'):
    # Set random seed for reproducibility
    np.random.seed(seed)

    # TODO: other distribution: lognormal, uniform etc.
    match x_distribution:
        case 'normal':
            x = np.random.normal(0, 1, (n, d))
        case 'uniform':
            x = np.random.uniform(0, 1, (n, d))
        case 'lognormal':
            x = np.random.lognormal(0, 1, (n, d))
        case 'weibull':
            x = np.random.weibull(0.7, (n, d))
        case 'pareto':
            x = np.random.pareto(0.7, (n, d))
        case _:
            raise ValueError(f"Unknown distribution: {x_distribution}")
    eps = np.random.normal(0, noise_var, n)

    match dgp_name:
        case 'UA':
            y = x[:,0] + eps
        case 'UB':
            y = x[:,0]**2 + eps
        case 'UC':
            # Step function with 7 steps: nonincreasing on (-∞,0), nondecreasing on (0,∞)
            # Define step boundaries and values
            steps = np.array([-1.5, -1, -0.5, 0.5, 1, 1.5])  # 6 boundaries create 7 steps
            step_values = np.array([1.5, 1, 0.5, 0, 0.5, 1, 1.5])  # 7 step values

            # Use np.digitize to assign each x value to a step
            step_indices = np.digitize(x[:,0], steps)
            y = step_values[step_indices] + eps

        case 'UD':
            # Linear relation with one discontinuity at x=0
            # Different coefficients before and after discontinuity
            coeff_neg = -2.0  # coefficient for x < 0
            coeff_pos = 1.5   # coefficient for x >= 0
            y = np.where(x[:,0] < 0, coeff_neg * x[:,0], coeff_pos * x[:,0]) + 2 * (x[:,0] > 0) + eps

        case 'UE':
            # Linear relation with two discontinuities
            # Three different coefficients: positive, negative, positive
            breakpoint1 = -1.0
            breakpoint2 = 1.0
            coeff1 = 2.0   # positive coefficient for x < -1
            coeff2 = -1.5  # negative coefficient for -1 <= x < 1
            coeff3 = 1.8   # positive coefficient for x >= 1

            y = (np.where(x[:,0] < breakpoint1,
                         coeff1 * (x[:,0] - breakpoint1),
                         np.where(x[:,0] < breakpoint2,
                                  coeff2 * (x[:,0] - breakpoint1),
                                  coeff3 * (x[:,0] - breakpoint2) + coeff2 * (breakpoint2 - breakpoint1)))
                 + 2 * (x[:,0] > -1) + (x[:,0] > 1) + eps)

        case 'UF':
            # Sine wave with three hills
            # This creates approximately 3 peaks in the range where most standard normal values lie
            frequency = 1  # controls number of hills
            amplitude = 2.0  # controls height
            y = amplitude * np.sin(frequency * np.pi * x[:,0]) + eps
        case 'UG':
            y = np.exp(x[:,0]) + eps
        case 'BA':
            y = x[:,0] + x[:,1] + eps
        case 'BB':
            y = x[:,0]**2 + x[:,1]**2 + eps
        case 'BC':
            y = x[:,0] + x[:,1] + x[:,0] * x[:,1] * eps
        case 'BD':
            y = x[:,0] + x[:,1] + 2 * x[:,0] * x[:,1] * eps
        case 'BE':
            y = x[:,0] + x[:,1] + 0.5 * x[:,0] * x[:,1] * eps
        case 'BF':
            y = x[:,0]**2 + x[:,1] + eps
        case 'BG':
            y = x[:,0]**2 + np.sin(x[:,1]) + eps
        case 'BH':
            y = x[:,0]**2 + np.log(np.abs(x[:,1])) + eps
        case 'BI':
            y = np.abs(x[:,0]) + np.exp(x[:,1]) + eps
        case 'BJ':
            y = x[:,0]**2 + x[:,1] + x[:,0] * x[:,1] + eps
        case 'CA':
            y = x[:,0] + x[:,1]**2 + np.exp(x[:,2]) + eps
        case 'MA':
            y = np.exp(0.2*x[:,0]) + 2*x[:,1] - x[:,2]**2 + eps  # + 0 * x[:,3]
        case 'MB':
            y = np.sqrt(np.abs(x[:,0]))  + 2*x[:,1] + np.log(7 + x[:,2]) + eps  # + 0 * x[:,3]
        case _:
            raise ValueError(f"Unknown DGP: {dgp_name}")

    data_dict = {f'X{i+1}': x[:,i] for i in range(d)}
    data_dict['y'] = y
    return pd.DataFrame(data_dict)

def estimate_coefs(X_train, X_test, y_train, y_test, n, dgp_name, param, method='LOESS',
                   bblr_outliers_method='up_to_N', N=10, eps_range=None,
                   bblr_regression_type='OLS', eps_multiplier=0.1, coef_comp='mean', ci=True,
                   pred_outliers_method=None, save_training=False, x_distribution='normal',
                   is_opt_eps=False):
    d = X_train.shape[1]
    if eps_range is None:
        eps_range = (0.1, 2 * d)
    # define column names
    columns_list = [f'X{i+1}' for i in range(d)]
    n_test = X_test.shape[0]
    if method == 'OLS':
        lr = sm.OLS(y_train, X_train).fit()
        betas = lr.params.to_numpy()
        coefs = pd.DataFrame({i:[betas[i]]*n_test for i in range(d)}).to_numpy()
        yp = lr.predict(X_test)
    if method == 'SVR':
        svr_best_model = svr_hypertune_and_predict(X_train, y_train, n_iter=20, random_state=42)
        yp = svr_best_model.predict(X_test)
        coefs = pd.DataFrame({i:[0]*n_test for i in range(d)}).to_numpy()  # dummy values to let the code work
    if method == 'LOESS':
        if param == 0:
            param = loess_optimal_frac(X_train, y_train, X_test, y_test, (0.01, 0.99))
        loess = LOESSRegressor(frac=param)
        loess.fit(X_train, y_train)
        yp, coefs, cis, std_errors = loess.predict(X_test, return_ci=ci)
        cis_array = np.array(cis)
        first_values = cis_array[:, :, 0]
        second_values = cis_array[:, :, 1]
        combined_data = np.concatenate([first_values, second_values], axis=1)
        if ci:
            cis = pd.DataFrame(combined_data)
            std_errors_df = pd.DataFrame(std_errors)
    if method == 'BBLR':
        # define BBLR model
        bblr = BMLRSNN(eps = param, column_names = columns_list, min_points=N)
        if param == 0:
            eps_range = eps_range
            bblr.find_optimal_radius(X_train, y_train, eps_range, scaling=False, maxiter=20, xatol=0.01,
                                     regression_type=bblr_regression_type, coef_comp=coef_comp,
                                     outliers_method=bblr_outliers_method, eps_multiplier=eps_multiplier)
            param = bblr.eps
            is_opt_eps = True
        bblr.predict_test_dataset(X_train, X_test, y_train, outliers_method=bblr_outliers_method,
                                        ci=ci,  regression_type=bblr_regression_type, scaling=False,
                                  eps_multiplier=eps_multiplier, coef_comp=coef_comp, save_training=save_training,
                                  DGP=dgp_name, x_distribution=x_distribution,
                                  pred_outliers_method=pred_outliers_method, is_opt_eps=is_opt_eps)
        yp = bblr.pred
        coefs = bblr.coefs.to_numpy()
        if ci:
            cis_raw = pd.DataFrame(bblr.conf_ints)
            cis_array = np.array(cis_raw)
            first_values = {f'L{i}': [ci[0] for ci in cis_array[:,i]] for i in range(d + 1)}
            second_values = {f'U{i}': [ci[1] for ci in cis_array[:,i]] for i in range(d + 1)}
            combined_data = np.concatenate([pd.DataFrame(first_values), pd.DataFrame(second_values)], axis=1)
            cis = pd.DataFrame(combined_data)
            std_errors_df = pd.DataFrame(bblr.std_errors_all)

    if ci:
        cis_col_names = ['L'+str(i) for i in range(d+1)] + ['U'+str(i) for i in range(d+1)]
        cis = cis.rename(columns={i:cis_col_names[i] for i in range((d+1)*2)})
        std_errors_df.rename(columns={i:'se'+str(i) for i in range(d+1)}, inplace=True)

    # Create B dictionary with proper reshaping
    B = {}
    for i in range(d):
        coef_col = coefs[:, i] if coefs.ndim > 1 else coefs
        # If it's 1D, reshape to match YP's format (n, 1) then flatten to (n,) for DataFrame
        if coef_col.ndim == 1:
            B[f'B{i+1}'] = coef_col  # Keep as 1D for DataFrame creation
        else:
            B[f'B{i+1}'] = coef_col.flatten()
    results_df = {
        # 'YP': yp,
        'YP': yp.flatten(),  # Flatten YP to match B's 1D format
        'GDP': dgp_name,
        'method': method,
        'parameter': param,
        'sample size': n
    }
    results_df.update(B)
    results_df = pd.concat([pd.DataFrame(results_df),
                            pd.DataFrame(X_test.values, columns=columns_list, index=X_test.index)], axis=1)
    if ci:
        results_df = pd.concat([results_df, cis, std_errors_df], axis=1)
    results_df['Y'] = np.array(y_test)
    if method == 'BBLR':
        results_df['pts_in_ball'] = np.array(bblr.n_points_in_ball_all)
    return pd.DataFrame(results_df)

def compute_true_coefs(X, result_df, dgp_name):
    X_size = len(X)
    match dgp_name:
        case 'UA':
            B_true = {'B1_true': np.ones(X_size)}
        case 'UB':
            B_true = {'B1_true': 2 * result_df['X1']}
        case 'UC':
            B_true = {'B1_true': np.zeros(X_size)}
        case 'UD':
            coeff_neg = -2.0
            coeff_pos = 1.5
            B_true = {'B1_true': np.where(result_df['X1'] < 0, coeff_neg, coeff_pos)}
        case 'UE':
            breakpoint1 = -1.0
            breakpoint2 = 1.0
            coeff1 = 2.0
            coeff2 = -1.5
            coeff3 = 1.8
            B_true = {'B1_true': np.where(result_df['X1'] < breakpoint1,
                                          coeff1,
                                          np.where(result_df['X1'] < breakpoint2,
                                                   coeff2,
                                                   coeff3))}
        case 'UF':
            frequency = 1
            amplitude = 2.0
            B_true = {'B1_true': amplitude * frequency * np.pi * np.cos(frequency * np.pi * result_df['X1'])}
        case 'UG':
            B_true = {'B1_true': np.exp(result_df['X1'])}
        case 'BA':
            B_true = {'B1_true': np.ones(X_size), 'B2_true': np.ones(X_size)}
        case 'BB':
            B_true = {'B1_true': 2 * result_df['X1'], 'B2_true': 2 * result_df['X2']}
        case 'BF':
            B_true = {'B1_true': 2 * result_df['X1'], 'B2_true': np.ones(X_size)}
        case 'BG':
            B_true = {'B1_true': 2 * result_df['X1'], 'B2_true': np.cos(result_df['X2'])}
        case 'BH':
            B_true = {'B1_true': 2 * result_df['X1'], 'B2_true': 1 / result_df['X2']}
        case 'BI':
            B_true = {'B1_true': np.sign(result_df['X1']), 'B2_true': np.exp(result_df['X2'])}
        case 'CA':
            B_true = {'B1_true': np.ones(X_size), 'B2_true': 2 * result_df['X2'], 'B3_true': np.exp(result_df['X3'])}
        case 'MA':
            B_true = {
                'B1_true': 0.2 * np.exp(0.2 * result_df['X1']),
                'B2_true': 2 * np.ones(X_size),
                'B3_true': -2 * result_df['X3'],
                'B4_true': np.zeros(X_size)
            }
        case 'MB':
            B_true = {
                'B1_true': np.where(result_df['X1'] > 0,
                                    1 / (2 * np.sqrt(np.abs(result_df['X1']))),
                                    -1 / (2 * np.sqrt(np.abs(result_df['X1'])))),
                'B2_true': 2 * np.ones(X_size),
                'B3_true': 1 / (result_df['X3'] + 7),
                'B4_true': np.zeros(X_size)
            }
        case _:
            raise ValueError(f"Unknown DGP: {dgp_name}")
    return B_true

def estimate_coefs_real(X, y, n, param, method='LOESS', bblr_outliers_method='up_to_N', N=10,
                        eps_range=None, bblr_regression_type='OLS', eps_multiplier=0.1, coef_comp='mean',
                        ci=True, pred_outliers_method=None, is_opt_eps=False):
    d = X.shape[1]
    if eps_range is None:
        eps_range = (0.1, 2 * d)
    # define column names
    columns_list = [f'X{i+1}' for i in range(d)]
    n_test = X.shape[0]
    if method == 'OLS':
        lr = sm.OLS(y, X).fit()
        betas = lr.params.to_numpy()
        coefs = pd.DataFrame({i:[betas[i]]*n_test for i in range(d)}).to_numpy()
        yp = lr.predict(X)
    if method == 'SVR':
        svr_best_model = svr_hypertune_and_predict(X, y, n_iter=20, random_state=42)
        yp = svr_best_model.predict(X)
        coefs = pd.DataFrame({i:[0]*n_test for i in range(d)}).to_numpy()  # dummy values to let the code work
    if method == 'LOESS':
        if param == 0:
            param = loess_optimal_frac(X, y, X, y, (0.01, 0.99))
        loess = LOESSRegressor(frac=param)
        loess.fit(X, y)
        yp, coefs, cis, std_errors = loess.predict(X, return_ci=ci)
        cis_array = np.array(cis)
        first_values = cis_array[:, :, 0]
        second_values = cis_array[:, :, 1]
        combined_data = np.concatenate([first_values, second_values], axis=1)
        if ci:
            cis = pd.DataFrame(combined_data)
            std_errors_df = pd.DataFrame(std_errors)
    if method == 'BBLR':
        # define BBLR model
        bblr = BMLRSNN(eps = param, column_names = columns_list, min_points=N)
        if param == 0:
            eps_range = eps_range
            bblr.find_optimal_radius(X, y, eps_range, scaling=False, maxiter=20, xatol=0.01,
                                     regression_type=bblr_regression_type, coef_comp=coef_comp,
                                     outliers_method=bblr_outliers_method, eps_multiplier=eps_multiplier)
            param = bblr.eps
            is_opt_eps = True
        bblr.predict_test_dataset(X, X, y, outliers_method=bblr_outliers_method,
                                        ci=ci,  regression_type=bblr_regression_type, scaling=False,
                                  eps_multiplier=eps_multiplier, coef_comp=coef_comp, save_training=False,
                                  DGP=dgp_name, x_distribution=x_distribution,
                                  pred_outliers_method=pred_outliers_method, is_opt_eps=is_opt_eps)
        yp = bblr.pred
        coefs = bblr.coefs.to_numpy()
        if ci:
            cis_raw = pd.DataFrame(bblr.conf_ints)
            cis_array = np.array(cis_raw)
            first_values = {f'L{i}': [ci[0] for ci in cis_array[:,i]] for i in range(d + 1)}
            second_values = {f'U{i}': [ci[1] for ci in cis_array[:,i]] for i in range(d + 1)}
            combined_data = np.concatenate([pd.DataFrame(first_values), pd.DataFrame(second_values)], axis=1)
            cis = pd.DataFrame(combined_data)
            std_errors_df = pd.DataFrame(bblr.std_errors_all)

    if ci:
        cis_col_names = ['L'+str(i) for i in range(d+1)] + ['U'+str(i) for i in range(d+1)]
        cis = cis.rename(columns={i:cis_col_names[i] for i in range((d+1)*2)})
        std_errors_df.rename(columns={i:'se'+str(i) for i in range(d+1)}, inplace=True)

    # Create B dictionary with proper reshaping
    B = {}
    for i in range(d):
        coef_col = coefs[:, i] if coefs.ndim > 1 else coefs
        # If it's 1D, reshape to match YP's format (n, 1) then flatten to (n,) for DataFrame
        if coef_col.ndim == 1:
            B[f'B{i+1}'] = coef_col  # Keep as 1D for DataFrame creation
        else:
            B[f'B{i+1}'] = coef_col.flatten()
    results_df = {
        # 'YP': yp,
        'YP': yp.flatten(),  # Flatten YP to match B's 1D format
        'GDP': dgp_name,
        'method': method,
        'parameter': param,
        'sample size': n
    }
    results_df.update(B)
    results_df = pd.concat([pd.DataFrame(results_df), pd.DataFrame(X_test, columns=columns_list)], axis=1)
    if ci:
        results_df = pd.concat([results_df, cis, std_errors_df], axis=1)
    results_df['Y'] = np.array(y_test)
    if method == 'BBLR':
        results_df['pts_in_ball'] = np.array(bblr.n_points_in_ball_all)
    return pd.DataFrame(results_df)







