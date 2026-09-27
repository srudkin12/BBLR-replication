import pandas as pd
import numpy as np

import sys

from sklearn.metrics import root_mean_squared_error

from scipy import stats
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from numpy.typing import NDArray
from statsmodels.stats.diagnostic import linear_reset, linear_rainbow, recursive_olsresiduals

import warnings
warnings.filterwarnings('ignore')

sys.path.append("../")
from bblr.bblr2 import *
from bblr.lowess import *
from bblr.lpr import *
from bblr.synthetic_data_generation import * 

def harvey_collier(
    res: RegressionResultsWrapper,
    order_by: NDArray[np.int_] | None = None,
) -> stats._stats_py.TtestResult:
    """Harvey--Collier linearity test, robust to the regressor count.

    The null hypothesis is that the regression is correctly modelled as
    linear. The statistic is a one-sample $t$-test that the recursive OLS
    residuals have mean zero; it has power against curvature only when the
    observations are ordered along a covariate that carries that curvature.

    Parameters
    ----------
    res : RegressionResultsWrapper
        A fitted OLS results instance.
    order_by : NDArray[np.int_] | None, optional
        Integer permutation ordering the observations (e.g. ``np.argsort``
        of a covariate). If ``None`` the current row order is used.

    Returns
    -------
    stats._stats_py.TtestResult
        Named tuple with ``statistic`` and ``pvalue`` fields.
    """
    skip = res.model.exog.shape[1]
    rr = recursive_olsresiduals(res, skip=skip, alpha=0.95, order_by=order_by)
    rresid = rr[3][skip:]
    rresid = rresid[~np.isnan(rresid)]
    return stats.ttest_1samp(rresid, 0)

def transform_coefs_array_to_dict(coefs, intercepts, d):
    # Create B dictionary with proper reshaping storing beta coefficients
    B = {}
    B[f'B{0}'] = intercepts
    for i in range(d):
        coef_col = coefs[:, i] if d > 1 else coefs
        B[f'B{i+1}'] = coef_col.flatten()
    return B

def transform_ci_dfs(cis_raw):
    cis_all = [np.append(cis_one[0, :], cis_one[1, :]) for cis_one in cis_raw]
    d = len(cis_raw[0][0, :]) - 1
    cis_col_names = ['L'+str(i) for i in range(d+1)] + ['U'+str(i) for i in range(d+1)]
    cis_all_df = pd.DataFrame(cis_all, columns=cis_col_names)
    return cis_all_df

def estimate_coefs(X_train, X_test, y_train, y_test, method, **kwargs):
    # d is a number of features in training set
    d = X_train.shape[1]
    ci = kwargs["ci"]
    columns_list = kwargs["column_names"]

    match method:
        case 'LR':
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            coefs = np.array([lr.coef_ for _ in range(X_test.shape[0])])
            intercepts = lr.intercept_
            param = None
            cis = None

            res = sm.OLS(y_train, sm.add_constant(X_train)).fit()
            hc_order_col = 0
            order = np.argsort(X_train[:, hc_order_col])
            reset_fitted = linear_reset(res, power=3, test_type="fitted", use_f=True)
            reset_exog = linear_reset(res, power=3, test_type="exog", use_f=True)
            hc = harvey_collier(res, order_by=order)
            rb_stat, rb_p = linear_rainbow(res)

        case 'BBLR':
            bblr = BallBasedLinearRegression(eps=kwargs["param"],
                                             column_names = kwargs["column_names"], 
                                             min_points = kwargs["min_pts"], 
                                             regression_type=kwargs["bblr_regression_type"],
                                             coef_comp=kwargs["coef_comp"], 
                                             outliers_method=kwargs["bblr_outliers_method"], 
                                             eps_multiplier=kwargs["eps_multiplier"],
                                             eps_range=kwargs["eps_range"], 
                                             ci=kwargs["ci"], 
                                             pred_outliers_method=kwargs["pred_outliers_method"], 
                                            #  **kwargs)
                                             confidence=kwargs["confidence"])
            bblr.fit(X_train, y_train)
            y_pred = bblr.predict(X_test)
            coefs = bblr.coefs
            intercepts = bblr.intercepts
            param = bblr.eps
            pts_in_balls = bblr.n_points_in_ball
            cis_raw = bblr.conf_ints
            if ci:
                std_errors_array = bblr.std_errors_all
                cis = transform_ci_dfs(cis_raw)

        case 'LOWESS':
            param = kwargs["param"]
            if param == 0:
                param = lowess_optimal_frac(X_train, y_train, (0.01, 0.99))
            lowess = LOWESSRegressor(frac=param)
            lowess.fit(X_train, y_train)
            y_pred = lowess.predict(X_test, return_ci=ci)
            y_pred, coefs, intercepts, cis_raw, std_errors = lowess.predict_all()
            # print(coefs)
            cis_array = np.array(cis_raw)
            first_values = cis_array[:, :, 0]
            second_values = cis_array[:, :, 1]
            combined_data = np.concatenate([first_values, second_values], axis=1)
            if ci:
                cis = pd.DataFrame(combined_data)
                std_errors_array = pd.DataFrame(std_errors)

        case 'LPR2':
            param = kwargs["param"]
            if param == 0:
                param = lpr_optimal_frac(X_train, y_train, (0.01, 0.99), degree=2)
            lpr = LocalPolynomialRegression(degree=2, frac=param, ci=False)
            lpr.fit(X_train, y_train)
            y_pred, coefs, intercepts, cis_raw, std_errors = lpr.predict(X_test)
            cis_array = np.array(cis_raw)
            first_values = cis_array[:, :, 0]
            second_values = cis_array[:, :, 1]
            combined_data = np.concatenate([first_values, second_values], axis=1)
            if ci:
                cis = pd.DataFrame(combined_data)
                std_errors_array = pd.DataFrame(std_errors)

        case 'LPR3':
            param = kwargs["param"]
            if param == 0:
                param = lpr_optimal_frac(X_train, y_train, X_test, y_test, (0.01, 0.99), degree=3)
            lpr = LocalPolynomialRegression(degree=3, frac=param, ci=False)
            lpr.fit(X_train, y_train)
            y_pred, coefs, intercepts, cis_raw, std_errors = lpr.predict(X_test)
            # print(coefs)
            cis_array = np.array(cis_raw)
            first_values = cis_array[:, :, 0]
            second_values = cis_array[:, :, 1]
            combined_data = np.concatenate([first_values, second_values], axis=1)
            if ci:
                cis = pd.DataFrame(combined_data)
                std_errors_array = pd.DataFrame(std_errors)

    # print(coefs)

    # Create B dictionary with proper reshaping storing beta coefficients
    B = transform_coefs_array_to_dict(coefs, intercepts, d)

    results_dict = {
            'YP': y_pred.flatten(),  # Flatten YP to match B's 1D format
            'method': method,
            'parameter': param
    }
    if method == 'BBLR':
        results_dict['pts_in_balls'] = pts_in_balls

    results_dict.update(B)
    results_df = pd.concat([pd.DataFrame(results_dict),
                            pd.DataFrame(X_test, columns=columns_list)], axis=1)

    if ci:
        std_errors_df = pd.DataFrame(std_errors_array, columns=['se'+str(i) for i in range(d+1)])
        # std_errors_df.columns = ['se'+str(i) for i in range(d+1)]
        results_df = pd.concat([results_df, cis, std_errors_df], axis=1)
    
    # add true values of y
    results_df['Y'] = np.array(y_test)

    if method == 'LR':
        lr_tests_dict = {"RESET_fitted_stat": float(reset_fitted.statistic),
                             "RESET_fitted_p": float(reset_fitted.pvalue),
                                "RESET_exog_stat": float(reset_exog.statistic),
                                "RESET_exog_p": float(reset_exog.pvalue),
                                "Harvey-Collier_stat": float(hc.statistic),
                                "Harvey-Collier_p": float(hc.pvalue),
                                "Rainbow_stat": float(rb_stat),
                                "Rainbow_p": float(rb_p)}
        return results_df, lr_tests_dict
    else:
        return results_df
        

def save_raw_results(results_df, n, method, dgp_name, x_distribution, train_test_flag, **kwargs):
    param = results_df["parameter"][0]
    # TODO: what is n? is it size of test dataset or whole dataset?
    if method == 'BBLR':
        bom = kwargs["bblr_outliers_method"]
        cc = kwargs["coef_comp"]
        brt = kwargs["bblr_regression_type"]
        if kwargs["ci"]:
            ci_flag = "ci"
        else:
            ci_flag = "noci"
        file_name = f"raw_results/{dgp_name}_{method}_{param}_{n}_{x_distribution}_{bom}_{brt}_{cc}_{ci_flag}_{train_test_flag}.csv"
    else:
        file_name = f"raw_results/{dgp_name}_{method}_{param}_{n}_{x_distribution}_{train_test_flag}.csv"
    
    results_df.to_csv(file_name)

def save_aggregated_results(results_df, n, method, dgp_name, x_distribution, train_test_flag, **kwargs):
    param = kwargs['param']
    if method == 'BBLR':
        bom = kwargs['bblr_outliers_method']
        cc = kwargs['coef_comp']
        brt = kwargs['bblr_regression_type']
        results_df.to_csv(f"aggregated_results/{dgp_name}_{method}_{param}_{n}_{x_distribution}_{bom}_{brt}_{cc}_{train_test_flag}.csv")
    else:
        results_df.to_csv(f"aggregated_results/{dgp_name}_{method}_{param}_{n}_{x_distribution}_{train_test_flag}.csv")
    return 0

def aggregate_raw_results(X_train, X_test, result_df, dgp_name, method, i, train_test_flag, a_vector=None, additional_columns=None):
    B_true = compute_true_coefs(X_test, result_df, dgp_name, a_vector=a_vector)

    d = X_test.shape[1]

    # Bias and RMSE calculations
    biases = {f'B{i}_bias': np.mean(result_df[f'B{i}'] - B_true[f'B{i}_true']) for i in range(0, d+1)}  # TODO: range(0, d+1)
    rmses = {f'B{i}_rmse': root_mean_squared_error(B_true[f'B{i}_true'], result_df[f'B{i}']) for i in range(0, d+1)}  # TODO: range(0, d+1)
    y_bias = np.mean(result_df['YP'] - result_df['Y'])
    y_rmse = root_mean_squared_error(result_df['YP'], result_df['Y'])

    results_rows = {
        'GDP': dgp_name,
        'method': method,
        'parameter': result_df['parameter'][0],
        'sample_size_train': X_train.shape[0],
        'sample_size_test': X_test.shape[0],
        'iter': i + 1,
        'y_bias': y_bias,
        'y_rmse': y_rmse,
        'train_test_flag': train_test_flag
    }
    results_rows.update(biases)
    results_rows.update(rmses)
    if method == 'LR':
        results_rows.update(additional_columns)

    return results_rows


