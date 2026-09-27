import pandas as pd
import numpy as np

import argparse
import sys

from sklearn.metrics import make_scorer, root_mean_squared_error, mean_squared_error
from sklearn.model_selection import train_test_split

import time

import warnings
warnings.filterwarnings('ignore')

sys.path.append("../")
from BMR.bmlr import *
from BMR.simulations_functions import *

def run(args):
    n = args.n
    dgp_name = args.dgp_name
    param = args.param
    method = args.method
    d = args.d
    noise_var = args.noise_var
    x_distribution = args.x_distribution
    min_pts = args.N
    # eps_range = args.eps_range
    eps_range = (0.1, 2 * d)
    eps_multiplier = args.eps_multiplier
    bblr_outliers_method = args.bblr_outliers_method
    coef_comp = args.coef_comp
    bblr_regression_type = args.bblr_regression_type
    seed = args.seed
    ci = bool(args.ci)
    pred_outliers_method = None
    test_size_prop = 0.5  # TODO: make this an argument
    k = 10  # TODO: make this an argument

    bom = bblr_outliers_method
    cc = coef_comp
    brt = bblr_regression_type

    def one_iteration(i):
        df = generate_data(n, dgp_name, d=d, noise_var=noise_var, seed=i, x_distribution=x_distribution)
        # define column names
        columns_list = [f'X{i+1}' for i in range(d)]
        X = df[columns_list]
        y = df['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_prop, random_state=42)
        X_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)
        result_df = estimate_coefs(X_train, X_test, y_train, y_test, n, dgp_name, param,
                                   method, bblr_outliers_method=bblr_outliers_method, N=min_pts, eps_range=eps_range,
                                   bblr_regression_type = bblr_regression_type, eps_multiplier = eps_multiplier,
                                   coef_comp = coef_comp, ci=ci, pred_outliers_method=pred_outliers_method,
                                   save_training=True, x_distribution=x_distribution, is_opt_eps=True)

        if method == 'LOESS':
            opt_eps = result_df['parameter'][0]
            result_df_train = estimate_coefs(X_train, X_train, y_train, y_train, n, dgp_name, opt_eps,
                                   method, bblr_outliers_method=bblr_outliers_method, N=min_pts, eps_range=eps_range,
                                   bblr_regression_type = bblr_regression_type, eps_multiplier = eps_multiplier,
                                   coef_comp = coef_comp, ci=ci, x_distribution=x_distribution)

        if method == 'BBLR':
            result_df.to_csv(
                f"raw_results/{dgp_name}_{method}_{param}_{n}_{x_distribution}_{bom}_{brt}_{cc}_{ci}.csv")
            # result_df_train.to_csv(
            #     f"raw_results/{dgp_name}_{method}_{param}_{n}_{x_distribution}_{bom}_{brt}_{cc}_train.csv")
        else:
            result_df.to_csv(f"raw_results/{dgp_name}_{method}_{param}_{n}_{x_distribution}_{ci}.csv")
            result_df_train.to_csv(f"raw_results/{dgp_name}_{method}_{param}_{n}_{x_distribution}_{ci}_train.csv")

        if method == 'BBLR':
            result_df_train = pd.read_csv(
                f"raw_results/{dgp_name}_{method}_{param}_{n}_{x_distribution}_{bom}_{brt}_{cc}_{ci}_train.csv")


        # True coefficients (for simplicity, assume 1 for linear case unless calculated)
        # X_test_df = pd.DataFrame(X_test)
        n_test = len(X_test)
        n_train = len(X_train)

        B_true = compute_true_coefs(X_test, result_df, dgp_name)
        B_true['B0_true'] = np.zeros(n_test)
        B_true_df = pd.DataFrame(B_true)
        B_true_df.rename(columns={col_name:col_name[:-5] for col_name in B_true_df.columns}, inplace=True)
        B_true_df = pd.concat([B_true_df, X_test], axis=1)
        B_true_df.to_csv(f"raw_results/true_{dgp_name}_{n}_{x_distribution}.csv")

        # if method == 'LOESS':
        B_true_train = compute_true_coefs(X_train, result_df_train, dgp_name)
        B_true_train['B0_true'] = np.zeros(n_train)
        B_true_df_train = pd.DataFrame(B_true_train)
        B_true_df_train.rename(columns={col_name:col_name[:-5] for col_name in B_true_df_train.columns}, inplace=True)
        B_true_df_train = pd.concat([B_true_df_train, X_train], axis=1)
        B_true_df_train.to_csv(f"raw_results/true_{dgp_name}_{n}_{x_distribution}_train.csv")

        # Bias and RMSE calculations
        biases = {f'B{i}_bias': np.mean(result_df[f'B{i}'] - B_true[f'B{i}_true']) for i in range(1, d+1)}
        rmses = {f'B{i}_rmse': root_mean_squared_error(B_true[f'B{i}_true'], result_df[f'B{i}']) for i in range(1, d+1)}
        y_bias = np.mean(result_df['YP'] - result_df['Y'])
        y_rmse = root_mean_squared_error(result_df['YP'], result_df['Y'])

        results_rows = {
            'GDP': dgp_name,
            'method': method,
            'parameter': result_df['parameter'][0],
            'sample size': n,
            'seed': i,
            'y_bias': y_bias,
            'y_rmse': y_rmse
        }
        results_rows.update(biases)
        results_rows.update(rmses)

        # if method == 'LOESS':
        # Bias and RMSE calculations
        biases_train = {f'B{i}_bias': np.mean(result_df_train[f'B{i}'] - B_true_train[f'B{i}_true']) for i in range(1, d+1)}
        rmses_train = {f'B{i}_rmse': root_mean_squared_error(B_true_train[f'B{i}_true'], result_df_train[f'B{i}']) for i in range(1, d+1)}
        y_bias_train = np.mean(result_df_train['YP'] - result_df_train['Y'])
        y_rmse_train = root_mean_squared_error(result_df_train['YP'], result_df_train['Y'])

        results_rows_train = {
            'GDP': dgp_name,
            'method': method,
            'parameter': result_df_train['parameter'][0],
            'sample size': n,
            'seed': i,
            'y_bias': y_bias_train,
            'y_rmse': y_rmse_train
        }
        results_rows_train.update(biases_train)
        results_rows_train.update(rmses_train)
        # else:
        #     results_rows_train = None

        return results_rows, results_rows_train

    results_rows, results_rows_train = one_iteration(seed)
    aggregated_results_df = pd.DataFrame([results_rows])
    aggregated_results_df_train = pd.DataFrame([results_rows_train])

    if method == 'BBLR':
        aggregated_results_df.to_csv(f"raw_results/agg_{dgp_name}_{method}_{param}_{n}_{x_distribution}_{bom}_{brt}_{cc}_{ci}.csv")
        aggregated_results_df_train.to_csv(
            f"raw_results/agg_{dgp_name}_{method}_{param}_{n}_{x_distribution}_{bom}_{brt}_{cc}_{ci}_train.csv")
    else:
        aggregated_results_df.to_csv(f"raw_results/agg_{dgp_name}_{method}_{param}_{n}_{x_distribution}_{ci}.csv")
        aggregated_results_df_train.to_csv(f"raw_results/agg_{dgp_name}_{method}_{param}_{n}_{x_distribution}_{ci}_train.csv")

    return aggregated_results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="Sample sizes")
    parser.add_argument("--dgp_name", type=str, default='BB', help="Name of the DGP")
    parser.add_argument("--param", type=float, default=0, help="Parameter beta of the regression method")
    parser.add_argument("--method", type=str, default='BBLR', help="Name of the method")
    parser.add_argument("--d", type=int, default=2, help="Number of dimensions in the dataset")
    parser.add_argument("--noise_var", type=float, default=0, help="Variance of the noise")
    parser.add_argument("--x_distribution", type=str, default='normal', help="Name of the distribution of x")
    parser.add_argument("--eps_multiplier", type=float, default=0.1,
                        help="Epsilon multiplier for epsilon_multiplied method of dealing with outliers")
    parser.add_argument("--N", type=int, default=10, help="Minimum number of points in a ball")
    parser.add_argument("--bblr_outliers_method", type=str, default='up_to_N',
                        help="Method of dealing with outliers in BBLR regression")
    parser.add_argument("--bblr_regression_type", type=str, default='OLS',
                        help="Method of in-ball regression in BBLR regression")
    parser.add_argument("--coef_comp", type=str, default='mean',
                        help="Method of coefficient computing in BBLR regression")
    parser.add_argument("--seed", type=int, default=42,
                        help="Method of coefficient computing in BBLR regression")
    parser.add_argument("--ci", type=int, default=0,
                        help="Confidence interval flag: 0 or 1")
    parser.add_argument("--pred_outliers_method", type=str, default=None,
                        help="Method of dealing with outliers in prediction of BBLR regression")
    args = parser.parse_args()
    t1 = time.time()
    run(args)
    t2 = time.time()
    print(t2 - t1)

if __name__ == "__main__":
    main()






