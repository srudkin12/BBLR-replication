import pandas as pd

import argparse
import sys
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split

sys.path.append("../")
from bblr.bblr2 import *
from bblr.simulations_functions import *

def run(args):
    M = args.M
    n = args.n
    dgp_name = args.dgp_name
    param = args.param
    method = args.method
    d = args.d
    noise_var = args.noise_var
    x_distribution = args.x_distribution
    min_pts = args.N
    eps_min = args.eps_min
    eps_max = args.eps_max
    eps_multiplier = args.eps_multiplier
    bblr_outliers_method = args.bblr_outliers_method
    coef_comp = args.coef_comp
    bblr_regression_type = args.bblr_regression_type
    bblr_outliers_method_pred = args.bblr_outliers_method_pred
    test_size=args.test_size
    include_training_data=args.include_training_data

# def monte_carlo_estimates(M, n, dgp_name, param, method, d, noise_var=0.2, min_pts=10, eps_range=(0.1,1.5), x_distribution='normal'):
    def one_iteration(i):
        # print(i)
        if dgp_name in ('UK', 'UL', 'UM', 'UN', 'UO', 'RA', 'RB', 'RC', 'RD'):
            df, a_vector = generate_data(n, dgp_name, d=d, noise_var=noise_var, seed=i, x_distribution=x_distribution)
        else:
            df = generate_data(n, dgp_name, d=d, noise_var=noise_var, seed=i, x_distribution=x_distribution)
            a_vector = None
        # define column names
        columns_list = [f'X{i+1}' for i in range(d)]
        if dgp_name == 'BK':
            columns_list += ['Z1', 'Z2']
        if dgp_name == 'MT':
            columns_list += ['Z1', 'Z2', 'Z3', 'Z4']
        X = df[columns_list]
        y = df['y']
        X = X.to_numpy()
        y = y.to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        if dgp_name == 'UL':
            # fit linear model and store its R2 score
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            r2_score = lr.score(X_test, y_test)

        if dgp_name == 'UJ':
            X_train = np.arange(0, 2, 4/n).reshape(-1, 1)
            X_test = X_train + 0.002
            y_train = X_train[:,0] ** (1/2) + np.random.normal(scale=noise_var, size=(X_train.shape[0], 1)).reshape(-1)
            y_test = X_test[:,0] ** (1/2) + np.random.normal(scale=noise_var, size=(X_test.shape[0], 1)).reshape(-1)

            print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        if method == 'LR':
            result_df, additional_columns = estimate_coefs(X_train, X_test, y_train, y_test, method,
                                    n=n, dgp_name=dgp_name, param=param, column_names=columns_list,  
                                    bblr_regression_type = bblr_regression_type, eps_multiplier = eps_multiplier,
                                    coef_comp = coef_comp, ci=False, pred_outliers_method=bblr_outliers_method_pred,
                                    min_pts = min_pts, eps_range=(eps_min, eps_max),
                                    bblr_outliers_method=bblr_outliers_method, confidence=0.95)
        else:
            additional_columns = None
            result_df = estimate_coefs(X_train, X_test, y_train, y_test, method,
                                    n=n, dgp_name=dgp_name, param=param, column_names=columns_list,  
                                    bblr_regression_type = bblr_regression_type, eps_multiplier = eps_multiplier,
                                    coef_comp = coef_comp, ci=False, pred_outliers_method=bblr_outliers_method_pred,
                                    min_pts = min_pts, eps_range=(eps_min, eps_max),
                                    bblr_outliers_method=bblr_outliers_method, confidence=0.95)
        if dgp_name == 'BK' or dgp_name == 'MT':  # use only if include_training_data=0 !
            columns_list = ['X1', 'X2']
            if dgp_name == 'MT':
                columns_list += ['X3', 'X4']
            X = df[columns_list]
            y = df['y']
            X = X.to_numpy()
            y = y.to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        results_rows = aggregate_raw_results(X_train, X_test, result_df, dgp_name, method, i, 'test', a_vector=a_vector, additional_columns=additional_columns)

        if include_training_data:
            # we set the optimal parameter to avoid repeating the process of searching for optimal parameter
            if param == 0:
                training_param = results_rows['parameter']
            else:
                training_param = param
            result_df_train = estimate_coefs(X_train, X_train, y_train, y_train, method,
                                        n=n, dgp_name=dgp_name, param=training_param, column_names=columns_list,  
                                        bblr_regression_type = bblr_regression_type, eps_multiplier = eps_multiplier,
                                        coef_comp = coef_comp, ci=False, pred_outliers_method=bblr_outliers_method_pred,
                                        min_pts = min_pts, eps_range=(eps_min, eps_max), 
                                        bblr_outliers_method=bblr_outliers_method, confidence=0.95)
            results_rows_train = aggregate_raw_results(X_train, X_train, result_df_train, dgp_name, method, i, 'train', a_vector=a_vector)
        else:
            results_rows_train = None

        if dgp_name in ('UK', 'UL', 'UM', 'UN', 'UO', 'RA', 'RB', 'RC', 'RD'):
            # if dgp_name == 'UK':
            #     return results_rows, results_rows_train, a_vector
            if dgp_name == 'UL':
                return results_rows, results_rows_train, a_vector, r2_score
            else:
                return results_rows, results_rows_train, a_vector
        else:
            return results_rows, results_rows_train

    # for debugging
    # for iter in range(M):
    #     one_iteration(iter)

    records = Parallel(n_jobs=-1)(delayed(one_iteration)(iter) for iter in range(M))

    if dgp_name in ('UK', 'UL', 'UM', 'UN', 'UO', 'RA', 'RB', 'RC', 'RD'):
        match dgp_name:
            case 'UK':
                results, results_train, a_vector = zip(*records)
                a_vectors = pd.DataFrame(a_vector, columns=['a0', 'a1', 'a2', 'a3'])
            case 'UL':
                results, results_train, a_vector, r2_score = zip(*records)
                a_vectors = pd.DataFrame(a_vector, columns=['a2'])
                r2_scores = pd.DataFrame(r2_score, columns=['r2_score'])
                r2_scores.to_csv(f"aggregated_results/{dgp_name}_r2_scores_{n}.csv")
            case 'UM':
                results, results_train, a_vector = zip(*records)
                a_vectors = pd.DataFrame(a_vector, columns=['aneg', 'apos'])
            case 'UN':
                results, results_train, a_vector = zip(*records)
                a_vectors = pd.DataFrame(a_vector, columns=['jump'])
            case 'UO':
                results, results_train, a_vector = zip(*records)
                a_vectors = pd.DataFrame(a_vector, columns=['aneg', 'apos', 'jump'])
            case 'RA':
                results, results_train, a_vector = zip(*records)
                a_vectors = pd.DataFrame(a_vector, columns=['a1', 'a2', 'a3'])
            case 'RB':
                results, results_train, a_vector = zip(*records)
                a_vectors = pd.DataFrame(a_vector, columns=['aneg1', 'apos1', 'aneg2', 'apos2', 'aneg3', 'apos3'])
            case 'RC':
                results, results_train, a_vector = zip(*records)
                a_vectors = pd.DataFrame(a_vector, columns=['jump1', 'jump2', 'jump3'])
            case 'RD':
                results, results_train, a_vector = zip(*records)
                a_vectors = pd.DataFrame(a_vector, columns=['aneg1', 'apos1', 'aneg2', 'apos2', 'aneg3', 'apos3', 'jump1', 'jump2', 'jump3'])
        a_vectors.to_csv(f"aggregated_results/{dgp_name}_a_vectors_{n}.csv")
    else:
        results, results_train = zip(*records)

    results = pd.DataFrame(results)
    results_train = pd.DataFrame(results_train)

    save_aggregated_results(results, n, method, dgp_name, x_distribution, 'test', param=param,
                            bblr_outliers_method=bblr_outliers_method,
                            coef_comp=coef_comp,
                            bblr_regression_type=bblr_regression_type)
    if include_training_data:
        save_aggregated_results(results_train, n, method, dgp_name, x_distribution, 'train', param=param,
                            bblr_outliers_method=bblr_outliers_method,
                            coef_comp=coef_comp,
                            bblr_regression_type=bblr_regression_type)

    return results, results_train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=1000, help="Number of repetitions")
    parser.add_argument("--n", type=int, default=1000, help="Sample sizes")
    parser.add_argument("--dgp_name", type=str, default='BB', help="Name of the DGP")
    parser.add_argument("--param", type=float, default=0, help="Parameter beta of the regression method")
    parser.add_argument("--method", type=str, default='BBLR', help="Name of the method")
    parser.add_argument("--d", type=int, default=2, help="Number of dimensions in the dataset")
    parser.add_argument("--noise_var", type=float, default=0, help="Variance of the noise")
    parser.add_argument("--x_distribution", type=str, default='normal', help="Name of the distribution of x")
    parser.add_argument("--N", type=int, default=10, help="Minimum number of points in a ball")
    parser.add_argument("--eps_multiplier", type=float, default=0.1,
                        help="Epsilon multiplier for epsilon_multiplied method of dealing with outliers")
    parser.add_argument("--eps_min", type=float, default=0.1,
                        help="Minimum of the range for searching the optimal value of the parameter")
    parser.add_argument("--eps_max", type=float, default=2.0,
                        help="Maximum of the range for searching the optimal value of the parameter")
    parser.add_argument("--bblr_outliers_method", type=str, default='up_to_N',
                        help="Method of dealing with outliers in BBLR regression")
    parser.add_argument("--coef_comp", type=str, default='mean',
                        help="Method of coefficient computing in BBLR regression")
    parser.add_argument("--bblr_regression_type", type=str, default='OLS',
                        help="Method of in-ball regression in BBLR regression")
    parser.add_argument("--bblr_outliers_method_pred", type=str, default='up_to_N',
                        help="Method of dealing with outliers in prediction in BBLR regression")
    parser.add_argument("--test_size", type=float, default=0.5,
                        help="Fraction of the dataset allocated to the test set")
    parser.add_argument("--include_training_data", type=int, default=0,
                        help="Flag deciding if the results of estimation for training data is saved")
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()



# example command to run:
# python simulations.py --M 10 --n 1000 --dgp_name 'BF' --param 0.7 --method 'BBLR' --d 2 --noise_var 0.2 --x_distribution 'normal' --N 10 --eps_multiplier 0.1 --eps_min 0.1 --eps_max 2.0 --bblr_outliers_method 'up_to_N' --coef_comp 'mean' --bblr_regression_type 'OLS' --bblr_outliers_method_pred 'up_to_N' --test_size 0.5 --include_training_data 0
# python simulations.py --M 10 --n 1000 --dgp_name 'UK' --param 0.0 --method 'BBLR' --d 1 --noise_var 0.5 --x_distribution 'normal' --N 10 --eps_multiplier 0.1 --eps_min 0.1 --eps_max 2.0 --bblr_outliers_method 'up_to_N' --coef_comp 'mean' --bblr_regression_type 'OLS' --bblr_outliers_method_pred 'up_to_N' --test_size 0.5 --include_training_data 0



