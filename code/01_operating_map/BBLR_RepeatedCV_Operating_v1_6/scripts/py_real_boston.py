import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
import argparse

import sys
import os
sys.path.append(os.path.abspath(".."))
from bblr.real_data_transform import prepare_and_split_data
from bblr.hyperparameters import optimize_ball_based_regression, optimize_lowess

def run(args):
    # Data preparation

    dataset_name = "boston"
    file_path = "automl_datasets/boston.csv"
    cat_features = []  # Assuming no categorical features in the Boston dataset
    num_features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    target = 'MEDV'
    test_size = 0.2

    X_train, X_test, y_train, y_test = prepare_and_split_data(file_path, cat_features, num_features, target, test_size, split_random_state=42)


    # BBLR

    best_est_bblr, best_par_bblr = optimize_ball_based_regression(X_train, y_train, eps_range=np.arange(0.3, 0.8, 0.025), 
                                                        min_points_range=[40, 50, 60, 70, 80, 90])

    bblr_best = best_est_bblr
    bblr_best.fit(X_train, y_train)
    y_pred_best = bblr_best.predict(X_test)
    rmse_best = root_mean_squared_error(y_test, y_pred_best)
    # print(f"Default BBLR RMSE: {rmse_best}")
    bblr_prediction_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred_best
    })
    bblr_prediction_df.to_csv(f"automl_preds/bblr_predictions_{dataset_name}.csv", index=False)
    # save rmse_best to a text file
    with open(f"automl_preds/bblr_rmse_{dataset_name}.txt", "w") as f:
        f.write(f"{rmse_best}")
    # save best parameters to a text file
    with open(f"automl_preds/bblr_best_params_{dataset_name}.txt", "w") as f:
        f.write(f"{best_par_bblr}")
    bblr_best.coefs.to_csv(f"automl_preds/bblr_coefs_{dataset_name}.csv", index=False)
    bblr_best.intercepts.to_csv(f"automl_preds/bblr_intercepts_{dataset_name}.csv", index=False)

    # LOWESS

    best_est_lowess, best_par_lowess = optimize_lowess(X_train, y_train, frac_range=np.arange(0.25, 0.35, 0.01))

    y_pred_lowess = best_est_lowess.predict(X_test, return_ci=False)
    rmse_lowess = root_mean_squared_error(y_test, y_pred_lowess)
    # print(f"Default LOWESS RMSE: {rmse_lowess}")

    lowess_prediction_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred_lowess
    })
    lowess_prediction_df.to_csv(f"automl_preds/lowess_predictions_{dataset_name}.csv", index=False)
    # save rmse_best to a text file
    with open(f"automl_preds/lowess_rmse_{dataset_name}.txt", "w") as f:
        f.write(f"{rmse_lowess}")
    # save best parameters to a text file
    with open(f"automl_preds/lowess_best_params_{dataset_name}.txt", "w") as f:
        f.write(f"{best_par_lowess}")
    best_est_lowess.coefs.to_csv(f"automl_preds/lowess_coefs_{dataset_name}.csv", index=False)
    best_est_lowess.intercepts.to_csv(f"automl_preds/lowess_intercepts_{dataset_name}.csv", index=False)

    return 0

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()

