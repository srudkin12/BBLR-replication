import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
import argparse

import sys
import os
sys.path.append(os.path.abspath(".."))
from bblr.real_data_transform import prepare_and_split_data
from bblr.hyperparameters import optimize_ball_based_regression, optimize_lowess
from bblr.real_data_predictions import *

def run(args):
    dataset_name = args.dataset_name
    test_size = args.test_size
    d_add = args.d_add
    subset_size = args.subset_size
    bblr_regr_type = args.bblr_regr_type

    # dataset_name = "house_sales"
    # test_size = 0.2
    # d_add = 0
    # subset_size = 5000

    perform_regression_for_dataset(dataset_name, split_random_state=42, subset_size=subset_size, test_size=test_size,
                                   subset_random_state=42, number_of_noise_cols_added=d_add, bblr_regr_type=bblr_regr_type)
    
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='boston', help="Name of the data set")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size for train-test split")
    parser.add_argument("--d_add", type=int, default=0, help="Number of noise columns to add")
    parser.add_argument("--subset_size", type=int, default=None, help="Size of the subset of the data to use for regression")
    parser.add_argument("--bblr_regr_type", type=str, default='OLS', help="Type of regression to perform in the balls in BBLR regression ('OLS' or 'WLS')")
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()


# example:
# python py_real.py --dataset_name 'boston' --test_size 0.2 --d_add 0
# python py_real.py --dataset_name 'house_sales' --test_size 0.2 --d_add 0 --subset_size 5000


