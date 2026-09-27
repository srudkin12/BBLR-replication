#!/bin/bash
#SBATCH --job-name=bblr_test
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=48
#SBATCH --time=23:00:00
#SBATCH --output=output%j.log

module load gnu12
module load openmpi4

source ~/.bashrc

micromamba activate mpi_env

python3 simulations.py --M 1000 --n 1000 --dgp_name 'BA' --param 0 --method 'BBLR' --d 2 --noise_var 0.5 --x_distribution 'normal' --N 10 --eps_multiplier 0.1 --eps_min 0.1 --eps_max 2.0 --bblr_outliers_method 'up_to_N' --coef_comp 'mean' --bblr_regression_type 'OLS' --bblr_outliers_method_pred 'up_to_N' --test_size 0.5 --include_training_data 1

python3 simulations.py --M 1000 --n 1000 --dgp_name 'BA' --param 0 --method 'LOWESS' --d 2 --noise_var 0.5 --x_distribution 'normal' --N 10 --eps_multiplier 0.1 --eps_min 0.1 --eps_max 2.0 --bblr_outliers_method 'up_to_N' --coef_comp 'mean' --bblr_regression_type 'OLS' --bblr_outliers_method_pred 'up_to_N' --test_size 0.5 --include_training_data 1

python3 simulations.py --M 1000 --n 1000 --dgp_name 'BB' --param 0 --method 'BBLR' --d 2 --noise_var 0.5 --x_distribution 'normal' --N 10 --eps_multiplier 0.1 --eps_min 0.1 --eps_max 2.0 --bblr_outliers_method 'up_to_N' --coef_comp 'mean' --bblr_regression_type 'OLS' --bblr_outliers_method_pred 'up_to_N' --test_size 0.5 --include_training_data 1

python3 simulations.py --M 1000 --n 1000 --dgp_name 'BB' --param 0 --method 'LOWESS' --d 2 --noise_var 0.5 --x_distribution 'normal' --N 10 --eps_multiplier 0.1 --eps_min 0.1 --eps_max 2.0 --bblr_outliers_method 'up_to_N' --coef_comp 'mean' --bblr_regression_type 'OLS' --bblr_outliers_method_pred 'up_to_N' --test_size 0.5 --include_training_data 1

python3 simulations.py --M 1000 --n 1000 --dgp_name 'BF' --param 0 --method 'BBLR' --d 2 --noise_var 0.5 --x_distribution 'normal' --N 10 --eps_multiplier 0.1 --eps_min 0.1 --eps_max 2.0 --bblr_outliers_method 'up_to_N' --coef_comp 'mean' --bblr_regression_type 'OLS' --bblr_outliers_method_pred 'up_to_N' --test_size 0.5 --include_training_data 1

python3 simulations.py --M 1000 --n 1000 --dgp_name 'BF' --param 0 --method 'LOWESS' --d 2 --noise_var 0.5 --x_distribution 'normal' --N 10 --eps_multiplier 0.1 --eps_min 0.1 --eps_max 2.0 --bblr_outliers_method 'up_to_N' --coef_comp 'mean' --bblr_regression_type 'OLS' --bblr_outliers_method_pred 'up_to_N' --test_size 0.5 --include_training_data 1

python3 simulations.py --M 1000 --n 1000 --dgp_name 'UA' --param 0 --method 'BBLR' --d 1 --noise_var 0.5 --x_distribution 'normal' --N 10 --eps_multiplier 0.1 --eps_min 0.1 --eps_max 2.0 --bblr_outliers_method 'up_to_N' --coef_comp 'mean' --bblr_regression_type 'OLS' --bblr_outliers_method_pred 'up_to_N' --test_size 0.5 --include_training_data 1

python3 simulations.py --M 1000 --n 1000 --dgp_name 'UA' --param 0 --method 'LOWESS' --d 1 --noise_var 0.5 --x_distribution 'normal' --N 10 --eps_multiplier 0.1 --eps_min 0.1 --eps_max 2.0 --bblr_outliers_method 'up_to_N' --coef_comp 'mean' --bblr_regression_type 'OLS' --bblr_outliers_method_pred 'up_to_N' --test_size 0.5 --include_training_data 1

python3 simulations.py --M 1000 --n 1000 --dgp_name 'UB' --param 0 --method 'BBLR' --d 1 --noise_var 0.5 --x_distribution 'normal' --N 10 --eps_multiplier 0.1 --eps_min 0.1 --eps_max 2.0 --bblr_outliers_method 'up_to_N' --coef_comp 'mean' --bblr_regression_type 'OLS' --bblr_outliers_method_pred 'up_to_N' --test_size 0.5 --include_training_data 1

python3 simulations.py --M 1000 --n 1000 --dgp_name 'UB' --param 0 --method 'LOWESS' --d 1 --noise_var 0.5 --x_distribution 'normal' --N 10 --eps_multiplier 0.1 --eps_min 0.1 --eps_max 2.0 --bblr_outliers_method 'up_to_N' --coef_comp 'mean' --bblr_regression_type 'OLS' --bblr_outliers_method_pred 'up_to_N' --test_size 0.5 --include_training_data 1

python3 simulations.py --M 1000 --n 1000 --dgp_name 'UE' --param 0 --method 'BBLR' --d 1 --noise_var 0.5 --x_distribution 'normal' --N 10 --eps_multiplier 0.1 --eps_min 0.1 --eps_max 2.0 --bblr_outliers_method 'up_to_N' --coef_comp 'mean' --bblr_regression_type 'OLS' --bblr_outliers_method_pred 'up_to_N' --test_size 0.5 --include_training_data 1

python3 simulations.py --M 1000 --n 1000 --dgp_name 'UE' --param 0 --method 'LOWESS' --d 1 --noise_var 0.5 --x_distribution 'normal' --N 10 --eps_multiplier 0.1 --eps_min 0.1 --eps_max 2.0 --bblr_outliers_method 'up_to_N' --coef_comp 'mean' --bblr_regression_type 'OLS' --bblr_outliers_method_pred 'up_to_N' --test_size 0.5 --include_training_data 1

python3 simulations.py --M 1000 --n 1000 --dgp_name 'UF' --param 0 --method 'BBLR' --d 1 --noise_var 0.5 --x_distribution 'normal' --N 10 --eps_multiplier 0.1 --eps_min 0.1 --eps_max 2.0 --bblr_outliers_method 'up_to_N' --coef_comp 'mean' --bblr_regression_type 'OLS' --bblr_outliers_method_pred 'up_to_N' --test_size 0.5 --include_training_data 1

python3 simulations.py --M 1000 --n 1000 --dgp_name 'UF' --param 0 --method 'LOWESS' --d 1 --noise_var 0.5 --x_distribution 'normal' --N 10 --eps_multiplier 0.1 --eps_min 0.1 --eps_max 2.0 --bblr_outliers_method 'up_to_N' --coef_comp 'mean' --bblr_regression_type 'OLS' --bblr_outliers_method_pred 'up_to_N' --test_size 0.5 --include_training_data 1

python3 simulations.py --M 1000 --n 1000 --dgp_name 'UH' --param 0 --method 'BBLR' --d 1 --noise_var 0.5 --x_distribution 'normal' --N 10 --eps_multiplier 0.1 --eps_min 0.1 --eps_max 2.0 --bblr_outliers_method 'up_to_N' --coef_comp 'mean' --bblr_regression_type 'OLS' --bblr_outliers_method_pred 'up_to_N' --test_size 0.5 --include_training_data 1

python3 simulations.py --M 1000 --n 1000 --dgp_name 'UH' --param 0 --method 'LOWESS' --d 1 --noise_var 0.5 --x_distribution 'normal' --N 10 --eps_multiplier 0.1 --eps_min 0.1 --eps_max 2.0 --bblr_outliers_method 'up_to_N' --coef_comp 'mean' --bblr_regression_type 'OLS' --bblr_outliers_method_pred 'up_to_N' --test_size 0.5 --include_training_data 1

