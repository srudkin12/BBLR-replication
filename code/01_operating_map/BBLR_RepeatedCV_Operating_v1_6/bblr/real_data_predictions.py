import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error

import sys
import os
sys.path.append(os.path.abspath(".."))
from bblr.real_data_transform import prepare_and_split_data
from bblr.hyperparameters import optimize_ball_based_regression, optimize_lowess


datasets_parameters = {
    "abalone": {
        'dataset_name':"abalone",
        'file_path':"automl_datasets/abalone.csv",
        'cat_features':['Sex'],
        'num_features':['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight'],
        'target':'Class_number_of_rings',
        # 'eps_range':np.arange(0.3, 0.9, 0.05),
        # 'r_range':np.arange(0.2, 0.9, 0.05),
        # 'min_points_range':[40, 50, 60, 70, 80],
        'eps_range':np.arange(0.7, 1.6, 0.05),
        'r_range':np.arange(0.5, 1.2, 0.05),
        'min_points_range':[40, 60, 60, 100, 150],
        'frac_range':np.arange(0.1, 0.4, 0.01),
    },
    'AirfoilSelfNoise': {
        'dataset_name':"AirfoilSelfNoise",
        'file_path':"automl_datasets/AirfoilSelfNoise.csv",
        'cat_features':[],
        'num_features':['Frequency', 'Angle_of_attack', 'Chord_length', 'Free_stream_velocity', 'Suction_side_displacement_thickness'],
        'target':'Scaled_sound_pressure_level',
        # 'eps_range':np.arange(0.8, 1.3, 0.025),
        # 'r_range':np.arange(0.5, 1.2, 0.05),
        # 'min_points_range':[8, 10, 15, 20],
        'eps_range':np.arange(0.6, 0.8, 0.05),
        'r_range':np.arange(1.0, 1.3, 0.05),
        'min_points_range':[8, 10, 12, 15],
        'frac_range':np.arange(0.1, 0.4, 0.01),
        # f,"alpha","c","U_infinity","delta","SSPL"
    },
    "AmesHousing": {
        'dataset_name': "AmesHousing",
        'file_path': "automl_datasets/AmesHousing.csv",
        'cat_features': [
            'Neighborhood',   # Lokalizacja
            'KitchenQual',    # Jakość kuchni (kategoryczna/porządkowa)
            'ExterQual'       # Jakość materiałów zewnętrznych
        ],
        'num_features': [
            'OverallQual',    # Ogólna jakość (często traktowana jako numeryczna w modelach)
            'GrLivArea',       # Powierzchnia mieszkalna nad ziemią
            'TotalBsmtSF',     # Całkowita powierzchnia piwnicy
            'GarageCars',      # Pojemność garażu
            '1stFlrSF',        # Powierzchnia 1. piętra
            'FullBath',        # Liczba pełnych łazienek
            'YearBuilt',       # Rok budowy
            'YearRemodAdd'     # Rok renowacji
        ],
        'target': 'SalePrice',
        'eps_range':np.arange(0.4, 1.2, 0.05),
        'r_range':np.arange(0.5, 1.2, 0.05),
        'min_points_range':[40, 50, 60, 70, 80, 90],
        'frac_range':np.arange(0.05, 0.3, 0.01),
    },
    'auto_mpg': {
        'dataset_name':"auto_mpg",
        'file_path':"automl_datasets/auto_mpg.csv",
        'cat_features':['origin'],  
        'num_features':['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year'],
        'target':'mpg',
        'eps_range':np.arange(0.2, 1.1, 0.025),
        'r_range':np.arange(0.2, 1.1, 0.025),
        'min_points_range':[10, 15, 20, 30],
        'frac_range':np.arange(0.1, 0.4, 0.01),
    },
    "boston": {
        'dataset_name':"boston",
        'file_path':"automl_datasets/boston.csv",
        'cat_features':[],  # Assuming no categorical features in the Boston dataset
        'num_features':['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
        'target':'MEDV',
        'eps_range':np.arange(0.05, 0.5, 0.025),
        'r_range':np.arange(0.05, 0.5, 0.025),
        'min_points_range':[15, 20, 30, 40, 50],
        'frac_range':np.arange(0.25, 0.35, 0.01),
    },
    "Brazilian_houses":{
        'dataset_name':"Brazilian_houses",
        'file_path':"automl_datasets/Brazilian_houses.csv",
        'cat_features':['city', 'animal', 'furniture'],
        'num_features':['area', 'rooms', 'bathroom', 'parking_spaces', 'floor'],
        'target':'total_(BRL)',
        'drop_features':['hoa_(BRL)', 'rent_amount_(BRL)', 'property_tax_(BRL)', 'fire_insurance_(BRL)'],
        'eps_range':np.arange(0.4, 1.1, 0.05),
        'r_range':np.arange(0.3, 0.8, 0.025),
        'min_points_range':[20, 40, 60, 80, 100, 150, 200],
        'frac_range':np.arange(0.05, 0.4, 0.01),
    },
    'brexit': {
        'dataset_name':"brexit",
        'file_path':"automl_datasets/brexit.csv",
        'cat_features':[],
        'num_features':[
            "c11HouseOutright",
            "c11HouseMortgage",
            "c11HouseholdOnePerson",
            "c11HouseholdMarried",
            "c11CarsNone",
            "c11CarsOne",
            "c11CarsTwo",
            "c11NSSECLowerManager",
            "c11QualNone",
            "c11QualLevel4",
            "c11HealthVeryGood",
            "c11HealthGood",
            "c11DeprivedNone",
            "c11Deprived1",
        ],
        'target':'leaveHanretty',
        'eps_range':np.arange(2.2, 3.0, 0.05),
        'r_range':np.arange(2.8, 3.3, 0.05),
        'min_points_range':[20, 25, 30, 35, 40],
        # 'eps_range':np.arange(1.7, 4.0, 0.05),
        # 'r_range':np.arange(1.0, 3.0, 0.05),
        # 'min_points_range':[10, 15, 20, 30, 40],
        # 'eps_range':np.arange(2.0, 4.0, 0.5),
        # 'r_range':np.arange(1.0, 3.0, 0.5),
        # 'min_points_range':[30],
        'frac_range':np.arange(0.05, 0.3, 0.01),
    },
    'california': {
        # longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income,median_house_value,ocean_proximity
        'dataset_name':"california",
        'file_path':"automl_datasets/california.csv",
        'cat_features':['ocean_proximity'],  
        # 'num_features':['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'],
        'num_features':['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income'],
        'target':'median_house_value',
        'eps_range':np.arange(2.0, 3.0, 0.1),
        'r_range':np.arange(1.5, 2.5, 0.1),
        'min_points_range':[80, 100, 120],
        'frac_range':np.arange(0.05, 0.3, 0.01),
    },
    'cbm': {
        'dataset_name':"cbm",
        'file_path':"automl_datasets/cbm.csv",
        'cat_features':[],
        'num_features':['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16'],
        'target':'y',
        'eps_range':np.arange(2.0, 3.0, 0.1),
        'r_range':np.arange(2.0, 3.0, 0.1),
        'min_points_range':[30, 40, 50, 60, 80],
        'frac_range':np.arange(0.05, 0.3, 0.01),
    },
    'ccpp': {
        'dataset_name':"ccpp",
        'file_path':"automl_datasets/ccpp.csv",
        'cat_features':[],
        'num_features':['AT', 'V', 'AP', 'RH'],
        'target':'PE',
        'eps_range':np.arange(0.5, 1.5, 0.1),
        'r_range':np.arange(0.5, 1.5, 0.1),
        'min_points_range':[40, 60, 80, 100],
        'frac_range':np.arange(0.05, 0.3, 0.01),
    },
    'concrete': {
        'dataset_name':"concrete",
        'file_path':"automl_datasets/concrete.csv",
        'cat_features':[],
        # 'num_features':['cement', 'slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age'],
        # Cement,Blast Furnace Slag,Fly Ash,Water,Superplasticizer,Coarse Aggregate,Fine Aggregate,Age,Strength
        'num_features': ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age'],
        'target':'Strength',
        'eps_range':np.arange(0.004, 0.012, 0.002),
        'r_range':np.arange(0.004, 0.012, 0.002),
        'min_points_range':[40, 50, 60],
        # 'eps_range':np.arange(0.01, 0.1, 0.01),
        # 'r_range':np.arange(0.01, 0.1, 0.01),
        # 'min_points_range':[40, 50, 60, 80, 100],
        'frac_range':np.arange(0.05, 0.3, 0.01),
    },
    'diamonds': {
        'dataset_name':"diamonds",
        'file_path':"automl_datasets/diamonds.csv",
        'cat_features':['cut', 'color', 'clarity'],
        'num_features':['carat', 'depth', 'table', 'x', 'y', 'z'],
        'target':'price',
        'eps_range':np.arange(0.06, 0.12, 0.05),
        'r_range':np.arange(0.06, 0.12, 0.05),
        'min_points_range':[80, 100, 120, 150],
        # 'eps_range':np.arange(0.1, 0.8, 0.05),
        # 'r_range':np.arange(0.1, 0.7, 0.05),
        # 'min_points_range':[20, 40, 60, 80, 100],
        'frac_range':np.arange(0.05, 0.3, 0.01),
    },
    "elevators": {
        'dataset_name':"elevators", 
        'file_path':"automl_datasets/elevators.csv",
        'cat_features':[],
        'num_features':['climbRate','Sgz','p','q','curRoll','absRoll','diffClb','diffRollRate','diffDiffClb','SaTime1',
                        'SaTime2','SaTime3','SaTime4','diffSaTime1','diffSaTime2','diffSaTime3','diffSaTime4','Sa'],
        'target':'Goal',
        'eps_range':np.arange(0.06, 0.12, 0.05),
        'r_range':np.arange(0.06, 0.12, 0.05),
        'min_points_range':[80, 100, 120, 150],
        # 'eps_range':np.arange(0.1, 0.35, 0.025),
        # 'r_range':np.arange(0.1, 0.4, 0.025),
        # 'min_points_range':[20, 40, 60, 80, 100],
        'frac_range':np.arange(0.05, 0.4, 0.01),
    },
    "energy1": {
        'dataset_name':"energy1", 
        'file_path':"automl_datasets/energy1.csv",
        'cat_features':[],
        'num_features':['X1','X2','X3','X4','X5','X6','X7','X8'],
        'target':'Y1',
        'eps_range':np.arange(0.03, 0.08, 0.01),
        'r_range':np.arange(0.03, 0.08, 0.01),
        'min_points_range':[35, 40, 45, 50, 55],
        'frac_range':np.arange(0.05, 0.35, 0.01),
    },
    "energy2": {
        'dataset_name':"energy2", 
        'file_path':"automl_datasets/energy2.csv",
        'cat_features':[],
        'num_features':['X1','X2','X3','X4','X5','X6','X7','X8'],
        'target':'Y1',
        'eps_range':np.arange(0.03, 0.08, 0.01),
        'r_range':np.arange(0.03, 0.08, 0.01),
        'min_points_range':[35, 40, 45, 50, 55],
        'frac_range':np.arange(0.05, 0.35, 0.01),
    },
    "house_16H": {
        'dataset_name':"house_16H",
        'file_path':"automl_datasets/house_16H.csv",
        'cat_features':[],
        'num_features':['P1','P5p1','P6p2','P11p4','P14p9','P15p1','P15p3','P16p2',
                        'P18p2','P27p4','H2p2','H8p2','H10p1','H13p1','H18pA','H40p4'],
        'target':'price',
        'eps_range':np.arange(0.06, 0.12, 0.05),
        'r_range':np.arange(0.06, 0.12, 0.05),
        'min_points_range':[60, 70, 80, 100, 120],
        # 'eps_range':np.arange(0.1, 0.5, 0.025),
        # 'r_range':np.arange(0.1, 0.5, 0.025),
        # 'min_points_range':[20, 30, 40, 60, 80, 100],
        'frac_range':np.arange(0.05, 0.35, 0.01),
    },
    "house_sales": {
        'dataset_name':"house_sales",
        'file_path':"automl_datasets/house_sales.csv",
        'cat_features':[],
        'num_features':['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors','waterfront', 'view', 
                        'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 
                        'lat', 'long', 'sqft_living15', 'sqft_lot15'],
        'target':'price',
        'drop_features':['date_year', 'date_month', 'date_day', 'zipcode'],
        # 'eps_range':np.arange(0.6, 2.0, 0.05),
        # 'r_range':np.arange(0.6, 2.0, 0.05),
        # 'min_points_range':[40, 50, 60, 70, 80, 90, 100],
        'eps_range':np.arange(1.2, 1.7, 0.05),
        'r_range':np.arange(0.9, 1.9, 0.05),
        'min_points_range':[25, 30, 35, 40, 45],
        'frac_range':np.arange(0.25, 0.35, 0.01),
    },
    'kin8nm': {
        'dataset_name':"kin8nm",
        'file_path':"automl_datasets/kin8nm.csv",
        'cat_features':[],
        'num_features':['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6', 'theta7', 'theta8'],
        'target':'y',
        'eps_range':np.arange(0.5, 1.5, 0.1),
        'r_range':np.arange(0.5, 1.5, 0.1),
        'min_points_range':[50, 60, 80, 100],
        'frac_range':np.arange(0.05, 0.3, 0.01),
    },
}


def perform_regression_for_dataset(dataset_name, split_random_state=42, subset_size=None, test_size=0.2,
                                   subset_random_state=42, number_of_noise_cols_added=0, noise_random_state=42,
                                   bblr_regr_type='OLS'):
    d_add = number_of_noise_cols_added
    params = datasets_parameters[dataset_name]
    file_path = params['file_path']
    cat_features = params['cat_features']
    num_features = params['num_features']
    target = params['target']
    drop_features = params.get('drop_features', None)
    eps_range = params['eps_range']
    r_range = params['r_range']
    min_points_range = params['min_points_range']
    frac_range = params['frac_range']
    if subset_size is not None:
        n_name = f"{subset_size}"
    else:
        n_name = "full"

    X_train, X_test, y_train, y_test = prepare_and_split_data(file_path, cat_features, num_features, target,
                                                               test_size, split_random_state=split_random_state, 
                           drop_features=drop_features, subset_size=subset_size, subset_random_state=subset_random_state, 
                           number_of_noise_cols_added=number_of_noise_cols_added, noise_random_state=noise_random_state)

    # BBLR
    best_est_bblr, best_par_bblr = optimize_ball_based_regression(X_train, y_train, eps_range=eps_range, 
                                                        r_range=r_range, min_points_range=min_points_range,
                                                        bblr_regr_type=bblr_regr_type)

    bblr_best = best_est_bblr
    bblr_best.fit(X_train, y_train)
    y_pred_best = bblr_best.predict(X_test)
    rmse_best = root_mean_squared_error(y_test, y_pred_best)
    # print(f"Default BBLR RMSE: {rmse_best}")
    bblr_prediction_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred_best
    })
    bblr_prediction_df.to_csv(f"automl_preds/bblr_predictions_{dataset_name}_{d_add}_{n_name}_{bblr_regr_type}.csv", index=False)
    # save rmse_best to a text file
    with open(f"automl_preds/bblr_rmse_{dataset_name}_{d_add}_{n_name}_{bblr_regr_type}.txt", "w") as f:
        f.write(f"{rmse_best}")
    # save best parameters to a text file
    with open(f"automl_preds/bblr_best_params_{dataset_name}_{d_add}_{n_name}_{bblr_regr_type}.txt", "w") as f:
        f.write(f"{best_par_bblr}")
    pd.DataFrame(bblr_best.coefs).to_csv(f"automl_preds/bblr_coefs_{dataset_name}_{d_add}_{n_name}_{bblr_regr_type}.csv", index=False)
    pd.DataFrame(bblr_best.intercepts).to_csv(f"automl_preds/bblr_intercepts_{dataset_name}_{d_add}_{n_name}_{bblr_regr_type}.csv", index=False)

    # LOWESS
    best_est_lowess, best_par_lowess = optimize_lowess(X_train, y_train, frac_range=frac_range)

    y_pred_lowess = best_est_lowess.predict(X_test, return_ci=False)
    rmse_lowess = root_mean_squared_error(y_test, y_pred_lowess)
    # print(f"Default LOWESS RMSE: {rmse_lowess}")

    lowess_prediction_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred_lowess
    })
    lowess_prediction_df.to_csv(f"automl_preds/lowess_predictions_{dataset_name}_{d_add}_{n_name}.csv", index=False)
    # save rmse_best to a text file
    with open(f"automl_preds/lowess_rmse_{dataset_name}_{d_add}_{n_name}.txt", "w") as f:
        f.write(f"{rmse_lowess}")
    # save best parameters to a text file
    with open(f"automl_preds/lowess_best_params_{dataset_name}_{d_add}_{n_name}.txt", "w") as f:
        f.write(f"{best_par_lowess}")
    # save lowess coefs and intercepts to csv files
    pd.DataFrame(best_est_lowess.coefs_).to_csv(f"automl_preds/lowess_coefs_{dataset_name}_{d_add}_{n_name}.csv", index=False)
    pd.DataFrame(best_est_lowess.intercepts_, columns=['B0']).to_csv(f"automl_preds/lowess_intercepts_{dataset_name}_{d_add}_{n_name}.csv", index=False)
    # TODO: add columns names

    return 0







