import numpy as np
import pandas as pd
# import time
# from scipy.optimize import minimize_scalar
#
# from sklearn.datasets import make_friedman2, make_friedman3
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing

# from ucimlrepo import fetch_ucirepo
#
# from xgboost import XGBRegressor
#
# import shap
# from lime.lime_tabular import LimeTabularExplainer
#
# import os
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder



def load_real_data(dataset_name, scaling=False, sample_size=None, random_state=42, dataset_file_name=None,
                   target=None):
    # set seed
    np.random.seed(random_state)
    random.seed(random_state)

    match dataset_name:
        case 'custom':
            # Load data
            df = pd.read_csv(f'../data/{dataset_file_name}')
            columns_names = df.columns.tolist()
            y = df[target]
            df.drop(columns=target, inplace=True)
            columns_names.remove(target)

            df_np = df.to_numpy()
            y_np = y.to_numpy()

            if sample_size is not None and sample_size > 0:
                # choose sample from whole dataset
                all_ids = range(df.shape[0])
                chosen_ids = random.sample(all_ids, sample_size)
                subsample_x = df.to_numpy()[chosen_ids,:]
                subsample_y = y_np[chosen_ids]
            else:
                subsample_x = df.to_numpy()
                subsample_y = y_np
            if scaling:
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(subsample_x)
            else:
                X_scaled = subsample_x
            X_scaled_df = pd.DataFrame(X_scaled)

            if sample_size is not None and sample_size > 0:
                y = y[chosen_ids]
            y_df = pd.Series(y).reset_index(drop=True)

            X_scaled_df.columns = columns_names
            return X_scaled_df, y_df, X_scaled, y, columns_names

        case 'california_housing':
            # data importing and processing
            housing = fetch_california_housing()
            raw_data = housing.data

            columns_list = ['Median_Income', 'Median_Age', 'Tot_Rooms',
                'Tot_Bedrooms', 'Population', 'Households', 'Latitude', 'Longitude']
            # columns_list = ['Median_Income', 'Median_Age', 'Tot_Rooms',
            #        'Tot_Bedrooms', 'Population', 'Households']
            raw_data = pd.DataFrame(raw_data, columns=columns_list)
            raw_data_selected = raw_data[columns_list]

            if sample_size is not None and sample_size > 0:
                # choose sample from whole dataset
                all_ids = range(raw_data_selected.shape[0])
                chosen_ids = random.sample(all_ids, sample_size)
                subsample_x = raw_data_selected.to_numpy()[chosen_ids,:]
            else:
                subsample_x = raw_data_selected.to_numpy()

            if scaling:
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(subsample_x)
            else:
                X_scaled = subsample_x
            X_scaled_df = pd.DataFrame(X_scaled)

            y = housing.target
            if sample_size is not None and sample_size > 0:
                y = y[chosen_ids]
            y_df = pd.Series(y).reset_index(drop=True)
            columns_names = X_scaled_df.columns
            X_scaled_df.columns = columns_list
            return X_scaled_df, y_df, X_scaled, y, columns_names
        
        case 'california_housing_with_counties':
            # California data with Counties names
            # https://www.kaggle.com/datasets/abdallahsamman/california-housing-with-name-of-counties

            columns_list = ['Median_Income', 'Median_Age', 'Tot_Rooms',
                'Tot_Bedrooms', 'Population', 'Households', 'Latitude', 'Longitude']
            # columns_list = ['Median_Income', 'Median_Age', 'Tot_Rooms',
            #        'Tot_Bedrooms', 'Population', 'Households']

            # data importing and processing
            raw_data = pd.read_csv('../data/California_Housing_CitiesAdded.csv')
            raw_data_selected = raw_data[columns_list]

            if sample_size is not None and sample_size > 0:
                # choose sample from whole dataset
                all_ids = range(raw_data_selected.shape[0])
                chosen_ids = random.sample(all_ids, sample_size)
                subsample_x = raw_data_selected.to_numpy()[chosen_ids,:]
            else:
                subsample_x = raw_data_selected.to_numpy()

            if scaling:
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(subsample_x)
            else:
                X_scaled = subsample_x
            X_scaled_df = pd.DataFrame(X_scaled)

            y = raw_data['Median_House_Value']/1e5
            if sample_size is not None and sample_size > 0:
                y = y[chosen_ids]
            y_df = pd.Series(y).reset_index(drop=True)
            columns_names = X_scaled_df.columns
            X_scaled_df.columns = columns_list
            return X_scaled_df, y_df, X_scaled, y, columns_names

        case 'efdi':
            # Load data
            efdi_data = pd.read_csv('../data/efdiall2.csv')
            efdi_data_array = efdi_data.to_numpy()

            # Prepare data
            y = efdi_data['efdi']
            columns_names = efdi_data.columns[0:(efdi_data.shape[1]-1)]

            if sample_size is not None and sample_size > 0:
                # choose sample from whole dataset
                all_ids = range(efdi_data_array.shape[0])
                chosen_ids = random.sample(all_ids, sample_size)
                X = efdi_data_array.to_numpy()[chosen_ids,:]
            else:
                X = efdi_data_array.to_numpy()

            if sample_size is not None and sample_size > 0:
                y = y[chosen_ids]
            y.reset_index(drop=True, inplace=True)

            if scaling:
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X
            X_scaled_df = pd.DataFrame(X_scaled)

            y_df = pd.Series(y).reset_index(drop=True)

            return X_scaled_df, y_df, X_scaled, y, columns_names
        
        case 'brexit_large':
            # upload and prepare data
            full_df = pd.read_csv("../data/brexitdata", sep="\t", encoding="iso-8859-1")

            # subset of features
            # large subset
            subset = [
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
            ]

            X = full_df[subset]
            y_df = full_df["leaveHanretty"]

            if scaling:
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X
            X_scaled_df = pd.DataFrame(X_scaled)

            X_np = X_scaled_df.copy().to_numpy()
            y = full_df["leaveHanretty"].copy().to_numpy()

            return X_scaled_df, y_df, X_np, y, X_scaled_df.columns

        case 'brexit_small':
            # upload and prepare data
            full_df = pd.read_csv("../data/brexitdata", sep="\t", encoding="iso-8859-1")

            # subset of features
            # small subset
            subset = [
                "c11CarsNone",
                "c11CarsTwo",
                "c11QualLevel4",
                "c11HealthVeryGood",
                "c11HealthGood",
            ]


            X = full_df[subset]
            y_df = full_df["leaveHanretty"]

            if scaling:
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X
            X_scaled_df = pd.DataFrame(X_scaled)

            X_np = X_scaled_df.copy().to_numpy()
            y = full_df["leaveHanretty"].copy().to_numpy()
            
            return X_scaled_df, y_df, X_np, y, X_scaled_df.columns



