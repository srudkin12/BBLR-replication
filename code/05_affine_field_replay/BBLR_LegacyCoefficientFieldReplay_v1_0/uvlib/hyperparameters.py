import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# from sklearn.metrics import root_mean_squared_error
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(".."))

from bblr.bblr2 import *
from bblr.lowess import *


def optimize_ball_based_regression(X_train, y_train, 
                                   eps_range=np.arange(0.3, 0.8, 0.025), 
                                   r_range=np.arange(0.3, 0.8, 0.025), 
                                   min_points_range=[10, 20, 30, 40, 50, 60, 70, 80],
                                   bblr_regr_type='OLS'):
    # Defining parameter grid for GridSearchCV
    param_grid = {
        'eps': eps_range,
        'r': r_range,
        'min_points': min_points_range,
        'coef_comp': ['mean', 'median']
    }
    
    # Initialising the model
    model = BallBasedLinearRegression(regression_type=bblr_regr_type)
    
    # Optimising hyperparameters with GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

def optimize_lowess(X_train, y_train, frac_range=np.arange(0.05, 0.4, 0.01)):
    param_grid = {
        'frac': frac_range
    }
    
    lowess = LOWESSRegressor()
    
    grid_search = RandomizedSearchCV(
        estimator=lowess,
        param_distributions=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_