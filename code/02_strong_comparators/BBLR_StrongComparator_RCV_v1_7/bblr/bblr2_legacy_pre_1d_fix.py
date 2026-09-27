import numpy as np
import pandas as pd
from numba import jit

from joblib import Parallel, delayed

import math

from scipy.stats import uniform
from scipy.optimize import minimize_scalar
from scipy.linalg import get_blas_funcs, eigh
from scipy.spatial.distance import cdist

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

import warnings
# warnings.filterwarnings('ignore')

import sys
sys.path.append("../")
from bblr.ci import *
# from bblr.lpr import LocalPolynomialRegression

# Define continuous parameter distribution for alpha
param_dist = {'alpha': uniform(0.01, 100)}  # uniform distribution from 0.001 to 1000.001

# Define default confidence level
alpha = 0.05 # 95% confidence interval
confidence = 0.95 # 95% confidence interval


@jit(nopython=True)
def tricube_weight_numba(X, pt, distances, bandwidth):
    """Numba-optimized tricube weight function"""
    n = X.shape[0]
    weights = np.zeros(n)
    for i in range(n):
        u = abs(distances[i]) / bandwidth
        if u < 1:
            weights[i] = (1 - u**3)**3
        else:
            weights[i] = 0.0
    return weights

@jit(nopython=True)
def fast_euclidean_distances(X, pt):
    """Numba-optimized Euclidean distance"""
    n, d = X.shape
    distances = np.zeros(n)
    for i in range(n):
        dist = 0.0
        for j in range(d):
            diff = X[i, j] - pt[j]
            dist += diff * diff
        distances[i] = math.sqrt(dist)
    return distances


class build_snn_model:
    def __init__(self, data, verbose=1):
        self.verbose = verbose

        self.mu = data.mean(axis=0)
        data = data - self.mu
        if data.shape[1]>1 and data.shape[1]<=2 :
            gemm = get_blas_funcs("gemm", [data.T, data])
            dTd = gemm(1, data.T, data)
            _, v = eigh(dTd, subset_by_index=[data.shape[1]-1, data.shape[1]-1])
            sort_vals = data@v.reshape(-1)

        elif data.shape[1]>2:
            dTd = np.dot(data.T, data)
            _, v = eigh(dTd, subset_by_index=[data.shape[1]-1, data.shape[1]-1])
            sort_vals = data@v.reshape(-1)

        else:
            sort_vals = data[:,0].reshape(-1)

        self.sort_id = np.argsort(sort_vals)
        self.sort_vals = sort_vals[self.sort_id]
        self.data = data[self.sort_id]
        self.v = v.reshape(-1)
        self.xxt = np.einsum('ij,ij->i', self.data, self.data) # np.linalg.norm(X, axis=1)**2

    def query_radius(self, query, r, return_distance=False):
        query = np.subtract(query, self.mu)
        sv_q = np.inner(query, self.v)
        left = np.searchsorted(self.sort_vals, sv_q-r)
        right = np.searchsorted(self.sort_vals, sv_q+r)
        dist_set = euclid(self.xxt[left:right], self.data[left:right], query)

        filter_r = dist_set <= r**2
        knn_ind = self.sort_id[left:right][filter_r]

        if return_distance:
            knn_dist = np.sqrt(dist_set[filter_r])
            return knn_ind, knn_dist
        else:
            return knn_ind

    def batch_query_radius(self, queries, r, return_distance=False):
        queries = queries - self.mu
        sv_q = queries @ self.v
        r_sq = r ** 2

        left = np.searchsorted(self.sort_vals, sv_q - r)
        right = np.searchsorted(self.sort_vals, sv_q + r)

        results = []
        distances = []

        for i in range(len(queries)):
            idx_range = slice(left[i], right[i])
            dist_sq = self.xxt[idx_range] + np.sum(queries[i] ** 2) - 2 * (self.data[idx_range] @ queries[i])

            mask = dist_sq <= r_sq
            indices = self.sort_id[idx_range][mask]

            if return_distance:
                results.append(indices)
                distances.append(np.sqrt(dist_sq[mask]))
            else:
                results.append(indices)

        return (results, distances) if return_distance else results


def euclid(xxt, X, v):
    return np.abs(xxt + np.inner(v, v).ravel() -2*X.dot(v)).astype(float)

def optimized_distances(X, pt):
    # Use cdist which is highly optimized
    return cdist(X, pt[None, :]).flatten()

class BallBasedLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, eps = 0.0, column_names = [], min_points = 10, regression_type='OLS',
                 coef_comp='mean', outliers_method='up_to_N', eps_multiplier=0.1,
                 eps_range=(0.1, 3), ci=False, pred_outliers_method=None, confidence=0.95, r=None):
        self.eps = eps
        self.fitted = False
        self.min_points = min_points
        self.column_names = column_names
        self.regression_type = regression_type
        self.coef_comp = coef_comp
        self.outliers_method = outliers_method
        self.eps_multiplier = eps_multiplier
        self.eps_range = eps_range
        self.pred_outliers_method = pred_outliers_method
        self.ci = ci
        self.confidence = confidence
        self.r = eps if r is None else r

    def find_optimal_radius(self, eps_range, k=10, maxiter=15, xatol=0.001):
        # k - number of splits in k-fold CV
        def bblr_mse(eps):

            # divide row indices into k parts
            kf = KFold(n_splits=k, shuffle=True, random_state=42)

            # store values predicted by BBLR in each fold
            y_pred_bblr = np.array([])

            # store true values of y
            y_true = np.array([])
            for fold, (train_index, test_index) in enumerate(kf.split(self.X_train)):
                # divide the whole training set into a training and test set for a certain fold 
                X_train_fold, X_test_fold = self.X_train[train_index, :], self.X_train[test_index, :]
                y_train_fold, y_test_fold = self.y_train[train_index], self.y_train[test_index]

                bblr_fold = BallBasedLinearRegression(eps=eps,
                                                      column_names=self.column_names,
                                                      min_points=self.min_points,
                                                      regression_type=self.regression_type,
                                                      coef_comp=self.coef_comp, 
                                                      outliers_method=self.outliers_method, 
                                                      eps_multiplier=self.eps_multiplier,
                                                      eps_range=None, 
                                                      ci=False, 
                                                      pred_outliers_method=None,
                                                      r=eps
                                                      )
                
                bblr_fold.fit(X_train_fold, y_train_fold)
                bblr_fold.predict(X_test_fold)

                y_true = np.append(y_true, y_test_fold)
                y_pred_bblr = np.append(y_pred_bblr, bblr_fold.pred)

            return mean_squared_error(y_pred_bblr, y_true)

        # Perform the optimization
        result = minimize_scalar(bblr_mse, bounds=eps_range, method='bounded', options={'maxiter': maxiter, 'xatol': xatol})

        # and set the value of the optimised parameter as a value of epsilon and predict
        self.eps = result.x

        # return the value of the optimised parameter
        return result.x

    def fit(self, X_train, y_train):
        if type(y_train) == pd.Series:
            y_train = np.array(y_train)
        self.X_train = X_train
        self.y_train = y_train

        if self.eps == 0:
            self.find_optimal_radius(self.eps_range)
            self.r = self.eps  # r was 0 (= initial eps); sync with the newly optimized eps

        # warn the user that to construct the confidence intervals, min_points > p + 2 is needed
        if self.ci and self.min_points <= self.X_train.shape[1] + 1 + 1:
            warnings.warn("To construct the confidence intervals, the minimum number of points in each ball should be larger than p+1, where p is the number of features. Increasing the values of the parameter 'min_points' might be necessary.")

        return self


    def predict(self, X_test):
        self.X_test = X_test

        # for each feature compute the maximum of absolute values in the training set and sum their squares
        large_distance = np.sqrt(np.sum(np.max(np.abs(self.X_train), axis=0)**2))

        n = len(self.X_train)

        # qnorm_0_975 = 1.959964

        # build snn to find the neighbours
        if self.X_train.shape[1] == 1:
            X_with_dummy = self.X_train.copy()
            # add columns of zeros to X_with_dummy
            X_with_dummy = np.hstack((X_with_dummy, np.zeros((X_with_dummy.shape[0], 1))))
            snn_model = build_snn_model(X_with_dummy)
        else:
            snn_model = build_snn_model(self.X_train)

        def estimate_betas_for_X_train(pt_index):
            # selecting indices of points in the epsilon neighbourhood of pt_index
            pt = self.X_train[pt_index:pt_index + 1, :]  # Keep 2D shape
            mask = snn_model.query_radius(pt.flatten(), self.r, return_distance=False)  #changing eps to r
            if len(mask) == 0:
                mask = []

            # computing coefficients and confidence intervals
            global_model_flag = False
            if len(mask) < self.min_points:
                match self.outliers_method:
                    case 'global':
                        global_model_flag = True
                    case 'up_to_N':
                        if len(mask) < self.min_points:
                            larger_eps = large_distance * 4
                            ind, dist = snn_model.query_radius(pt.flatten(), larger_eps, return_distance=True)
                            sort_ind = np.argsort(dist)
                            mask = ind[sort_ind][:self.min_points].tolist()
                    case 'epsilon_multiplied':
                        larger_eps = 1 * self.eps
                        while len(mask) < self.min_points:
                            larger_eps = (1 + self.eps_multiplier) * larger_eps
                            ind, dist = snn_model.query_radius(pt.flatten(), larger_eps, return_distance=True)
                            mask = ind[0].tolist() if len(ind) > 0 else []
                    case _:
                        raise ValueError(f"Unknown outliers method: {self.outliers_method}")

            # n_points_in_ball = len(mask)

            match self.regression_type:
                case 'LPR2':  # LPR2 is the local polynomial regression with degree 2
                    poly = PolynomialFeatures(degree=self.degree, include_bias=False)
                    poly_features_train = poly.fit_transform(self.X_train)
                    poly_features_test = poly.fit_transform(self.X_test)
                    if global_model_flag:
                        model = LinearRegression().fit(poly_features_train, self.y_train)
                        weights = np.ones(self.X_train.shape[0])
                    else:
                        model = LinearRegression().fit(poly_features_train[mask, :], self.y_train[mask])
                        weights = np.ones(self.X_train[mask, :].shape[0])
                case 'WLS':
                    if global_model_flag:
                        d = fast_euclidean_distances(self.X_train, pt.squeeze())
                        weights = tricube_weight_numba(self.X_train, pt.squeeze(), d, self.eps)
                        model = LinearRegression().fit(self.X_train, self.y_train, sample_weight=weights)
                    else:
                        d = fast_euclidean_distances(self.X_train[mask, :], pt.squeeze())
                        weights = tricube_weight_numba(self.X_train[mask, :], pt.squeeze(), d, self.eps)
                        model = LinearRegression().fit(self.X_train[mask, :], self.y_train[mask], sample_weight=weights)

                    # y_predict = model.predict(pt)

                case 'Ridge':
                    if global_model_flag:
                        # Use sklearn for hyperparameter tuning
                        ridge = Ridge()
                        random_search = RandomizedSearchCV(
                            ridge,
                            param_dist,
                            n_iter=20,
                            cv=5,
                            scoring='neg_mean_squared_error',
                            random_state=42
                        )
                        random_search.fit(self.X_train, self.y_train)
                        optimal_alpha = random_search.best_params_['alpha']
                        model = Ridge(alpha=optimal_alpha).fit(self.X_train, self.y_train)
                        weights = np.ones(self.X_train.shape[0])

                    else:
                        # Use sklearn for hyperparameter tuning
                        ridge = Ridge()
                        random_search = RandomizedSearchCV(
                            ridge,
                            param_dist,
                            n_iter=20,
                            cv=5,
                            scoring='neg_mean_squared_error',
                            random_state=42
                        )
                        random_search.fit(self.X_train[mask, :], self.y_train[mask])
                        optimal_alpha = random_search.best_params_['alpha']
                        model = Ridge(alpha=optimal_alpha).fit(self.X_train[mask, :], self.y_train[mask])
                        weights = np.ones(self.X_train[mask, :].shape[0])

                    # y_predict = model.predict(pt)

                case 'OLS':
                    if global_model_flag:
                        model = LinearRegression().fit(self.X_train, self.y_train)
                        weights = np.ones(self.X_train.shape[0])

                    else:
                        model = LinearRegression().fit(self.X_train[mask, :], self.y_train[mask])
                        weights = np.ones(self.X_train[mask, :].shape[0])

                    # y_predict = model.predict(pt)

            return model.coef_, model.intercept_


        # uncomment for debugging
        # for pt_index in range(len(self.X_train)):
        #     estimate_betas_for_X_train(pt_index)

        # estimate beta coefficients for training dataset
        results_train = Parallel(n_jobs=-1)(delayed(estimate_betas_for_X_train)(pt_index)
                                            for pt_index in range(len(self.X_train)))

        # Pre-allocate results arrays
        n_train = len(self.X_train)
        n_features = self.X_train.shape[1]

        X_train_points_coefs = np.zeros((n_train, n_features))
        X_train_points_intercepts = np.zeros(n_train)

        # Fill arrays directly in parallel computation
        for i, result in enumerate(results_train):
            X_train_points_coefs[i] = result[0]
            X_train_points_intercepts[i] = result[1]

        # build snn to find the neighbours for whole dataset
        X_all_scaled = np.concatenate((self.X_train, self.X_test), axis=0)
        if X_all_scaled.shape[1] == 1:
            X_with_dummy = X_all_scaled.copy()
            # add columns of zeros to X_with_dummy
            X_with_dummy = np.hstack((X_with_dummy, np.zeros((X_with_dummy.shape[0], 1))))
            snn_model_full = build_snn_model(X_with_dummy)
        else:
            snn_model_full = build_snn_model(X_all_scaled)

        # TODO: include pred_outliers_method in the loop
        # if pred_outliers_method is None:
        #     pred_outliers_method = outliers_method

        def estimate_betas_and_y_for_X_test(pt_index):
            # selecting indices of points in the epsilon neighbourhood of pt_index
            pt = self.X_test[[pt_index], :]
            ind = snn_model_full.query_radius(pt.flatten(), self.eps, return_distance=False)
            mask = [i for i in ind if i < n]

            # computing coefficients and confidence intervals
            global_model_flag = False
            if len(mask) < self.min_points:
                match self.outliers_method:
                    case 'global':
                        mask = [i for i in range(n)]
                    case 'up_to_N':
                        if len(mask) < self.min_points:
                            larger_eps = large_distance * 4  # TODO: 4 may be not enough
                            ind, dist = snn_model_full.query_radius(pt.flatten(), larger_eps,
                                                                    return_distance=True)
                            sort_ind = np.argsort(dist)
                            mask = [i for i in ind[sort_ind] if i < n][:(self.min_points)]
                    case 'epsilon_multiplied':
                        larger_eps = 1 * self.eps
                        while len(mask) < self.min_points:
                            larger_eps = (1 + self.eps_multiplier) * larger_eps
                            ind, dist = snn_model_full.query_radius(pt.flatten(), larger_eps,
                                                                    return_distance=True)
                            mask = [i for i in ind if i < n]
                    case _:
                        raise ValueError(f"Unknown outliers method: {self.outliers_method}")

            n_points_in_ball = len(mask)

            # Convert to 2D array
            X_train_points_coefs_np = np.vstack(X_train_points_coefs)

            match self.coef_comp:
                case 'mean':
                    betas = np.mean(X_train_points_coefs_np[mask,:], axis=0)
                    intercept = np.mean(np.array(X_train_points_intercepts)[mask])
                case 'median':
                    betas = np.median(X_train_points_coefs_np[mask,:], axis=0)
                    intercept = np.median(np.array(X_train_points_intercepts)[mask])
                case 'wmean':
                    max_dist = max(fast_euclidean_distances(self.X_train[mask, :], pt.squeeze()))
                    weights = tricube_weight_numba(self.X_train[mask, :], pt.squeeze(),
                                                  fast_euclidean_distances(self.X_train[mask, :], pt.squeeze()), max_dist)
                    betas = np.average(X_train_points_coefs_np[mask,:], axis=0, weights=weights)
                    intercept = np.average(np.array(X_train_points_intercepts)[mask], weights=weights)
                case _:
                    raise ValueError(f"Unknown coef_comp method: {self.coef_comp}")
            if self.regression_type is not 'LPR2':
                yp = intercept + np.dot(betas, pt.squeeze())
            else:
                poly = PolynomialFeatures(degree=self.degree, include_bias=False)
                pt_poly = poly.fit_transform(pt)
                yp = intercept + np.dot(betas, pt_poly.squeeze())
                # betas = betas[:self.X_train.shape[1]] + 2 * betas[self.X_train.shape[1]:] * pt.squeeze()
            weights = np.ones(len(mask))
            # TODO: check if CI is correctly computed
            y_fitted = [intercept + np.dot(betas, pt.squeeze()) for pt in self.X_train[mask,:]]
            if self.ci:
                conf_int, std_errors = compute_se_construct_ci(self.X_train[mask, :], self.y_train[mask],
                                                               y_fitted, intercept, betas,
                                                               weights, confidence=self.confidence)
                # print(conf_int)
            else:
                conf_int = np.array([(0, 0)] * (self.X_train.shape[0] + 1))
                std_errors = np.array([0] * (self.X_train.shape[0] + 1))

            return betas, yp, intercept, conf_int, std_errors, n_points_in_ball

        # uncomment for debugging
        # for pt_index in range(len(self.X_test)):
        #     estimate_betas_and_y_for_X_test(pt_index)

        results_test = Parallel(n_jobs=-1)(delayed(estimate_betas_and_y_for_X_test)(pt_index) for pt_index in range(len(self.X_test)))
        betas, ypred, intercepts, conf_ints, std_errors_all, n_points_in_ball_all = zip(*results_test)

        self.pred = np.array(ypred)
        self.coefs = np.array(betas)
        self.intercepts = np.array(intercepts)
        self.conf_ints = conf_ints
        self.n_points_in_ball = n_points_in_ball_all
        self.std_errors_all = np.array(std_errors_all)

        return self.pred

