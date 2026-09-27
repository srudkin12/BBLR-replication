import networkx as nx
import pandas as pd
import numpy as np

from joblib import Parallel, delayed

from scipy.stats import t, uniform, norm
from scipy.spatial import distance
from scipy.optimize import minimize_scalar
from scipy.linalg import get_blas_funcs, eigh

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from numba import jit

import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm

import sys
sys.path.append("../")
from BMR.ballmapper import BallMapper  # TODO: check if BallMapper can be imported from pyballmapper

# Define continuous parameter distribution for alpha
param_dist = {'alpha': uniform(0.01, 100)}  # uniform distribution from 0.001 to 1000.001

def compute_se_construct_ci(X, y, y_fitted, intercept, coefs, weights, confidence = 0.95):
    # Compute confidence intervals for coefficients
    residuals = y - y_fitted
    dof = len(y) - (X.shape[1] + 1)  # Degrees of freedom

    if dof > 0:
        mse = np.sum((residuals ** 2) * weights) / dof
        X_design = np.column_stack((np.ones(len(y)), X))

        try:
            XtWX_inv = np.linalg.inv(X_design.T @ np.diag(weights) @ X_design)
            std_errors = np.sqrt(np.abs(np.diag(mse * XtWX_inv)))
            t_val = t.ppf((1 + confidence) / 2, dof)

            # Confidence intervals for intercept and coefficients
            all_coefs = np.hstack([intercept, coefs])
            conf_int = [(coef - t_val * se, coef + t_val * se)
                        for coef, se in zip(all_coefs, std_errors)]
        except np.linalg.LinAlgError:
            # Handle singular matrix case
            # conf_int = np.array([np.nan, np.nan]) * (X.shape[1] + 1)
            conf_int = [(0, 0)] * (X.shape[1] + 1)
            std_errors = np.array([0] * (X.shape[1] + 1))

    else:
        conf_int = [(0, 0)] * (X.shape[1] + 1)
        std_errors = np.array([0] * (X.shape[1] + 1))

    return conf_int, std_errors

class build_snn_model:
    def __init__( self, data, verbose=1):
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

alpha = 0.05 # 95% confidence interval
confidence = 0.95 # 95% confidence interval
def get_conf_int(X, y, alpha = 0.05):
    lr = sm.OLS(y, sm.add_constant(X)).fit()
    conf_interval = lr.conf_int(alpha)
    if conf_interval.shape != (X.shape[1]+1, 2):
        # Ensure the first column is NOT an intercept, then manually add an intercept
        X = np.column_stack((np.ones(X.shape[0]), X))  # Prepend a column of ones
        lr = sm.OLS(y, X).fit()
        conf_interval = lr.conf_int(alpha)
    return conf_interval

def dist_eucl(x, x0):
    # TODO: replace distance.euclidean with sqrt_einsum_T which is much faster
    # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
    if len(x.shape) == 1:
        d = np.array([distance.euclidean(np.array([xi]), np.array([x0])) for xi in x])
    else:
        d = np.array([distance.euclidean(np.array(xi), np.array(x0)) for xi in x])
    return d

def tricube_weight(x, x0, d, bandwidth):
    """Compute the tricube weight for each data point."""
    # TODO: if bandwidth=max(dist)=0
    u = d / bandwidth
    w = np.where(u < 1, (1 - u**3)**3, 0)
    return w

class BMLR:
    def __init__(self, eps: float, column_names: list, min_points = 10):
        self.eps = eps
        self.fitted = False
        self.ball_mapper = None
        self.min_points = min_points
        self.column_names = column_names

    def find_optimal_radius(self, X_df, y, eps_range, scaling=False, maxiter=15, xatol=0.01,
                            regression_type='OLS', coef_comp='mean', outliers_method='up_to_N', eps_multiplier=0.1):
        # TODO: find the optimal radius with the new BBLR (as in predict_test_fast)
        def bblr_mse(eps):
            k=10
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            y_pred_bblr = np.array([])
            y_true = np.array([])
            for fold, (train_index, test_index) in enumerate(kf.split(X_df)):
                X_train, X_test = X_df.loc[train_index, :], X_df.loc[test_index, :]
                y_train, y_test = y[train_index], y[test_index]

                # TODO: fix the arguments of predict_test_dataset_function
                ypred, _ = predict_test_dataset_function(X_train, X_test, y_train, self.column_names, eps, self.min_points,
                                                         outliers_method=outliers_method, ci=False,
                                                         regression_type=regression_type, scaling=scaling,
                                                         coef_comp=coef_comp, eps_multiplier=eps_multiplier)

                y_true = np.append(y_true, y_test)
                y_pred_bblr = np.append(y_pred_bblr, ypred)

            return mean_squared_error(y_pred_bblr, y_true)

        # Perform the optimization
        result = minimize_scalar(bblr_mse, bounds=eps_range, method='bounded', options={'maxiter': maxiter, 'xatol': xatol})

        # and set the value of the optimised parameter as a value of epsilon and predict
        self.eps = result.x
        # self.predict_whole_dataset_fast(X_df, y, outliers_method=outliers_method, ci=False, scaling=scaling)

        # return the value of the optimised parameter
        return result.x

    def set_eps(self, eps):
        self.eps = eps

    # def coefs_plot(self, feature_name, regression_line=True, conf_int=True):
    #     feature_number = -1  # TODO: return error if feature name is wrong
    #     for i in range(len(self.column_names)):
    #         if self.column_names[i] == feature_name:
    #             feature_number = i
    #     self.plot_df = pd.DataFrame({"x": self.X_true[:, feature_number],
    #                                "lower": [ci[0] for ci in self.confidence_intervals[feature_number]],
    #                                "coefs": self.coefs[:, feature_number],
    #                                "upper": [ci[1] for ci in self.confidence_intervals[feature_number]]})
    #     self.plot_df.sort_values("x", ascending=False, inplace=True)
    #     X = self.plot_df["x"]
    #     Y = self.plot_df["coefs"]
    #     X = X.reset_index()
    #     Y = Y.reset_index()
    #     X = X["x"]
    #     Y = Y["coefs"]
    #
    #     # Perform LOWESS regression to get the central trend
    #     lowess_frac = 0.3  # Smoothing parameter (adjustable)
    #     lowess_result = sm.nonparametric.lowess(Y, X, frac=lowess_frac, return_sorted=True)
    #     X_fit, Y_fit = lowess_result[:, 0], lowess_result[:, 1]
    #
    #     # Compute confidence bands using local percentiles
    #     window_size = 50  # Number of nearest neighbors for local estimation
    #     lower_band = np.zeros_like(X_fit)
    #     upper_band = np.zeros_like(X_fit)
    #
    #     for i in range(len(X_fit)):
    #         # Select a window of closest points
    #         distances = np.abs(X - X_fit[i])
    #         nearest_indices = np.argsort(distances)[:window_size]
    #         y_nearest = Y[nearest_indices]
    #
    #         # Compute 95% confidence bands (2.5% and 97.5% quantiles)
    #         lower_band[i] = np.percentile(y_nearest, 2.5)
    #         upper_band[i] = np.percentile(y_nearest, 97.5)
    #
    #     # Plot the scatter points
    #     plt.scatter(X, Y, color='gray', alpha=0.5, label="Coefficients")
    #
    #     if regression_line:
    #         # Plot the LOWESS regression line
    #         plt.plot(X_fit, Y_fit, color='red', label="LOWESS Fit", linewidth=2)
    #
    #     if conf_int:
    #         # Plot the confidence bands
    #         plt.fill_between(X_fit, lower_band, upper_band, color='pink', alpha=0.3, label="95% Confidence Band")
    #
    #     # Formatting
    #     plt.xlabel("X")
    #     plt.ylabel("Y")
    #     plt.title("Coefficients of " + feature_name +" with 95% Confidence Bands")
    #     plt.legend()
    #     plt.savefig("figures/" + feature_name + ".pdf")
    #     plt.show()

    def fit_predict(self, X_train, X_test, y_train, outliers_method='global', ci=False, regression_type='OLS',
                                   scaling=True, coef_comp='mean', eps_multiplier=0.1, conf_level=0.95,
                             save_training=False, DGP='MA', x_distribution='normal', pred_outliers_method=None,
                             is_opt_eps=False):
        n = len(X_train)

        conf_alpha = 1 - conf_level

        # qnorm_0_975 = 1.959964

        # scale the data
        if scaling:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        # build snn to find the neighbours
        if X_train_scaled.shape[1] == 1:
            X_with_dummy = X_train_scaled.copy()
            # add columns of zeros to X_with_dummy
            X_with_dummy= np.hstack((X_with_dummy, np.zeros((X_with_dummy.shape[0], 1))))
            snn_model = build_snn_model(X_with_dummy)
        else:
            snn_model = build_snn_model(X_train_scaled)

        # col_number = len(self.column_names)
        def estimate_betas_for_X_train(pt_index):
            # selecting indices of points in the epsilon neighbourhood of pt_index
            pt = X_train_scaled[pt_index:pt_index + 1, :]  # Keep 2D shape
            ind = snn_model.query_radius(pt.flatten(), self.eps, return_distance=False)
            mask = ind[0].tolist() if len(ind) > 0 else []

            # computing coefficients and confidence intervals
            global_model_flag = False
            if len(mask) < self.min_points:
                match outliers_method:
                    case 'global':
                        global_model_flag = True
                    case 'up_to_N':
                        if len(mask) < self.min_points:
                            larger_eps = 1000 * self.eps
                            ind, dist = snn_model.query_radius(pt.flatten(), larger_eps, return_distance=True)
                            sort_ind = np.argsort(dist[0])
                            mask = ind[0][sort_ind][:self.min_points].tolist()
                    case 'epsilon_multiplied':
                        larger_eps = 1 * self.eps
                        while len(mask) < self.min_points:
                            larger_eps = (1 + eps_multiplier) * larger_eps
                            ind, dist = snn_model.query_radius(pt.flatten(), larger_eps, return_distance=True)
                            mask = ind[0].tolist() if len(ind) > 0 else []
                    case _:
                        raise ValueError(f"Unknown outliers method: {outliers_method}")

            n_points_in_ball = len(mask)

            match regression_type:
                case 'WLS':
                    if global_model_flag:
                        d = dist_eucl(X_train_scaled.squeeze(), pt.squeeze())
                        weights = tricube_weight(X_train_scaled.squeeze(), pt.squeeze(), d, bandwidth=np.max(d))
                        model = LinearRegression().fit(X_train_scaled, y_train, sample_weight=weights)
                    else:
                        d = dist_eucl(X_train_scaled[mask, :].squeeze(), pt.squeeze())
                        weights = tricube_weight(X_train_scaled[mask, :].squeeze(), pt.squeeze(), d, self.eps)
                        model = LinearRegression().fit(X_train_scaled[mask, :], y_train[mask], sample_weight=weights)

                    y_predict = model.predict(pt)

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
                        random_search.fit(X_train_scaled, y_train)
                        optimal_alpha = random_search.best_params_['alpha']
                        model = Ridge(alpha=optimal_alpha).fit(X_train_scaled, y_train)
                        weights = np.ones(X_train_scaled.shape[0])

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
                        random_search.fit(X_train_scaled[mask, :], y_train[mask])
                        optimal_alpha = random_search.best_params_['alpha']
                        model = Ridge(alpha=optimal_alpha).fit(X_train_scaled[mask, :], y_train[mask])
                        weights = np.ones(X_train_scaled[mask, :].shape[0])

                    y_predict = model.predict(pt)

                case 'OLS':
                    if global_model_flag:
                        model = LinearRegression().fit(X_train_scaled, y_train)
                        weights = np.ones(X_train_scaled.shape[0])

                    else:
                        model = LinearRegression().fit(X_train_scaled[mask, :], y_train[mask])
                        weights = np.ones(X_train_scaled[mask, :].shape[0])

                    y_predict = model.predict(pt)

            if global_model_flag:
                y_fitted = model.predict(X_train_scaled)
                conf_int, std_errors = compute_se_construct_ci(X_train_scaled, y_train,
                                                               y_fitted, model.intercept_, model.coef_,
                                                               weights, confidence = conf_alpha)
            else:
                y_fitted = model.predict(X_train_scaled[mask, :])
                conf_int, std_errors = compute_se_construct_ci(X_train_scaled.loc[mask, :], y_train[mask],
                                                               y_fitted, model.intercept_, model.coef_,
                                                               weights, confidence = conf_alpha)

            if not ci:
                std_errors = np.array([0 for i in range(X_train_scaled.shape[1] + 1)])

            lower_ci = [ci_tuple[0] for ci_tuple in conf_int]
            upper_ci = [ci_tuple[1] for ci_tuple in conf_int]

            return model.coef_, model.intercept_, std_errors**2, y_predict, lower_ci, upper_ci, n_points_in_ball


        # uncomment for debugging
        # for pt_index in tqdm(X_train_scaled.index):
        #     estimate_betas_for_X_train(pt_index)

        # estimate beta coefficients for training dataset
        results_train = Parallel(n_jobs=-1)(delayed(estimate_betas_for_X_train)(pt_index)
                                            for pt_index in range(len(X_train_scaled)))

        # Unpack results
        (X_train_points_coefs, X_train_points_intercepts,
         X_train_points_variances, X_train_points_y_predicts,
         X_train_points_lower_cis, X_train_points_upper_cis, n_points_in_ball_all) = zip(*results_train)

        if save_training:
            n_all = len(X_train) + len(X_test)
            n_features = X_train.shape[1]

            # Create numpy arrays for all components
            case_summary_data = np.column_stack([
                X_train_points_y_predicts.flatten(),
                np.full(len(X_train_points_y_predicts), DGP),
                np.full(len(X_train_points_y_predicts), 'BBLR'),
                np.full(len(X_train_points_y_predicts), self.eps),
                np.full(len(X_train_points_y_predicts), n_all)
            ])

            # Coefficients and intercepts
            coefs_intercept_data = np.column_stack([X_train_points_intercepts, X_train_points_coefs])

            # Standard errors
            std_errors_data = np.sqrt(X_train_points_variances)

            # Confidence intervals
            lower_cis_data = np.array(X_train_points_lower_cis)
            upper_cis_data = np.array(X_train_points_upper_cis)

            # Y values and points in ball
            y_pts_data = np.column_stack([y_train, n_points_in_ball_all])

            # Combine all data
            training_results_data = np.column_stack([
                case_summary_data,
                coefs_intercept_data,
                X_train,  # Original X_train data
                std_errors_data,
                lower_cis_data,
                upper_cis_data,
                y_pts_data
            ])

            # Create column names
            case_summary_columns = ['YP', 'GDP', 'method', 'parameter', 'sample_size']
            coef_columns = [f'B{i}' for i in range(n_features + 1)]
            feature_columns = [f'X{i + 1}' for i in range(n_features)]
            se_columns = [f'se{i}' for i in range(n_features + 1)]
            lower_ci_columns = [f'L{i}' for i in range(n_features + 1)]
            upper_ci_columns = [f'U{i}' for i in range(n_features + 1)]
            y_pts_columns = ['Y', 'pts_in_ball']

            training_results_columns = (
                    case_summary_columns + coef_columns + feature_columns +
                    se_columns + lower_ci_columns + upper_ci_columns + y_pts_columns
            )

            if is_opt_eps:
                eps_bblr = 0.0
            else:
                eps_bblr = self.eps

            # TODO: move the saving part elsewhere
            # Convert to pandas DataFrame only for saving
            # training_results_df = pd.DataFrame(training_results_data, columns=training_results_columns)
            # training_results_df.to_csv(f"raw_results/{DGP}_BBLR_{eps_bblr}_{n_all}_{x_distribution}_{outliers_method}_{regression_type}_{coef_comp}_{ci}_train.csv", index=False)

            # Return numpy data and columns for further use if needed
            return training_results_data, training_results_columns

        # build snn to find the neighbours for whole dataset
        X_all_scaled = pd.concat((X_train_scaled, X_test_scaled), axis=0)
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
            pt = X_test_scaled[[pt_index], :]
            ind = snn_model_full.query_radius(pt.flatten(), self.eps, return_distance=False)
            mask = [i for i in ind if i < n]

            # computing coefficients and confidence intervals
            global_model_flag = False
            if len(mask) < self.min_points:
                match outliers_method:
                    case 'global':
                        mask = [i for i in range(n)]
                    case 'up_to_N':
                        if len(mask) < self.min_points:
                            larger_eps = 1000 * self.eps
                            ind, dist = snn_model_full.query_radius(pt.flatten(), larger_eps,
                                                                    return_distance=True)
                            sort_ind = np.argsort(dist)
                            mask = [i for i in ind[sort_ind] if i < n][:(self.min_points)]
                    case 'epsilon_multiplied':
                        larger_eps = 1 * self.eps
                        while len(mask) < self.min_points:
                            larger_eps = (1 + eps_multiplier) * larger_eps
                            ind, dist = snn_model_full.query_radius(pt.flatten(), larger_eps,
                                                                    return_distance=True)
                            mask = [i for i in ind if i < n]
                    case _:
                        raise ValueError(f"Unknown outliers method: {outliers_method}")

            n_points_in_ball = len(mask)

            match coef_comp:
                case 'mean':
                    betas = np.mean(X_train_points_coefs[mask,:], axis=0)
                    intercept = np.mean(X_train_points_intercepts[mask])
                case 'median':
                    betas = np.median(X_train_points_coefs[mask,:], axis=0)
                    intercept = np.median(X_train_points_intercepts[mask])
            yp = intercept + np.dot(betas, pt.squeeze())
            conf_int = np.array([(0, 0)] * (X_train_scaled.shape[0] + 1))
            std_errors = np.array([0] * (X_train_scaled.shape[0] + 1))
            weights = np.ones(len(mask))
            y_fitted = [intercept + np.dot(betas, pt.squeeze()) for pt in X_train_points_coefs[mask,:]]
            if ci:
                conf_int, std_errors = compute_se_construct_ci(X_train_scaled.loc[mask, :], y_train[mask],
                                                               y_fitted, intercept, betas,
                                                               weights, confidence=conf_alpha)

            return betas, yp, intercept, conf_int, std_errors, n_points_in_ball

        # uncomment for debugging
        # for pt_index in tqdm(X_test_scaled.index):
        #     estimate_betas_and_y_for_X_test(pt_index)

        results_test = Parallel(n_jobs=-1)(delayed(estimate_betas_and_y_for_X_test)(pt_index) for pt_index in X_test_scaled.index)
        betas, ypred, intercepts, conf_ints, std_errors_all, n_points_in_ball_all = zip(*results_test)

        self.pred = ypred
        # self.coefs = pd.DataFrame(betas, columns=self.column_names)
        self.coefs = betas
        self.intercepts = intercepts
        self.conf_ints = conf_ints
        self.n_points_in_ball_all = n_points_in_ball_all
        self.std_errors_all = std_errors_all

    #
    # def find_inball_points_mask(self, query_point):
    #     distances_mask = np.linalg.norm(self.data - query_point, axis=1) < self.eps
    #     return distances_mask
    #
    # def create_graph(self, seed=None):
    #     order = None
    #     self.ball_mapper = BallMapper(X=self.data, eps=self.eps, order=order, verbose=False)
    #     self._graph_layout = nx.spring_layout(self.ball_mapper.Graph, seed=seed)
    #
    # def plot_coefficient(self, feature, ax=None, color_map=plt.cm.coolwarm):
    #     if not self.fitted:
    #         raise ValueError("Run .fit() first!")
    #     if self.ball_mapper is None:
    #         raise ValueError("Run .create_graph() first!")
    #     if feature not in self.columns:
    #         raise ValueError(f'Coefficient {feature} not found! Options are {self.columns}')
    #     column_id = self.columns.index(feature)
    #     # compute average value of the coeffcient each ball
    #     node_coef = []
    #     for in_ball_ids in self.ball_mapper.points_covered_by_landmarks.values():
    #         avg = np.mean([self.models_inball[pt_id].coef_[column_id] for pt_id in in_ball_ids])
    #         node_coef.append(avg)
    #     color_normalization = mpl.colors.Normalize()
    #     sm = plt.cm.ScalarMappable(norm=color_normalization, cmap=color_map)
    #     node_color = sm.to_rgba(node_coef)
    #     show_plot = False
    #     if ax is None:
    #         fig, ax = plt.subplots(1, 1)
    #         show_plot = True
    #     nx.draw_networkx(self.ball_mapper.Graph, pos=self._graph_layout, node_color=node_color, ax=ax)
    #     plt.colorbar(sm, ax=ax)
    #     ax.set_title(feature)
    #     if show_plot:
    #         plt.show()

