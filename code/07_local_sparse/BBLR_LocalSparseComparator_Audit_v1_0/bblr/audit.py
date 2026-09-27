from __future__ import annotations

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso

from .nd_core import BBLRND, LocalPolynomialND, GlobalLinearND, radius_grid_from_knn
from .dgps import generate_sample, reference_alpha_truth
from .local_sparse import AdaptiveLocalLinearLassoND


def _sse(a, b):
    z = np.asarray(a, float) - np.asarray(b, float)
    return float(np.sum(z * z))


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def fold_iterator(n, folds, repeats, seed):
    ss = np.random.SeedSequence(int(seed))
    for ch in ss.spawn(int(repeats)):
        rs = int(ch.generate_state(1)[0])
        kf = KFold(n_splits=int(folds), shuffle=True, random_state=rs)
        for tr, va in kf.split(np.arange(n)):
            yield tr, va


def _stage1_min_points(d):
    return max(int(d) + 5, 15)


def _k_targets(n, d):
    n = int(n); d = int(d)
    cap = max(d + 5, int(n * 0.75))
    raw = [max(40, 4*d+16), max(80, 8*d+32), max(160, 16*d+64), max(320, 32*d+128)]
    return sorted(set(min(cap, int(k)) for k in raw if min(cap, int(k)) >= d + 5))


def _candidate_radii(X, d):
    return radius_grid_from_knn(np.asarray(X, float), _k_targets(len(X), d))


class GlobalLassoND:
    def __init__(self, alpha=0.01):
        self.alpha = float(alpha)
    def fit(self, X, y):
        self.X = np.asarray(X, float); self.d = self.X.shape[1]
        self.model = Lasso(alpha=self.alpha, fit_intercept=True, max_iter=10000, tol=1e-8).fit(self.X, np.asarray(y, float))
        return self
    def predict_with_gradient(self, Q):
        Q = np.asarray(Q, float)
        p = self.model.predict(Q)
        g = np.tile(np.asarray(self.model.coef_, float), (len(Q), 1))
        return p, g, {}


def _fit_eval_model(method, cfg, X, y, Q):
    d = X.shape[1]
    min1 = _stage1_min_points(d)
    if method == 'BBLR':
        return BBLRND(cfg['r'], cfg['eps'], min_stage1=min1).fit(X, y).predict_with_gradient(Q)
    if method == 'LLR':
        return LocalPolynomialND(1, r=cfg['r'], min_points=min1).fit(X, y).predict_with_gradient(Q)
    if method == 'LL_ALASSO':
        return AdaptiveLocalLinearLassoND(
            r=cfg['r'], lambda_frac=cfg['lambda_frac'], min_points=min1,
            adaptive_gamma=1.0, tau_frac=0.05, tol=1e-8, max_iter=2000,
        ).fit(X, y).predict_with_gradient(Q)
    if method == 'LASSO':
        return GlobalLassoND(cfg['alpha']).fit(X, y).predict_with_gradient(Q)
    if method == 'OLS':
        return GlobalLinearND().fit(X, y).predict_with_gradient(Q)
    raise ValueError(method)


def select_methods(X, y, folds, repeats, seed,
                   eps_mults=(0.0, 0.5, 1.0, 1.5, 2.0),
                   lambda_fracs=(0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.70)):
    """Outcome-CV tuning for BBLR, LLR, LL-ALASSO, and global Lasso.

    The local sparse benchmark jointly selects locality radius and penalty fraction.
    lambda_frac is relative to each query's weighted adaptive-Lasso lambda_max, which
    keeps the penalty scale comparable across local neighborhoods and dimensions.
    """
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, d = X.shape
    min1 = _stage1_min_points(d)
    radii = _candidate_radii(X, d)
    bcfg = [{'r': r, 'eps': em*r} for r in radii for em in eps_mults]
    lcfg = [{'r': r} for r in radii]
    acfg = [{'r': r, 'lambda_frac': lf} for r in radii for lf in lambda_fracs]
    gcfg = [{'alpha': a} for a in (0.0005, 0.002, 0.01, 0.05, 0.2)]
    methods = {'BBLR': bcfg, 'LLR': lcfg, 'LL_ALASSO': acfg, 'LASSO': gcfg}
    sse = {(m, i): 0.0 for m, cfgs in methods.items() for i in range(len(cfgs))}
    nobs = 0
    idx_b = {(float(c['r']), float(c['eps'])): i for i, c in enumerate(bcfg)}
    idx_l = {float(c['r']): i for i, c in enumerate(lcfg)}
    idx_a = {(float(c['r']), float(c['lambda_frac'])): i for i, c in enumerate(acfg)}

    for tr, va in fold_iterator(n, folds, repeats, seed):
        nobs += len(va)
        Xt, yt = X[tr], y[tr]
        Xv, yv = X[va], y[va]
        for r in radii:
            # Exact zero-stage BBLR and LLR share the same fit.
            try:
                pl, _, _ = LocalPolynomialND(1, r=r, min_points=min1).fit(Xt, yt).predict_with_gradient(Xv)
                val = _sse(pl, yv)
                sse[('LLR', idx_l[float(r)])] += val
                sse[('BBLR', idx_b[(float(r), 0.0)])] += val
            except Exception:
                sse[('LLR', idx_l[float(r)])] = np.inf
                sse[('BBLR', idx_b[(float(r), 0.0)])] = np.inf
            # Positive Stage 2 candidates.
            try:
                bm = BBLRND(r, 1.0, min_stage1=min1).fit(Xt, yt)
                for em in eps_mults:
                    if em == 0:
                        continue
                    eps = float(em*r)
                    pb, _, _ = bm.predict_with_gradient_eps(Xv, eps)
                    sse[('BBLR', idx_b[(float(r), eps)])] += _sse(pb, yv)
            except Exception:
                for em in eps_mults:
                    if em != 0:
                        sse[('BBLR', idx_b[(float(r), float(em*r))])] = np.inf
            # Local adaptive-Lasso path: all penalty fractions from one local path/query.
            try:
                am = AdaptiveLocalLinearLassoND(
                    r=r, lambda_frac=0.1, min_points=min1,
                    adaptive_gamma=1.0, tau_frac=0.05, tol=1e-8, max_iter=2000,
                ).fit(Xt, yt)
                pred_path, _, _ = am.predict_path(Xv, lambda_fracs)
                for lf in lambda_fracs:
                    sse[('LL_ALASSO', idx_a[(float(r), float(lf))])] += _sse(pred_path[float(lf)], yv)
            except Exception:
                for lf in lambda_fracs:
                    sse[('LL_ALASSO', idx_a[(float(r), float(lf))])] = np.inf
        # Global Lasso path kept from the production comparison.
        for i, cfg in enumerate(gcfg):
            try:
                p, _, _ = GlobalLassoND(cfg['alpha']).fit(Xt, yt).predict_with_gradient(Xv)
                sse[('LASSO', i)] += _sse(p, yv)
            except Exception:
                sse[('LASSO', i)] = np.inf

    selected = []
    for m, cfgs in methods.items():
        vals = []
        for i, cfg in enumerate(cfgs):
            risk = math.sqrt(sse[(m, i)] / nobs) if np.isfinite(sse[(m, i)]) else np.inf
            # Deterministic tie-break only; exact ties are rare. No one-SE sparsity preference.
            vals.append((risk, i, cfg))
        vals.sort(key=lambda z: (z[0], tuple(z[2].get(k, np.inf) for k in ('lambda_frac','eps','r','alpha'))))
        risk, _, cfg = vals[0]
        selected.append({'method': m, **cfg, 'cv_rmse': risk})
    selected.append({'method': 'OLS', 'cv_rmse': np.nan})
    return selected


def _nuisance_metrics(gh, active_dim):
    gh = np.asarray(gh, float); d = gh.shape[1]; a = int(active_dim); q = d-a
    if q <= 0:
        return {
            'nuisance_dim': 0, 'nuisance_component_rmse': np.nan,
            'nuisance_vector_rmse': np.nan, 'nuisance_vector_rmse_per_sqrt_dim': np.nan,
            'nuisance_zero_share': np.nan, 'nuisance_scaled_share_gt_010': np.nan,
        }
    z = gh[:, a:]
    norms = np.sqrt(np.sum(z*z, axis=1)); scale = math.sqrt(q)
    return {
        'nuisance_dim': int(q),
        'nuisance_component_rmse': float(np.sqrt(np.mean(z*z))),
        'nuisance_vector_rmse': float(np.sqrt(np.mean(norms*norms))),
        'nuisance_vector_rmse_per_sqrt_dim': float(np.sqrt(np.mean(norms*norms))/scale),
        'nuisance_zero_share': float(np.mean(np.abs(z) <= 1e-12)),
        'nuisance_scaled_share_gt_010': float(np.mean((norms/scale) > 0.10)),
    }


def evaluate_rep(cell: dict, rep: int):
    d = int(cell['d']); a = int(cell['active_dim'])
    n = int(cell['n_dev']); nt = int(cell['n_test']); noise = float(cell['noise_sd'])
    seed = int(cell['base_seed']) + 1000003*int(rep)
    dgp = str(cell['dgp']); design = str(cell['design'])
    X, y, _, _, _ = generate_sample(dgp, n, d, a, noise, seed, design)
    Xt, yt, mut, gt, at = generate_sample(dgp, nt, d, a, noise, seed+500000001, design)
    alpha0 = reference_alpha_truth(dgp, d, a)
    q0 = np.zeros((1,d), float)
    selected = select_methods(X, y, int(cell['cv_folds']), int(cell['cv_repeats']), seed+1717)
    rows = []
    for sel in selected:
        m = sel['method']
        cfg = {k: sel[k] for k in ('r','eps','lambda_frac','alpha') if k in sel and pd.notna(sel[k])}
        try:
            p, gh, diag = _fit_eval_model(m, cfg, X, y, Xt)
            ah = p - np.sum(Xt*gh, axis=1)
            p0, _, _ = _fit_eval_model(m, cfg, X, y, q0)
            grad_err = gh-gt
            if a > 0:
                active_err = grad_err[:,:a]
                active_rmse = float(np.sqrt(np.mean(np.sum(active_err*active_err,axis=1))))
                active_zero_share = float(np.mean(np.abs(gh[:,:a]) <= 1e-12))
            else:
                active_rmse = np.nan; active_zero_share = np.nan
            row = {
                **cell, 'rep': int(rep), 'seed': int(seed), 'method': m, **cfg,
                'cv_rmse': sel.get('cv_rmse', np.nan),
                'test_mean_rmse': _rmse(p, mut),
                'gradient_vector_rmse': float(np.sqrt(np.mean(np.sum(grad_err*grad_err,axis=1)))),
                'active_gradient_vector_rmse': active_rmse,
                'tangent_alpha_rmse': _rmse(ah, at),
                'reference_alpha0_truth': float(alpha0),
                'reference_alpha0_hat': float(p0[0]),
                'reference_alpha0_abs_error': abs(float(p0[0])-float(alpha0)),
                'joint_coordinate_rmse_scale_specific': float(np.sqrt(np.mean((ah-at)**2 + np.sum(grad_err*grad_err,axis=1)))),
                'slope_zero_share': float(np.mean(np.abs(gh) <= 1e-12)),
                'active_zero_share': active_zero_share,
                'selected_positive_local_penalty': float(m=='LL_ALASSO' and cfg.get('lambda_frac',0)>0),
            }
            row.update(_nuisance_metrics(gh,a))
            rows.append(row)
        except Exception as exc:
            rows.append({**cell,'rep':int(rep),'seed':int(seed),'method':m,**cfg,'cv_rmse':sel.get('cv_rmse',np.nan),'error':repr(exc)})
    return pd.DataFrame(rows)
