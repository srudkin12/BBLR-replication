from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso

from .nd_core import (
    BBLRND,
    LocalPolynomialND,
    GlobalLinearND,
    RidgeLinearND,
    GlobalQuadraticND,
    radius_grid_from_knn,
    quadratic_ncoef,
)
from .dgps import generate_sample, reference_alpha_truth


def _sse(a, b):
    z = np.asarray(a) - np.asarray(b)
    return float(np.sum(z * z))


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _vector_rmse(est, truth):
    e = np.asarray(est, float) - np.asarray(truth, float)
    if e.ndim != 2 or e.shape[1] == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.sum(e * e, axis=1))))


def fold_iterator(n, folds, repeats, seed):
    ss = np.random.SeedSequence(int(seed))
    kids = ss.spawn(int(repeats))
    for ch in kids:
        rs = int(ch.generate_state(1)[0])
        kf = KFold(n_splits=int(folds), shuffle=True, random_state=rs)
        for tr, va in kf.split(np.arange(n)):
            yield tr, va


def _k_targets(n, d, degree=1):
    n = int(n)
    d = int(d)
    cap = max(d + 5, int(n * 0.75))
    if degree == 1:
        raw = [
            max(40, 4 * d + 16),
            max(80, 8 * d + 32),
            max(160, 16 * d + 64),
            max(320, 32 * d + 128),
        ]
    else:
        p = quadratic_ncoef(d)
        raw = [max(80, p + 20), max(160, 2 * p + 40), max(280, 4 * p + 60), max(420, 6 * p + 100)]
    out = sorted(set(min(cap, int(k)) for k in raw if min(cap, int(k)) >= d + 5))
    return out


def _candidate_radii(X, d):
    return radius_grid_from_knn(X, _k_targets(len(X), d, 1))


def _stage1_min_points(d):
    """Common Stage-1/LLR safeguard enforcing BBLR(r,0)=LLR(r)."""
    return max(int(d) + 5, 15)


class GlobalLassoND:
    def __init__(self, alpha=0.01):
        self.alpha = float(alpha)

    def fit(self, X, y):
        self.X = np.asarray(X, float)
        self.d = self.X.shape[1]
        self.model = Lasso(alpha=self.alpha, fit_intercept=True, max_iter=10000, tol=1e-8).fit(self.X, np.asarray(y, float))
        return self

    def predict_with_gradient(self, Q):
        Q = np.asarray(Q, float)
        pred = self.model.predict(Q)
        grad = np.tile(np.asarray(self.model.coef_, float), (len(Q), 1))
        return pred, grad, {}


class MeanOnlyND:
    def fit(self, X, y):
        self.d = np.asarray(X).shape[1]
        self.mean = float(np.mean(y))
        return self

    def predict_with_gradient(self, Q):
        Q = np.asarray(Q, float)
        return np.full(len(Q), self.mean), np.zeros((len(Q), self.d)), {}


def _fit_eval_model(method, cfg, X, y, Q, active_dim=None):
    if method in ("BBLR", "BBLR_ORACLE_ACTIVE"):
        Xa = X if method == "BBLR" else X[:, :active_dim]
        Qa = Q if method == "BBLR" else Q[:, :active_dim]
        min1 = _stage1_min_points(Xa.shape[1])
        mdl = BBLRND(cfg["r"], cfg["eps"], min_stage1=min1).fit(Xa, y)
        p, g, diag = mdl.predict_with_gradient(Qa)
        if method == "BBLR_ORACLE_ACTIVE":
            full = np.zeros((len(Q), X.shape[1]))
            full[:, :active_dim] = g
            g = full
        return p, g, diag
    if method in ("LLR", "LLR_MATCHED_BBLR_R"):
        return LocalPolynomialND(1, r=cfg["r"], min_points=_stage1_min_points(X.shape[1])).fit(X, y).predict_with_gradient(Q)
    if method == "AKLLR":
        return LocalPolynomialND(1, k=int(cfg["k"])).fit(X, y).predict_with_gradient(Q)
    if method == "AKLP2":
        return LocalPolynomialND(2, k=int(cfg["k"])).fit(X, y).predict_with_gradient(Q)
    if method == "OLS":
        return GlobalLinearND().fit(X, y).predict_with_gradient(Q)
    if method == "OLS_ORACLE_ACTIVE":
        p, g, _ = GlobalLinearND().fit(X[:, :active_dim], y).predict_with_gradient(Q[:, :active_dim])
        full = np.zeros((len(Q), X.shape[1]))
        full[:, :active_dim] = g
        return p, full, {}
    if method == "RIDGE":
        return RidgeLinearND(cfg["alpha"]).fit(X, y).predict_with_gradient(Q)
    if method == "LASSO":
        return GlobalLassoND(cfg["alpha"]).fit(X, y).predict_with_gradient(Q)
    if method == "GQ":
        return GlobalQuadraticND().fit(X, y).predict_with_gradient(Q)
    if method == "MEAN_ONLY":
        return MeanOnlyND().fit(X, y).predict_with_gradient(Q)
    raise ValueError(method)

def select_methods(X, y, active_dim, folds, repeats, seed, eps_mults=(0.0, 0.5, 1.0, 1.5, 2.0)):
    n, d = X.shape
    min1 = _stage1_min_points(d)
    radii = _candidate_radii(X, d)
    ks1 = _k_targets(n, d, 1)
    ks2 = _k_targets(n, d, 2)
    bcfg = [{"r": r, "eps": e * r} for r in radii for e in eps_mults]
    methods = {
        "BBLR": bcfg,
        "LLR": [{"r": r} for r in radii],
        "AKLLR": [{"k": k} for k in ks1],
        "AKLP2": [{"k": k} for k in ks2],
        "RIDGE": [{"alpha": a} for a in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)],
        "LASSO": [{"alpha": a} for a in (0.0005, 0.002, 0.01, 0.05, 0.2)],
    }
    if 0 < active_dim < d:
        r_or = _candidate_radii(X[:, :active_dim], active_dim)
        methods["BBLR_ORACLE_ACTIVE"] = [{"r": r, "eps": e * r} for r in r_or for e in eps_mults]
    else:
        r_or = []

    sse = {(m, i): 0.0 for m, cfgs in methods.items() for i in range(len(cfgs))}
    nobs = 0
    idx_b = {(float(c["r"]), float(c["eps"])): i for i, c in enumerate(methods["BBLR"])}
    idx_llr = {float(c["r"]): i for i, c in enumerate(methods["LLR"])}
    idx_or = {(float(c["r"]), float(c["eps"])): i for i, c in enumerate(methods.get("BBLR_ORACLE_ACTIVE", []))}

    for tr, va in fold_iterator(n, folds, repeats, seed):
        nobs += len(va)
        Xt, yt = X[tr], y[tr]
        Xv = X[va]

        for r in radii:
            try:
                pl, _, _ = LocalPolynomialND(1, r=r, min_points=min1).fit(Xt, yt).predict_with_gradient(Xv)
                val = _sse(pl, y[va])
                sse[("LLR", idx_llr[float(r)])] += val
                sse[("BBLR", idx_b[(float(r), 0.0)])] += val
            except Exception:
                sse[("LLR", idx_llr[float(r)])] = np.inf
                sse[("BBLR", idx_b[(float(r), 0.0)])] = np.inf
            try:
                bm = BBLRND(r, 1.0, min_stage1=min1).fit(Xt, yt)
                for em in eps_mults:
                    if em == 0:
                        continue
                    eps = float(em * r)
                    pb, _, _ = bm.predict_with_gradient_eps(Xv, eps)
                    sse[("BBLR", idx_b[(float(r), eps)])] += _sse(pb, y[va])
            except Exception:
                for em in eps_mults:
                    if em != 0:
                        sse[("BBLR", idx_b[(float(r), float(em * r))])] = np.inf

        if 0 < active_dim < d:
            Xta = Xt[:, :active_dim]
            Xva = Xv[:, :active_dim]
            min1_or = _stage1_min_points(active_dim)
            for r in r_or:
                try:
                    pl, _, _ = LocalPolynomialND(1, r=r, min_points=min1_or).fit(Xta, yt).predict_with_gradient(Xva)
                    sse[("BBLR_ORACLE_ACTIVE", idx_or[(float(r), 0.0)])] += _sse(pl, y[va])
                except Exception:
                    sse[("BBLR_ORACLE_ACTIVE", idx_or[(float(r), 0.0)])] = np.inf
                try:
                    bm = BBLRND(r, 1.0, min_stage1=min1_or).fit(Xta, yt)
                    for em in eps_mults:
                        if em == 0:
                            continue
                        eps = float(em * r)
                        pb, _, _ = bm.predict_with_gradient_eps(Xva, eps)
                        sse[("BBLR_ORACLE_ACTIVE", idx_or[(float(r), eps)])] += _sse(pb, y[va])
                except Exception:
                    for em in eps_mults:
                        if em != 0:
                            sse[("BBLR_ORACLE_ACTIVE", idx_or[(float(r), float(em * r))])] = np.inf

        for m in ("AKLLR", "AKLP2", "RIDGE", "LASSO"):
            for i, cfg in enumerate(methods[m]):
                try:
                    p, _, _ = _fit_eval_model(m, cfg, Xt, yt, Xv, active_dim)
                    sse[(m, i)] += _sse(p, y[va])
                except Exception:
                    sse[(m, i)] = np.inf

    selected = []
    for m, cfgs in methods.items():
        vals = []
        for i, cfg in enumerate(cfgs):
            risk = math.sqrt(sse[(m, i)] / nobs) if np.isfinite(sse[(m, i)]) else np.inf
            vals.append((risk, i, cfg))
        vals.sort(key=lambda z: (z[0], tuple(z[2].get(k, np.inf) for k in ("eps", "r", "k", "alpha"))))
        risk, i, cfg = vals[0]
        selected.append({"method": m, **cfg, "cv_rmse": risk})

    # Conditional matched-radius Stage-2 comparator: do not retune.
    bsel = next(z for z in selected if z["method"] == "BBLR")
    selected.append({
        "method": "LLR_MATCHED_BBLR_R",
        "r": float(bsel["r"]),
        "matched_bblr_eps": float(bsel["eps"]),
        "matched_bblr_eps_over_r": float(bsel["eps"] / bsel["r"]),
        "cv_rmse": np.nan,
    })

    selected.extend([
        {"method": "OLS", "cv_rmse": np.nan},
        {"method": "GQ", "cv_rmse": np.nan},
    ])
    if 0 < active_dim < d:
        selected.append({"method": "OLS_ORACLE_ACTIVE", "cv_rmse": np.nan})
    if active_dim == 0:
        selected.append({"method": "MEAN_ONLY", "cv_rmse": np.nan})
    return selected


def _nuisance_metrics(gh, active_dim):
    d = gh.shape[1]
    a = int(active_dim)
    q = d - a
    empty = {
        "nuisance_dim": 0,
        "nuisance_component_rmse": np.nan,
        "nuisance_vector_rmse": np.nan,
        "nuisance_vector_rmse_per_sqrt_dim": np.nan,
        "nuisance_mean_norm": np.nan,
        "nuisance_median_norm": np.nan,
        "nuisance_q90_norm": np.nan,
        "nuisance_q95_norm": np.nan,
        "nuisance_mean_norm_per_sqrt_dim": np.nan,
        "nuisance_median_norm_per_sqrt_dim": np.nan,
        "nuisance_q90_norm_per_sqrt_dim": np.nan,
        "nuisance_q95_norm_per_sqrt_dim": np.nan,
        "nuisance_share_gt_005": np.nan,
        "nuisance_share_gt_010": np.nan,
        "nuisance_share_gt_020": np.nan,
        "nuisance_scaled_share_gt_005": np.nan,
        "nuisance_scaled_share_gt_010": np.nan,
        "nuisance_scaled_share_gt_020": np.nan,
    }
    if q <= 0:
        return empty
    z = np.asarray(gh[:, a:], float)
    norms = np.sqrt(np.sum(z * z, axis=1))
    scale = math.sqrt(q)
    sn = norms / scale
    return {
        "nuisance_dim": int(q),
        "nuisance_component_rmse": float(np.sqrt(np.mean(z * z))),
        "nuisance_vector_rmse": float(np.sqrt(np.mean(norms * norms))),
        "nuisance_vector_rmse_per_sqrt_dim": float(np.sqrt(np.mean(norms * norms)) / scale),
        "nuisance_mean_norm": float(np.mean(norms)),
        "nuisance_median_norm": float(np.median(norms)),
        "nuisance_q90_norm": float(np.quantile(norms, 0.90)),
        "nuisance_q95_norm": float(np.quantile(norms, 0.95)),
        "nuisance_mean_norm_per_sqrt_dim": float(np.mean(sn)),
        "nuisance_median_norm_per_sqrt_dim": float(np.median(sn)),
        "nuisance_q90_norm_per_sqrt_dim": float(np.quantile(sn, 0.90)),
        "nuisance_q95_norm_per_sqrt_dim": float(np.quantile(sn, 0.95)),
        "nuisance_share_gt_005": float(np.mean(norms > 0.05)),
        "nuisance_share_gt_010": float(np.mean(norms > 0.10)),
        "nuisance_share_gt_020": float(np.mean(norms > 0.20)),
        "nuisance_scaled_share_gt_005": float(np.mean(sn > 0.05)),
        "nuisance_scaled_share_gt_010": float(np.mean(sn > 0.10)),
        "nuisance_scaled_share_gt_020": float(np.mean(sn > 0.20)),
    }

def evaluate_rep(cell: dict, rep: int):
    d = int(cell["d"])
    a = int(cell["active_dim"])
    n = int(cell["n_dev"])
    nt = int(cell["n_test"])
    noise = float(cell["noise_sd"])
    seed = int(cell["base_seed"]) + 1000003 * int(rep)
    dgp = str(cell["dgp"])
    design = str(cell["design"])

    X, y, mu, gtrue, atrue = generate_sample(dgp, n, d, a, noise, seed, design)
    Xt, yt, mut, gt, at = generate_sample(dgp, nt, d, a, noise, seed + 500000001, design)
    alpha0 = reference_alpha_truth(dgp, d, a)
    q0 = np.zeros((1, d), dtype=float)
    origin_nn_distance = float(np.min(np.sqrt(np.sum(X * X, axis=1))))

    selected = select_methods(X, y, a, int(cell["cv_folds"]), int(cell["cv_repeats"]), seed + 1717)
    out = []
    for sel in selected:
        m = sel["method"]
        cfg = {k: sel[k] for k in ("r", "eps", "k", "alpha") if k in sel and pd.notna(sel[k])}
        try:
            p, gh, diag = _fit_eval_model(m, cfg, X, y, Xt, a)
            ah = p - np.sum(Xt * gh, axis=1)
            p0, g0, _ = _fit_eval_model(m, cfg, X, y, q0, a)
            alpha0_hat = float(p0[0])

            grad_err = gh - gt
            grad_vector_rmse = float(np.sqrt(np.mean(np.sum(grad_err * grad_err, axis=1))))
            if a > 0:
                active_err = grad_err[:, :a]
                active_vector_rmse = float(np.sqrt(np.mean(np.sum(active_err * active_err, axis=1))))
                active_component_rmse = float(np.sqrt(np.mean(active_err * active_err)))
            else:
                active_vector_rmse = np.nan
                active_component_rmse = np.nan

            row = {
                **cell,
                "rep": rep,
                "seed": seed,
                "method": m,
                **cfg,
                "cv_rmse": sel.get("cv_rmse", np.nan),
                "matched_bblr_eps": sel.get("matched_bblr_eps", np.nan),
                "matched_bblr_eps_over_r": sel.get("matched_bblr_eps_over_r", np.nan),
                "test_y_rmse": _rmse(p, yt),
                "test_mean_rmse": _rmse(p, mut),
                "gradient_component_rmse": _rmse(gh, gt),
                "gradient_vector_rmse": grad_vector_rmse,
                "active_gradient_component_rmse": active_component_rmse,
                "active_gradient_vector_rmse": active_vector_rmse,
                "tangent_alpha_rmse": _rmse(ah, at),
                "tangent_alpha_bias": float(np.mean(ah - at)),
                "tangent_alpha_hat_mean": float(np.mean(ah)),
                "tangent_alpha_hat_sd": float(np.std(ah)),
                "reference_alpha0_truth": alpha0,
                "reference_alpha0_hat": alpha0_hat,
                "reference_alpha0_abs_error": abs(alpha0_hat - alpha0),
                "reference_alpha0_sq_error": (alpha0_hat - alpha0) ** 2,
                "origin_nn_distance": origin_nn_distance,
                "coefficient_field_vector_rmse": float(np.sqrt(np.mean((ah - at) ** 2 + np.sum(grad_err * grad_err, axis=1)))),
                "stage2_positive": float(m.startswith("BBLR") and cfg.get("eps", 0) > 0),
                "selected_eps_over_r": float(cfg.get("eps", 0.0) / cfg["r"]) if m.startswith("BBLR") and "r" in cfg else np.nan,
            }
            row.update(_nuisance_metrics(gh, a))
            out.append(row)
        except Exception as e:
            out.append({
                **cell,
                "rep": rep,
                "seed": seed,
                "method": m,
                **cfg,
                "cv_rmse": sel.get("cv_rmse", np.nan),
                "matched_bblr_eps": sel.get("matched_bblr_eps", np.nan),
                "matched_bblr_eps_over_r": sel.get("matched_bblr_eps_over_r", np.nan),
                "error": repr(e),
            })
    return pd.DataFrame(out)
