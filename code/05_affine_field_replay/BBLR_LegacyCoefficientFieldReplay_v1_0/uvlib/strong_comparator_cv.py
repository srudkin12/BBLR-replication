"""Repeated-CV tuning for strong one-dimensional BBLR comparators.

All tuning uses observed outcomes only.  True means and derivatives are exposed
only after a configuration has been selected and refitted on the complete
development sample.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline
from sklearn.model_selection import KFold

from .univariate import DGP_REGISTRY, differentiable_mask, rmse
from .univariate_fast import FastBBLR1D, KNNLocalPolynomial1D
from .univariate_support import (
    ComponentwiseSmoothingSpline1D, ComponentwiseKNNLocalPolynomial1D,
)
from .repeated_cv import evaluate_bblr_cv_surface

Array = np.ndarray


def _fold_iterator(n: int, folds: int, repeats: int, seed: int):
    x = np.empty(n)
    for rr in range(int(repeats)):
        rs = int(np.random.SeedSequence([int(seed), 9001, rr]).generate_state(1, dtype=np.uint32)[0])
        kf = KFold(n_splits=int(folds), shuffle=True, random_state=rs)
        for tri, vai in kf.split(x):
            yield rr, tri, vai


def _strict_xy(x: Array, y: Array):
    order = np.argsort(x, kind="mergesort")
    xs = np.asarray(x, float)[order]
    ys = np.asarray(y, float)[order]
    ux, inv, counts = np.unique(xs, return_inverse=True, return_counts=True)
    if len(ux) != len(xs):
        ys = np.bincount(inv, weights=ys) / counts
        xs = ux
    return xs, ys


def _safe_rmse(pred: Array, y: Array) -> float:
    pred = np.asarray(pred, float)
    y = np.asarray(y, float)
    if pred.size != y.size or not np.all(np.isfinite(pred)):
        return np.inf
    return float(np.sqrt(np.mean((pred-y)**2)))


def _predict_config(method: str, config: dict, x_train: Array, y_train: Array, x_query: Array,
                    *, min_stage1: int, min_stage2: int, component_min_points: int):
    method = method.upper()
    diag = {}
    if method == "BBLR":
        m = FastBBLR1D(
            r=float(config["r"]), eps=float(config["eps"]),
            min_points_stage1=int(min_stage1), min_centres_stage2=int(min_stage2),
        ).fit(x_train, y_train)
        pred, slope, diag = m.predict_with_slope(x_query)
    elif method == "LLR":
        m = FastBBLR1D(
            r=float(config["r"]), eps=0.0,
            min_points_stage1=int(min_stage1), min_centres_stage2=int(min_stage2),
        ).fit(x_train, y_train)
        pred, slope, diag = m.predict_with_slope(x_query)
    elif method in ("AKLLR", "AKLP2"):
        degree = 1 if method == "AKLLR" else 2
        m = KNNLocalPolynomial1D(k=int(config["k"]), degree=degree).fit(x_train, y_train)
        pred, slope, diag = m.predict_with_slope(x_query)
    elif method == "SPLINE":
        xs, ys = _strict_xy(x_train, y_train)
        sp = make_smoothing_spline(xs, ys, lam=float(config["lam"]))
        pred = np.asarray(sp(x_query), float)
        slope = np.asarray(sp.derivative()(x_query), float)
    elif method in ("CAKLLR", "CAKLP2"):
        degree = 1 if method == "CAKLLR" else 2
        m = ComponentwiseKNNLocalPolynomial1D(
            k=int(config["k"]), degree=degree, gap_factor=float(config["gap_factor"]),
            min_component_points=int(component_min_points),
        ).fit(x_train, y_train)
        pred, slope, diag = m.predict_with_slope(x_query)
    elif method == "CSPLINE":
        m = ComponentwiseSmoothingSpline1D(
            lam=float(config["lam"]), gap_factor=float(config["gap_factor"]),
            min_component_points=int(component_min_points),
        ).fit(x_train, y_train)
        pred, slope, diag = m.predict_with_slope(x_query)
    else:
        raise ValueError(f"unknown method {method}")
    return np.asarray(pred, float), np.asarray(slope, float), diag


def _method_configs(method: str, *, r_grid, eps_grid, k_grid, lambda_grid, gap_factor_grid):
    method = method.upper()
    if method == "BBLR":
        return [{"r":float(r), "eps":float(e)} for r in r_grid for e in eps_grid]
    if method == "LLR":
        return [{"r":float(r), "eps":0.0} for r in r_grid]
    if method in ("AKLLR", "AKLP2"):
        return [{"k":int(k)} for k in k_grid]
    if method == "SPLINE":
        return [{"lam":float(lam)} for lam in lambda_grid]
    if method in ("CAKLLR", "CAKLP2"):
        return [{"k":int(k), "gap_factor":float(gf)} for gf in gap_factor_grid for k in k_grid]
    if method == "CSPLINE":
        return [{"lam":float(lam), "gap_factor":float(gf)} for gf in gap_factor_grid for lam in lambda_grid]
    raise ValueError(method)


def repeated_cv_risk_surface(
    x_dev: Array, y_dev: Array, *, methods: Sequence[str],
    r_grid: Sequence[float], eps_grid: Sequence[float], k_grid: Sequence[int],
    lambda_grid: Sequence[float], gap_factor_grid: Sequence[float],
    folds: int, repeats: int, seed: int,
    min_stage1: int = 15, min_stage2: int = 1, component_min_points: int = 15,
) -> pd.DataFrame:
    """Return outcome-only repeated-CV risk for every method/configuration."""
    x_dev = np.asarray(x_dev, float).reshape(-1)
    y_dev = np.asarray(y_dev, float).reshape(-1)
    methods = [m.upper() for m in methods]
    rows = []

    # BBLR is highly optimized and also yields the nested LLR risk surface.
    if "BBLR" in methods or "LLR" in methods:
        b = evaluate_bblr_cv_surface(
            x_dev, y_dev, r_grid=r_grid, eps_grid=eps_grid,
            min_stage1=min_stage1, min_stage2=min_stage2,
            folds=folds, repeats=repeats, seed=seed,
            stability_grid=np.linspace(float(np.min(x_dev)), float(np.max(x_dev)), 41),
        )
        if "BBLR" in methods:
            for row in b.itertuples(index=False):
                rows.append({
                    "method":"BBLR", "r":float(row.r), "eps":float(row.eps),
                    "k":np.nan, "lam":np.nan, "gap_factor":np.nan,
                    "repeated_cv_rmse":float(row.repeated_cv_rmse),
                })
        if "LLR" in methods:
            for row in b[b["eps"].eq(0.0)].itertuples(index=False):
                rows.append({
                    "method":"LLR", "r":float(row.r), "eps":0.0,
                    "k":np.nan, "lam":np.nan, "gap_factor":np.nan,
                    "repeated_cv_rmse":float(row.repeated_cv_rmse),
                })

    other = [m for m in methods if m not in ("BBLR", "LLR")]
    if other:
        configs = {m:_method_configs(m, r_grid=r_grid, eps_grid=eps_grid, k_grid=k_grid,
                                    lambda_grid=lambda_grid, gap_factor_grid=gap_factor_grid)
                   for m in other}
        sse = {(m,i):0.0 for m in other for i in range(len(configs[m]))}
        nobs = 0
        for _, tri, vai in _fold_iterator(len(x_dev), folds, repeats, seed):
            xt, yt = x_dev[tri], y_dev[tri]
            xv, yv = x_dev[vai], y_dev[vai]
            nobs += len(vai)
            for m in other:
                for i,cfg in enumerate(configs[m]):
                    try:
                        pred, _, _ = _predict_config(
                            m, cfg, xt, yt, xv, min_stage1=min_stage1,
                            min_stage2=min_stage2, component_min_points=component_min_points,
                        )
                        if np.all(np.isfinite(pred)):
                            sse[(m,i)] += float(np.sum((pred-yv)**2))
                        else:
                            sse[(m,i)] = np.inf
                    except Exception:
                        sse[(m,i)] = np.inf
        for m in other:
            for i,cfg in enumerate(configs[m]):
                risk = sse[(m,i)]
                rows.append({
                    "method":m,
                    "r":cfg.get("r",np.nan), "eps":cfg.get("eps",np.nan),
                    "k":cfg.get("k",np.nan), "lam":cfg.get("lam",np.nan),
                    "gap_factor":cfg.get("gap_factor",np.nan),
                    "repeated_cv_rmse":float(np.sqrt(risk/nobs)) if np.isfinite(risk) else np.inf,
                })
    return pd.DataFrame(rows)


def select_minimum_risk(surface: pd.DataFrame) -> pd.DataFrame:
    """One outcome-selected configuration per method with deterministic ties."""
    chosen = []
    for method,g in surface.groupby("method", sort=False):
        z = g.copy()
        # NaN tuning dimensions sort last.  Smaller smoothing/config values are
        # deterministic tie-breakers only; prediction risk remains primary.
        z["_gap"] = z["gap_factor"].replace(np.inf, 1e300)
        z = z.sort_values(
            ["repeated_cv_rmse","eps","r","k","lam","_gap"],
            na_position="last", kind="mergesort",
        )
        chosen.append(z.iloc[0].drop(labels=["_gap"]).to_dict())
    return pd.DataFrame(chosen)


def evaluate_selected(
    selected: pd.DataFrame, x_dev: Array, y_dev: Array, x_test: Array, y_test: Array,
    *, dgp_code: str, guard_width: float = 0.10, n_grid: int = 401,
    min_stage1: int = 15, min_stage2: int = 1, component_min_points: int = 15,
) -> pd.DataFrame:
    """Refit selected configurations and reveal simulation truth afterwards."""
    dgp = DGP_REGISTRY[dgp_code]
    x_grid = np.linspace(dgp.support[0], dgp.support[1], int(n_grid))
    true_mean_test = np.asarray(dgp.mean(x_test), float)
    true_slope_test = np.asarray(dgp.derivative(x_test), float)
    true_slope_grid = np.asarray(dgp.derivative(x_grid), float)
    mask_test = differentiable_mask(np.asarray(x_test,float), dgp, guard_width)
    mask_grid = differentiable_mask(x_grid, dgp, guard_width)
    rows=[]
    for row in selected.to_dict("records"):
        cfg={k:row[k] for k in ("r","eps","k","lam","gap_factor") if k in row and pd.notna(row[k])}
        try:
            pred,slope,diag=_predict_config(
                row["method"],cfg,x_dev,y_dev,x_test,
                min_stage1=min_stage1,min_stage2=min_stage2,
                component_min_points=component_min_points,
            )
            _,slope_grid,_=_predict_config(
                row["method"],cfg,x_dev,y_dev,x_grid,
                min_stage1=min_stage1,min_stage2=min_stage2,
                component_min_points=component_min_points,
            )
            out=dict(row)
            out.update({
                "test_y_rmse":rmse(pred,y_test),
                "test_mean_rmse":rmse(pred,true_mean_test),
                "test_slope_rmse":rmse(slope,true_slope_test,mask_test),
                "grid_slope_rmse":rmse(slope_grid,true_slope_grid,mask_grid),
                "n_support_components":float(diag.get("n_support_components",np.nan)) if isinstance(diag,dict) else np.nan,
                "max_gap_ratio":float(diag.get("max_gap_ratio",np.nan)) if isinstance(diag,dict) else np.nan,
            })
        except Exception:
            out=dict(row)
            out.update({"test_y_rmse":np.nan,"test_mean_rmse":np.nan,"test_slope_rmse":np.nan,
                        "grid_slope_rmse":np.nan,"n_support_components":np.nan,"max_gap_ratio":np.nan})
        rows.append(out)
    return pd.DataFrame(rows)
