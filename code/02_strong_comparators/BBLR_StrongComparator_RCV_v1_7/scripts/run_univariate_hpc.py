#!/usr/bin/env python3
"""Production-oriented parallel Monte Carlo runner for 1D BBLR.

Key design principles
---------------------
1. Hyperparameters are selected using ONLY held-out observed outcome Y RMSE.
2. True conditional-mean and true-slope errors are revealed only afterwards.
3. Every tuning configuration is retained, enabling outcome-RMSE / slope-RMSE
   performance clouds and Pareto-frontier analysis.
4. Parallelism is across Monte Carlo replication chunks; workers are single-
   threaded internally to avoid BLAS oversubscription on large Linux machines.
5. Each chunk writes its own gzip CSV shard. Runs are resumable and the parent
   process never has to hold the complete experiment in RAM.

The fast BBLR and hard-ball LLR implementations exploit exact 1D prefix sums.
Local quadratic, tricube local-linear and smoothing-spline comparators provide
stronger conventional benchmarks.
"""

from __future__ import annotations

# Must be set before importing numpy/scipy/sklearn in worker processes.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import gzip
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.interpolate import make_smoothing_spline
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bblr.univariate import (
    DGP_REGISTRY, SUPPORT_DESIGN_REGISTRY, LocalPolynomial1D, boundary_mask,
    differentiable_mask, generate_sample, rmse, support_design_metadata,
)
from bblr.univariate_fast import (
    FastBBLR1D, FastLocalLinear1D, TricubeLocalLinear1D, KNNLocalPolynomial1D,
)
from bblr.univariate_support import (
    ComponentwiseSmoothingSpline1D, ComponentwiseKNNLocalPolynomial1D,
)


def parse_float_list(s: str) -> List[float]:
    return [float(v) for v in s.split(",") if v.strip()]


def parse_int_list(s: str) -> List[int]:
    return [int(float(v)) for v in s.split(",") if v.strip()]


def parse_str_list(s: str) -> List[str]:
    return [v.strip() for v in s.split(",") if v.strip()]


def stable_seed(base: int, *parts: int) -> int:
    """Collision-resistant deterministic uint32 seed from structured indices."""
    ss = np.random.SeedSequence([int(base), *map(int, parts)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def dgp_complexity(code: str, guard_width: float, n_grid: int = 4001) -> Dict[str, float]:
    """Numerical descriptors of derivative heterogeneity/nonlinearity.

    These descriptors are diagnostics rather than tuning information.  They are
    useful for relating BBLR gains/losses to the shape of the DGP.
    """
    dgp = DGP_REGISTRY[code]
    lo, hi = dgp.support
    x = np.linspace(lo, hi, n_grid)
    b = np.asarray(dgp.derivative(x), float)
    mask = differentiable_mask(x, dgp, guard_width)
    # Curvature is the derivative of the slope surface.  Use a numerical
    # derivative so new DGPs need only provide m and m'.
    curv = np.gradient(b, x)
    if mask.sum() < 3:
        return {
            "slope_sd_uniform": np.nan, "slope_rms_uniform": np.nan,
            "slope_abs_max_uniform": np.nan, "curvature_rms_uniform": np.nan,
            "curvature_abs_mean_uniform": np.nan, "curvature_abs_max_uniform": np.nan,
            "slope_total_variation": np.nan, "roughness_length_uniform": np.nan,
            "mean_sd_uniform": np.nan, "mean_range_uniform": np.nan,
            "n_nondifferentiable_points": len(dgp.nondifferentiable_points),
        }
    bm = b[mask]; cm = curv[mask]
    # Total variation computed over contiguous grid but excluding guard bands;
    # this remains a descriptive diagnostic for non-smooth cases.
    tv = float(np.sum(np.abs(np.diff(b))))
    mm = np.asarray(dgp.mean(x), float)[mask]
    slope_rms = float(np.sqrt(np.mean(bm * bm)))
    curvature_rms = float(np.sqrt(np.mean(cm * cm)))
    return {
        "slope_sd_uniform": float(np.std(bm)),
        "slope_rms_uniform": slope_rms,
        "slope_abs_max_uniform": float(np.max(np.abs(bm))),
        "curvature_rms_uniform": curvature_rms,
        "curvature_abs_mean_uniform": float(np.mean(np.abs(cm))),
        "curvature_abs_max_uniform": float(np.max(np.abs(cm))),
        "slope_total_variation": tv,
        "roughness_length_uniform": float(np.std(bm) / curvature_rms) if curvature_rms > 1e-14 else np.inf,
        "mean_sd_uniform": float(np.std(mm)),
        "mean_range_uniform": float(np.max(mm) - np.min(mm)),
        "n_nondifferentiable_points": int(len(dgp.nondifferentiable_points)),
    }


def _metric_row(method: str, dgp_code: str, design: str, rep: int, n: int,
                noise_sd: float, r: float = np.nan, eps: float = np.nan,
                k: float = np.nan, gap_factor: float = np.nan,
                tuning_value: float = np.nan, tuning_name: str = "",
                min_stage1: float = np.nan, min_stage2: float = np.nan,
                pred_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                pred_test: np.ndarray | None = None, slope_test: np.ndarray | None = None,
                x_test: np.ndarray | None = None, y_test: np.ndarray | None = None,
                pred_grid: np.ndarray | None = None, slope_grid: np.ndarray | None = None,
                x_grid: np.ndarray | None = None, guard_width: float = 0.10,
                diagnostics: Dict[str, np.ndarray] | None = None,
                complexity: Dict[str, float] | None = None) -> Dict[str, object]:
    dgp = DGP_REGISTRY[dgp_code]
    true_mean_test = dgp.mean(x_test)
    true_slope_test = dgp.derivative(x_test)
    true_mean_grid = dgp.mean(x_grid)
    true_slope_grid = dgp.derivative(x_grid)
    mask_test = differentiable_mask(x_test, dgp, guard_width)
    mask_grid = differentiable_mask(x_grid, dgp, guard_width)
    bnd = boundary_mask(x_grid, dgp.support)
    bnd_mask = mask_grid & bnd
    int_mask = mask_grid & (~bnd)

    val_resid = np.asarray(pred_val, float) - np.asarray(y_val, float)
    val_sq = val_resid * val_resid
    val_y_mse = float(np.mean(val_sq))
    val_y_mse_se = float(np.std(val_sq, ddof=1) / np.sqrt(len(val_sq))) if len(val_sq) > 1 else np.nan
    val_y_rmse = float(np.sqrt(val_y_mse))
    test_y_rmse = rmse(pred_test, y_test)
    support_spec = support_design_metadata(design)
    out: Dict[str, object] = {
        "dgp": dgp_code,
        "dgp_label": dgp.label,
        "dgp_family": dgp.family,
        "dgp_parameter_name": dgp.parameter_name,
        "dgp_parameter_value": dgp.parameter_value,
        "dgp_secondary_parameter_name": dgp.secondary_parameter_name,
        "dgp_secondary_parameter_value": dgp.secondary_parameter_value,
        "design": design,
        "design_label": support_spec.label,
        "design_family": support_spec.family,
        "design_parameter_name": support_spec.parameter_name,
        "design_parameter_value": support_spec.parameter_value,
        "design_secondary_parameter_name": support_spec.secondary_parameter_name,
        "design_secondary_parameter_value": support_spec.secondary_parameter_value,
        "rep": int(rep),
        "n_train": int(n),
        "noise_sd": float(noise_sd),
        "method": method,
        "r": float(r) if np.isfinite(r) else np.nan,
        "eps": float(eps) if np.isfinite(eps) else np.nan,
        "k": float(k) if np.isfinite(k) else np.nan,
        "gap_factor": float(gap_factor) if np.isfinite(gap_factor) else (np.inf if np.isinf(gap_factor) else np.nan),
        "tuning_name": tuning_name,
        "tuning_value": float(tuning_value) if np.isfinite(tuning_value) else np.nan,
        "min_stage1": min_stage1,
        "min_stage2": min_stage2,
        # Feasible selection quantity.
        "validation_y_rmse": float(val_y_rmse),
        "validation_y_mse": float(val_y_mse),
        "validation_y_mse_se": float(val_y_mse_se),
        "n_validation": int(len(val_sq)),
        # Independent test outcome performance: x-axis of the proposed map.
        "test_y_rmse": float(test_y_rmse),
        # Simulation-only diagnostics revealed after fitting/tuning.
        "test_mean_rmse": rmse(pred_test, true_mean_test),
        "test_slope_rmse": rmse(slope_test, true_slope_test, mask_test),
        "test_slope_bias": float(np.mean((slope_test - true_slope_test)[mask_test])),
        "grid_mean_rmse": rmse(pred_grid, true_mean_grid),
        "grid_slope_rmse": rmse(slope_grid, true_slope_grid, mask_grid),
        "grid_slope_rmse_interior": rmse(slope_grid, true_slope_grid, int_mask) if int_mask.any() else np.nan,
        "grid_slope_rmse_boundary": rmse(slope_grid, true_slope_grid, bnd_mask) if bnd_mask.any() else np.nan,
        "n_test_slope_scored": int(mask_test.sum()),
        "n_grid_slope_scored": int(mask_grid.sum()),
        # Irreducible-noise-adjusted diagnostic; never used for tuning.
        "test_excess_mse_over_noise": float(max(test_y_rmse**2 - noise_sd**2, 0.0)),
        "selected_by_validation_y": False,
        "selected_by_validation_y_1se_low_smoothing": False,
        "selected_by_validation_y_tol_0p5_low_smoothing": False,
        "selected_by_validation_y_tol_1p0_low_smoothing": False,
        "oracle_slope_within_method": False,
        "oracle_mean_within_method": False,
        "pareto_all_configurations": False,
        "pareto_selected_methods": False,
    }
    # Support-regime diagnostics for the controlled transition family.
    if dgp.family == "support_transition":
        delta = float(dgp.parameter_value)
        centre = float(dgp.secondary_parameter_value)
        # A transition window of +/-2*delta contains most of the slope change.
        near_test = mask_test & (np.abs(x_test - centre) <= 2.0*delta)
        away_test = mask_test & (~near_test)
        left_test = mask_test & (x_test < centre)
        right_test = mask_test & (x_test >= centre)
        err_test = slope_test - true_slope_test
        out["feature_delta"] = delta
        out["feature_centre"] = centre
        out["n_test_near_feature"] = int(near_test.sum())
        out["n_test_away_feature"] = int(away_test.sum())
        out["n_test_left_regime"] = int(left_test.sum())
        out["n_test_right_regime"] = int(right_test.sum())
        out["test_slope_rmse_near_feature"] = (
            float(np.sqrt(np.mean(err_test[near_test]**2))) if near_test.any() else np.nan
        )
        out["test_slope_rmse_away_feature"] = (
            float(np.sqrt(np.mean(err_test[away_test]**2))) if away_test.any() else np.nan
        )
        out["test_slope_rmse_left_regime"] = (
            float(np.sqrt(np.mean(err_test[left_test]**2))) if left_test.any() else np.nan
        )
        out["test_slope_rmse_right_regime"] = (
            float(np.sqrt(np.mean(err_test[right_test]**2))) if right_test.any() else np.nan
        )

    if complexity:
        out.update(complexity)
    if diagnostics:
        if "fallback" in diagnostics:
            out["query_fallback_share"] = float(np.mean(diagnostics["fallback"]))
            out["mean_query_neighbours"] = float(np.mean(diagnostics["neighbour_count"]))
        if "adaptive_bandwidth" in diagnostics:
            ab = np.asarray(diagnostics["adaptive_bandwidth"], float)
            out["mean_adaptive_bandwidth"] = float(np.nanmean(ab))
            out["median_adaptive_bandwidth"] = float(np.nanmedian(ab))
        if "stage2_fallback" in diagnostics:
            out["stage2_fallback_share"] = float(np.mean(diagnostics["stage2_fallback"]))
            out["mean_stage2_centres"] = float(np.mean(diagnostics["stage2_centre_count"]))
            out["stage1_fallback_share"] = float(np.mean(diagnostics["stage1_fallback"]))
            out["mean_stage1_neighbours"] = float(np.mean(diagnostics["stage1_count"]))
        if "n_support_components" in diagnostics:
            out["n_support_components"] = int(diagnostics["n_support_components"])
            out["support_spacing_baseline"] = float(diagnostics["spacing_baseline"])
            out["support_max_observed_gap"] = float(diagnostics["max_observed_gap"])
            out["support_max_gap_ratio"] = float(diagnostics["max_gap_ratio"])
            out["support_split_points"] = diagnostics.get("split_points", "")
    return out


def _pareto_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-dominated mask for minimisation of x and y."""
    n = len(x)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(x[i]) or not np.isfinite(y[i]):
            keep[i] = False
            continue
        dominates = ((x <= x[i]) & (y <= y[i]) & ((x < x[i]) | (y < y[i])))
        dominates[i] = False
        if np.any(dominates):
            keep[i] = False
    return keep


def _low_smoothing_sort(g: pd.DataFrame) -> pd.DataFrame:
    """Deterministic ordering from less to more smoothing/structural pooling."""
    z=g.copy()
    # NaN means parameter is inapplicable; place it after applicable values.
    cols=[]
    for c in ["eps","r","k","tuning_value","gap_factor"]:
        if c in z.columns:
            cols.append(c)
    if not cols:
        return z
    # For gap_factor, larger values imply fewer splits / weaker support
    # adaptation. It is deliberately a final tie-breaker only.
    if "gap_factor" in cols:
        z["_neg_gap_factor"]=-z["gap_factor"].replace(np.inf,1e300)
        cols=[c for c in cols if c!="gap_factor"]+["_neg_gap_factor"]
    return z.sort_values(cols,na_position="last",kind="mergesort")


def _mark_selection_and_pareto(df: pd.DataFrame) -> pd.DataFrame:
    # Every feasible rule uses observed validation Y only. True slope enters
    # only the explicitly-labelled oracle diagnostics after tuning.
    for method, idx in df.groupby("method", sort=False).groups.items():
        g=df.loc[idx]
        base=g.sort_values(["validation_y_rmse","eps","r","k","tuning_value"],
                           na_position="last",kind="mergesort")
        sel=base.index[0]
        df.loc[sel,"selected_by_validation_y"]=True

        # One-SE rule on validation MSE, then choose the least-smoothed eligible
        # configuration. This is an empirically feasible conservative selector.
        mse_min=float(df.loc[sel,"validation_y_mse"])
        mse_se=float(df.loc[sel,"validation_y_mse_se"])
        if np.isfinite(mse_min) and np.isfinite(mse_se):
            eligible=g[g["validation_y_mse"] <= mse_min+mse_se+1e-15]
            if len(eligible):
                df.loc[_low_smoothing_sort(eligible).index[0],
                       "selected_by_validation_y_1se_low_smoothing"]=True

        # Relative prediction-tolerance rules are also feasible and are useful
        # for quantifying how much coefficient smoothing is justified by Y fit.
        best_rmse=float(df.loc[sel,"validation_y_rmse"])
        for tol,col in [(0.005,"selected_by_validation_y_tol_0p5_low_smoothing"),
                        (0.010,"selected_by_validation_y_tol_1p0_low_smoothing")]:
            eligible=g[g["validation_y_rmse"] <= best_rmse*(1.0+tol)+1e-15]
            if len(eligible):
                df.loc[_low_smoothing_sort(eligible).index[0],col]=True

        slope=g["test_slope_rmse"].astype(float)
        if slope.notna().any():
            df.loc[slope.idxmin(),"oracle_slope_within_method"]=True
        mean=g["test_mean_rmse"].astype(float)
        if mean.notna().any():
            df.loc[mean.idxmin(),"oracle_mean_within_method"]=True

    mask=_pareto_mask(df["test_y_rmse"].to_numpy(float),df["test_slope_rmse"].to_numpy(float))
    df.loc[df.index[mask],"pareto_all_configurations"]=True
    chosen=df.index[df["selected_by_validation_y"].astype(bool)]
    if len(chosen):
        g=df.loc[chosen]
        pm=_pareto_mask(g["test_y_rmse"].to_numpy(float),g["test_slope_rmse"].to_numpy(float))
        df.loc[g.index[pm],"pareto_selected_methods"]=True
    return df


def one_replication(*, dgp_code: str, dgp_i: int, design: str, design_i: int,
                    n: int, n_i: int, noise_sd: float, noise_i: int, rep: int,
                    seed_offset: int, n_validation: int, n_test: int, n_grid_eval: int,
                    r_grid: Sequence[float], eps_grid: Sequence[float],
                    spline_lambda_grid: Sequence[float], knn_k_grid: Sequence[int],
                    gap_factor_grid: Sequence[float], component_min_points: int,
                    min_stage1: int,
                    min_stage2_grid: Sequence[int], guard_width: float,
                    comparators: Sequence[str], complexity: Dict[str, float]) -> pd.DataFrame:
    dgp = DGP_REGISTRY[dgp_code]
    lo, hi = dgp.support
    n_val = n if n_validation <= 0 else n_validation
    n_tst = max(2000, n) if n_test <= 0 else n_test

    seed_base_parts = (dgp_i, design_i, n_i, noise_i, rep)
    x_train, y_train, _ = generate_sample(dgp_code, n, noise_sd,
                                           stable_seed(seed_offset, *seed_base_parts, 0), design)
    x_val, y_val, _ = generate_sample(dgp_code, n_val, noise_sd,
                                       stable_seed(seed_offset, *seed_base_parts, 1), design)
    x_test, y_test, _ = generate_sample(dgp_code, n_tst, noise_sd,
                                         stable_seed(seed_offset, *seed_base_parts, 2), design)
    x_grid = np.linspace(lo, hi, n_grid_eval)
    queries = np.concatenate([x_val, x_test, x_grid])
    sv = slice(0, n_val); st = slice(n_val, n_val+n_tst); sg = slice(n_val+n_tst, None)

    rows: List[Dict[str, object]] = []
    comparators = set(comparators)

    def add(method: str, pred: np.ndarray, slope: np.ndarray, **meta):
        diag = meta.pop("diagnostics", None)
        rows.append(_metric_row(
            method=method, dgp_code=dgp_code, design=design, rep=rep, n=n,
            noise_sd=noise_sd, pred_val=pred[sv], y_val=y_val,
            pred_test=pred[st], slope_test=slope[st], x_test=x_test, y_test=y_test,
            pred_grid=pred[sg], slope_grid=slope[sg], x_grid=x_grid,
            guard_width=guard_width, diagnostics=diag, complexity=complexity,
            min_stage1=meta.pop("min_stage1", np.nan),
            min_stage2=meta.pop("min_stage2", np.nan),
            k=meta.pop("k", np.nan), gap_factor=meta.pop("gap_factor", np.nan), **meta
        ))

    if "OLS" in comparators:
        lr = LinearRegression().fit(x_train.reshape(-1,1), y_train)
        pred = lr.predict(queries.reshape(-1,1))
        slope = np.full_like(queries, float(lr.coef_[0]))
        add("OLS", pred, slope)

    for r in r_grid:
        if "LLR" in comparators:
            ll = FastLocalLinear1D(r=r, min_points=min_stage1).fit(x_train, y_train)
            pred, slope, diag = ll.predict_with_slope(queries)
            add("LLR", pred, slope, r=r, eps=0.0, tuning_name="r", tuning_value=r,
                min_stage1=min_stage1, diagnostics=diag)

        if "LP2" in comparators:
            lp2 = LocalPolynomial1D(r=r, degree=2, min_points=max(min_stage1,3)).fit(x_train,y_train)
            pred, slope, diag = lp2.predict_with_slope(queries)
            add("LP2", pred, slope, r=r, eps=0.0, tuning_name="r", tuning_value=r,
                min_stage1=min_stage1, diagnostics=diag)

        if "KLLR" in comparators:
            kl = TricubeLocalLinear1D(h=r, min_points=min_stage1).fit(x_train,y_train)
            pred, slope, diag = kl.predict_with_slope(queries)
            add("KLLR", pred, slope, r=r, eps=0.0, tuning_name="bandwidth", tuning_value=r,
                min_stage1=min_stage1, diagnostics=diag)

        if "BBLR" in comparators:
            # Stage 1 is invariant to epsilon and is fitted exactly once per r.
            b = FastBBLR1D(r=r, eps=float(eps_grid[0]), min_points_stage1=min_stage1,
                           min_centres_stage2=int(min_stage2_grid[0])).fit(x_train,y_train)
            for m2 in min_stage2_grid:
                for eps in eps_grid:
                    b.set_eps(float(eps), int(m2))
                    pred, slope, diag = b.predict_with_slope(queries)
                    add("BBLR", pred, slope, r=r, eps=eps, tuning_name="r_eps",
                        tuning_value=np.nan, min_stage1=min_stage1, min_stage2=m2,
                        diagnostics=diag)

    if "AKLLR" in comparators or "AKLP2" in comparators:
        for k in knn_k_grid:
            if "AKLLR" in comparators:
                ak1 = KNNLocalPolynomial1D(k=int(k), degree=1).fit(x_train, y_train)
                pred, slope, diag = ak1.predict_with_slope(queries)
                add("AKLLR", pred, slope, k=k, tuning_name="k", tuning_value=k,
                    diagnostics=diag)
            if "AKLP2" in comparators:
                ak2 = KNNLocalPolynomial1D(k=int(k), degree=2).fit(x_train, y_train)
                pred, slope, diag = ak2.predict_with_slope(queries)
                add("AKLP2", pred, slope, k=k, tuning_name="k", tuning_value=k,
                    diagnostics=diag)

    if any(m in comparators for m in ("CSPLINE","CAKLLR","CAKLP2")):
        for gf in gap_factor_grid:
            if "CAKLLR" in comparators or "CAKLP2" in comparators:
                for k in knn_k_grid:
                    if "CAKLLR" in comparators:
                        try:
                            cm=ComponentwiseKNNLocalPolynomial1D(
                                k=int(k),degree=1,gap_factor=float(gf),
                                min_component_points=component_min_points).fit(x_train,y_train)
                            pred,slope,diag=cm.predict_with_slope(queries)
                            add("CAKLLR",pred,slope,k=k,gap_factor=gf,
                                tuning_name="k_gapfactor",tuning_value=k,diagnostics=diag)
                        except Exception:
                            add("CAKLLR",np.full_like(queries,np.nan),np.full_like(queries,np.nan),
                                k=k,gap_factor=gf,tuning_name="k_gapfactor",tuning_value=k)
                    if "CAKLP2" in comparators:
                        try:
                            cm=ComponentwiseKNNLocalPolynomial1D(
                                k=int(k),degree=2,gap_factor=float(gf),
                                min_component_points=component_min_points).fit(x_train,y_train)
                            pred,slope,diag=cm.predict_with_slope(queries)
                            add("CAKLP2",pred,slope,k=k,gap_factor=gf,
                                tuning_name="k_gapfactor",tuning_value=k,diagnostics=diag)
                        except Exception:
                            add("CAKLP2",np.full_like(queries,np.nan),np.full_like(queries,np.nan),
                                k=k,gap_factor=gf,tuning_name="k_gapfactor",tuning_value=k)
            if "CSPLINE" in comparators:
                for lam in spline_lambda_grid:
                    try:
                        cm=ComponentwiseSmoothingSpline1D(
                            lam=float(lam),gap_factor=float(gf),
                            min_component_points=component_min_points).fit(x_train,y_train)
                        pred,slope,diag=cm.predict_with_slope(queries)
                        add("CSPLINE",pred,slope,gap_factor=gf,
                            tuning_name="lambda_gapfactor",tuning_value=lam,diagnostics=diag)
                    except Exception:
                        add("CSPLINE",np.full_like(queries,np.nan),np.full_like(queries,np.nan),
                            gap_factor=gf,tuning_name="lambda_gapfactor",tuning_value=lam)

    if "SPLINE" in comparators:
        order = np.argsort(x_train, kind="mergesort")
        xs = x_train[order]; ys = y_train[order]
        # Continuous DGPs make exact duplicates extremely unlikely.  Aggregate
        # defensively if they occur so make_smoothing_spline receives strict x.
        ux, inv, counts = np.unique(xs, return_inverse=True, return_counts=True)
        if len(ux) != len(xs):
            sums = np.bincount(inv, weights=ys)
            xs = ux; ys = sums / counts
        for lam in spline_lambda_grid:
            try:
                sp = make_smoothing_spline(xs, ys, lam=float(lam))
                pred = np.asarray(sp(queries), float)
                slope = np.asarray(sp.derivative()(queries), float)
                add("SPLINE", pred, slope, tuning_name="lambda", tuning_value=lam)
            except Exception:
                # Preserve the tuning configuration as failed rather than
                # silently deleting a difficult case.
                pred = np.full_like(queries, np.nan); slope = np.full_like(queries, np.nan)
                add("SPLINE", pred, slope, tuning_name="lambda", tuning_value=lam)

    df = pd.DataFrame(rows)
    return _mark_selection_and_pareto(df)


@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: int
    dgp_i: int
    dgp_code: str
    design_i: int
    design: str
    n_i: int
    n: int
    noise_i: int
    noise_sd: float
    rep_start: int
    rep_stop: int


def chunk_filename(spec: ChunkSpec) -> str:
    safe_noise = ("%.8g" % spec.noise_sd).replace("-","m").replace(".","p")
    return (f"chunk_{spec.chunk_id:07d}_{spec.dgp_code}_{spec.design}_"
            f"n{spec.n}_sd{safe_noise}_rep{spec.rep_start:06d}-{spec.rep_stop-1:06d}.csv.gz")


def run_chunk(spec: ChunkSpec, cfg: Dict[str, object], shard_dir: str) -> Dict[str, object]:
    t0 = time.perf_counter()
    path = Path(shard_dir) / chunk_filename(spec)
    if bool(cfg["resume"]) and path.exists() and path.stat().st_size > 40:
        return {"chunk_id": spec.chunk_id, "status": "skipped", "path": str(path),
                "rows": np.nan, "seconds": 0.0}
    complexity = dgp_complexity(spec.dgp_code, float(cfg["guard_width"]))
    frames = []
    for rep in range(spec.rep_start, spec.rep_stop):
        frames.append(one_replication(
            dgp_code=spec.dgp_code, dgp_i=spec.dgp_i,
            design=spec.design, design_i=spec.design_i,
            n=spec.n, n_i=spec.n_i, noise_sd=spec.noise_sd, noise_i=spec.noise_i,
            rep=rep, seed_offset=int(cfg["seed_offset"]),
            n_validation=int(cfg["n_validation"]), n_test=int(cfg["n_test"]),
            n_grid_eval=int(cfg["n_grid_eval"]), r_grid=cfg["r_grid"], eps_grid=cfg["eps_grid"],
            spline_lambda_grid=cfg["spline_lambda_grid"], knn_k_grid=cfg["knn_k_grid"],
            gap_factor_grid=cfg["gap_factor_grid"], component_min_points=int(cfg["component_min_points"]),
            min_stage1=int(cfg["min_stage1"]),
            min_stage2_grid=cfg["min_stage2_grid"], guard_width=float(cfg["guard_width"]),
            comparators=cfg["comparators"], complexity=complexity,
        ))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)
    return {"chunk_id": spec.chunk_id, "status": "written", "path": str(path),
            "rows": int(len(df)), "seconds": float(time.perf_counter()-t0)}


def build_chunks(dgps, designs, n_grid, noise_grid, reps, chunk_reps) -> List[ChunkSpec]:
    out=[]; cid=0
    for di,dgp in enumerate(dgps):
        for xi,design in enumerate(designs):
            for ni,n in enumerate(n_grid):
                for si,sd in enumerate(noise_grid):
                    for start in range(0,reps,chunk_reps):
                        stop=min(start+chunk_reps,reps)
                        out.append(ChunkSpec(cid,di,dgp,xi,design,ni,n,si,sd,start,stop)); cid+=1
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dgps",default="R01C01,R01C02,R01C03,R01C04,R01C05,R02C01,R02C02,R02C03,R02C04,R02C05,R03C01,R03C02,R03C03,R03C04,R03C05,R04C01,R04C02,R04C03,R04C04,R04C05,R05C01,R05C02,R05C03,R05C04,R05C05")
    ap.add_argument("--designs",default="G00,G10,G25,G40,G60,G80,D75,D50,D25,D10,D02")
    ap.add_argument("--n-grid",default="500")
    ap.add_argument("--noise-grid",default="0.25")
    ap.add_argument("--reps",type=int,default=100)
    ap.add_argument("--n-validation",type=int,default=0,help="0 => same as n_train")
    ap.add_argument("--n-test",type=int,default=2000,help="independent outcome/slope test sample")
    ap.add_argument("--n-grid-eval",type=int,default=401,help="uniform support grid for shape diagnostics")
    ap.add_argument("--r-grid",default="0.08,0.12,0.18,0.26,0.38,0.55,0.80,1.20,2.00,4.00")
    ap.add_argument("--eps-grid",default="0,0.04,0.08,0.12,0.18,0.26,0.38,0.55,0.80,1.20,2.00,4.00")
    ap.add_argument("--spline-lambda-grid",default="1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000")
    ap.add_argument("--knn-k-grid",default="15,25,40,60,90,130,180,250,350,500")
    ap.add_argument("--gap-factor-grid",default="16,20,30,inf",
                    help="training-X spacing ratios for support-component detection; inf=no split")
    ap.add_argument("--component-min-points",type=int,default=15)
    ap.add_argument("--min-stage1",type=int,default=15)
    ap.add_argument("--min-stage2-grid",default="1")
    ap.add_argument("--guard-width",type=float,default=0.10)
    ap.add_argument("--comparators",default="OLS,LLR,LP2,KLLR,AKLLR,AKLP2,SPLINE,BBLR")
    ap.add_argument("--seed-offset",type=int,default=271828)
    ap.add_argument("--jobs",type=int,default=1)
    ap.add_argument("--chunk-reps",type=int,default=5)
    ap.add_argument("--backend",choices=["loky"],default="loky")
    ap.add_argument("--output-dir",default=str(ROOT/"results_hpc"))
    ap.add_argument("--resume",action="store_true",default=False)
    args=ap.parse_args()

    dgps=parse_str_list(args.dgps); designs=parse_str_list(args.designs)
    n_grid=parse_int_list(args.n_grid); noise_grid=parse_float_list(args.noise_grid)
    r_grid=parse_float_list(args.r_grid); eps_grid=parse_float_list(args.eps_grid)
    lam_grid=parse_float_list(args.spline_lambda_grid); knn_k_grid=parse_int_list(args.knn_k_grid)
    gap_factor_grid=parse_float_list(args.gap_factor_grid)
    m2_grid=parse_int_list(args.min_stage2_grid)
    comps=[s.upper() for s in parse_str_list(args.comparators)]
    valid_methods={"OLS","LLR","LP2","KLLR","AKLLR","AKLP2","SPLINE","BBLR",
                   "CSPLINE","CAKLLR","CAKLP2"}
    unknown=set(comps)-valid_methods
    if unknown: raise ValueError(f"Unknown comparators: {sorted(unknown)}")
    for d in dgps:
        if d not in DGP_REGISTRY: raise ValueError(f"Unknown DGP {d}")
    for design in designs:
        if design not in SUPPORT_DESIGN_REGISTRY and design not in {"uniform","normal","skewed","gap"}:
            raise ValueError(f"Unknown support design {design}")
    if args.jobs==0 or args.jobs < -1: raise ValueError("jobs must be -1 or positive")
    if args.chunk_reps<1: raise ValueError("chunk-reps must be >=1")

    outdir=Path(args.output_dir); shard_dir=outdir/"shards"
    shard_dir.mkdir(parents=True,exist_ok=True)
    cfg={
        "dgps":dgps,"designs":designs,"n_grid":n_grid,"noise_grid":noise_grid,
        "reps":args.reps,"n_validation":args.n_validation,"n_test":args.n_test,
        "n_grid_eval":args.n_grid_eval,"r_grid":r_grid,"eps_grid":eps_grid,
        "spline_lambda_grid":lam_grid,"knn_k_grid":knn_k_grid,
        "gap_factor_grid":gap_factor_grid,"component_min_points":args.component_min_points,
        "min_stage1":args.min_stage1,
        "min_stage2_grid":m2_grid,"guard_width":args.guard_width,
        "comparators":comps,"seed_offset":args.seed_offset,"jobs":args.jobs,
        "chunk_reps":args.chunk_reps,"backend":args.backend,"resume":args.resume,
        "thread_env":{v:os.environ.get(v) for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")},
    }
    (outdir/"RUN_CONFIG.json").write_text(json.dumps(cfg,indent=2)+"\n")
    chunks=build_chunks(dgps,designs,n_grid,noise_grid,args.reps,args.chunk_reps)
    pd.DataFrame([asdict(c) | {"filename":chunk_filename(c)} for c in chunks]).to_csv(outdir/"EXPECTED_CHUNKS.csv",index=False)

    t0=time.perf_counter()
    if args.jobs==1:
        results=[run_chunk(c,cfg,str(shard_dir)) for c in chunks]
    else:
        with parallel_config(backend="loky", n_jobs=args.jobs, inner_max_num_threads=1):
            results=Parallel(verbose=10,batch_size=1,pre_dispatch="2*n_jobs")(
                delayed(run_chunk)(c,cfg,str(shard_dir)) for c in chunks
            )
    elapsed=time.perf_counter()-t0
    manifest=pd.DataFrame(results).sort_values("chunk_id")
    manifest.to_csv(outdir/"CHUNK_MANIFEST.csv",index=False)
    missing=[chunk_filename(c) for c in chunks if not (shard_dir/chunk_filename(c)).exists()]
    audit={
        "expected_chunks":len(chunks),"present_chunks":len(chunks)-len(missing),
        "missing_chunks":len(missing),"missing_files":missing[:100],
        "wall_seconds":elapsed,"jobs":args.jobs,"chunk_reps":args.chunk_reps,
        "status":"PASS" if not missing else "FAIL",
    }
    (outdir/"COMPLETION_AUDIT.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))
    print(f"Shards: {shard_dir}")

if __name__=="__main__":
    main()
