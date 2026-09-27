"""Repeated cross-validation diagnostics for the one-dimensional BBLR laboratory.

The module is deliberately BBLR-focused.  It compares alternative *observable*
outcome-based tuning rules for the same BBLR tuning cloud while using the true
conditional mean and derivative only after tuning for Monte Carlo evaluation.

Design principle
----------------
A development sample is used for all tuning rules.  The legacy single-split
selector uses one half for fitting and one half for validation; K-fold and
repeated K-fold CV use the complete development sample.  After a tuning rule is
chosen, the selected BBLR configuration is refitted on the complete development
sample before independent test evaluation.  This holds final estimation sample
size fixed across selectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .univariate import DGP_REGISTRY, differentiable_mask, rmse
from .univariate_fast import FastBBLR1D

Array = np.ndarray


@dataclass(frozen=True)
class CVConfig:
    r: float
    eps: float


def _grid(r_grid: Sequence[float], eps_grid: Sequence[float]) -> List[CVConfig]:
    return [CVConfig(float(r), float(eps)) for r in r_grid for eps in eps_grid]


def _fit_predict_grid(
    x_train: Array,
    y_train: Array,
    x_query: Array,
    stability_grid: Array,
    r_grid: Sequence[float],
    eps_grid: Sequence[float],
    min_stage1: int,
    min_stage2: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return predictions on x_query and slopes on stability_grid for all configs.

    Shape is (n_r, n_eps, n_query) for predictions and
    (n_r, n_eps, n_stability_grid) for slopes.  Stage 1 is fitted exactly once
    per r and reused across epsilon values.
    """
    x_query = np.asarray(x_query, float).reshape(-1)
    stability_grid = np.asarray(stability_grid, float).reshape(-1)
    pred = np.empty((len(r_grid), len(eps_grid), len(x_query)), dtype=float)
    slope = np.empty((len(r_grid), len(eps_grid), len(stability_grid)), dtype=float)
    for ir, r in enumerate(r_grid):
        model = FastBBLR1D(
            r=float(r), eps=float(eps_grid[0]),
            min_points_stage1=int(min_stage1), min_centres_stage2=int(min_stage2),
        ).fit(x_train, y_train)
        for ie, eps in enumerate(eps_grid):
            model.set_eps(float(eps), int(min_stage2))
            pp, _, _ = model.predict_with_slope(x_query)
            _, bb, _ = model.predict_with_slope(stability_grid)
            pred[ir, ie, :] = pp
            slope[ir, ie, :] = bb
    return pred, slope


def _risk_from_predictions(pred: np.ndarray, y: Array) -> np.ndarray:
    y = np.asarray(y, float).reshape(-1)
    err = pred - y[None, None, :]
    return np.mean(err * err, axis=2)


def evaluate_bblr_cv_surface(
    x_dev: Array,
    y_dev: Array,
    *,
    r_grid: Sequence[float],
    eps_grid: Sequence[float],
    min_stage1: int = 15,
    min_stage2: int = 1,
    folds: int = 5,
    repeats: int = 10,
    seed: int = 12345,
    stability_grid: Array | None = None,
) -> pd.DataFrame:
    """Estimate BBLR prediction risk under single split, K-fold and repeated CV.

    No true DGP quantities are used here.
    """
    x_dev = np.asarray(x_dev, float).reshape(-1)
    y_dev = np.asarray(y_dev, float).reshape(-1)
    if x_dev.size != y_dev.size:
        raise ValueError("x_dev and y_dev length mismatch")
    if x_dev.size < 2 * max(int(min_stage1), int(folds)):
        raise ValueError("development sample too small")
    if folds < 2 or repeats < 1:
        raise ValueError("folds must be >=2 and repeats >=1")
    if stability_grid is None:
        lo, hi = float(np.min(x_dev)), float(np.max(x_dev))
        stability_grid = np.linspace(lo, hi, 81)
    stability_grid = np.asarray(stability_grid, float).reshape(-1)

    nr, ne = len(r_grid), len(eps_grid)

    # One deterministic 50:50 split, preserving the information structure of
    # the legacy train/validation procedure while later refitting on all data.
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(x_dev.size)
    cut = x_dev.size // 2
    tr = perm[:cut]
    va = perm[cut:]
    pred, _ = _fit_predict_grid(
        x_dev[tr], y_dev[tr], x_dev[va], stability_grid,
        r_grid, eps_grid, min_stage1, min_stage2,
    )
    single_mse = _risk_from_predictions(pred, y_dev[va])

    first_sse = np.zeros((nr, ne), float)
    first_n = 0
    repeated_sse = np.zeros((nr, ne), float)
    repeated_n = 0

    # Welford accumulator for the slope surface across fold fits.
    stab_count = 0
    stab_mean = np.zeros((nr, ne, stability_grid.size), float)
    stab_m2 = np.zeros_like(stab_mean)

    for rr in range(int(repeats)):
        rs = int(np.random.SeedSequence([int(seed), 9001, rr]).generate_state(1, dtype=np.uint32)[0])
        kf = KFold(n_splits=int(folds), shuffle=True, random_state=rs)
        for fold_i, (tri, vai) in enumerate(kf.split(x_dev)):
            pred, slopes = _fit_predict_grid(
                x_dev[tri], y_dev[tri], x_dev[vai], stability_grid,
                r_grid, eps_grid, min_stage1, min_stage2,
            )
            err = pred - y_dev[vai][None, None, :]
            sse = np.sum(err * err, axis=2)
            repeated_sse += sse
            repeated_n += len(vai)
            if rr == 0:
                first_sse += sse
                first_n += len(vai)

            stab_count += 1
            delta = slopes - stab_mean
            stab_mean += delta / stab_count
            stab_m2 += delta * (slopes - stab_mean)

    cv5_mse = first_sse / float(first_n)
    rcv_mse = repeated_sse / float(repeated_n)
    if stab_count > 1:
        point_var = stab_m2 / float(stab_count - 1)
        # Same units as a slope: RMS fold-to-fold SD across the stability grid.
        instability = np.sqrt(np.mean(point_var, axis=2))
    else:
        instability = np.full((nr, ne), np.nan)

    rows = []
    for ir, r in enumerate(r_grid):
        for ie, eps in enumerate(eps_grid):
            rows.append({
                "r": float(r),
                "eps": float(eps),
                "single_split_rmse": float(np.sqrt(single_mse[ir, ie])),
                "cv5_rmse": float(np.sqrt(cv5_mse[ir, ie])),
                "repeated_cv_rmse": float(np.sqrt(rcv_mse[ir, ie])),
                "coefficient_instability": float(instability[ir, ie]),
                "cv_folds": int(folds),
                "cv_repeats": int(repeats),
                "cv_fold_fits": int(folds * repeats),
            })
    return pd.DataFrame(rows)


def evaluate_full_bblr_grid(
    x_dev: Array,
    y_dev: Array,
    x_test: Array,
    y_test: Array,
    *,
    dgp_code: str,
    r_grid: Sequence[float],
    eps_grid: Sequence[float],
    min_stage1: int = 15,
    min_stage2: int = 1,
    guard_width: float = 0.10,
    x_grid: Array | None = None,
) -> pd.DataFrame:
    """Refit every BBLR configuration on all development data and evaluate truth.

    True mean/slope are simulation diagnostics only and are never consumed by
    the tuning functions above.
    """
    dgp = DGP_REGISTRY[dgp_code]
    x_dev = np.asarray(x_dev, float).reshape(-1)
    y_dev = np.asarray(y_dev, float).reshape(-1)
    x_test = np.asarray(x_test, float).reshape(-1)
    y_test = np.asarray(y_test, float).reshape(-1)
    if x_grid is None:
        x_grid = np.linspace(dgp.support[0], dgp.support[1], 401)
    x_grid = np.asarray(x_grid, float).reshape(-1)
    true_mean_test = np.asarray(dgp.mean(x_test), float)
    true_slope_test = np.asarray(dgp.derivative(x_test), float)
    true_slope_grid = np.asarray(dgp.derivative(x_grid), float)
    mask_test = differentiable_mask(x_test, dgp, guard_width)
    mask_grid = differentiable_mask(x_grid, dgp, guard_width)

    rows = []
    for r in r_grid:
        model = FastBBLR1D(
            r=float(r), eps=float(eps_grid[0]),
            min_points_stage1=int(min_stage1), min_centres_stage2=int(min_stage2),
        ).fit(x_dev, y_dev)
        for eps in eps_grid:
            model.set_eps(float(eps), int(min_stage2))
            pred, slope, _ = model.predict_with_slope(x_test)
            _, slope_grid, _ = model.predict_with_slope(x_grid)
            rows.append({
                "r": float(r),
                "eps": float(eps),
                "test_y_rmse": rmse(pred, y_test),
                "test_mean_rmse": rmse(pred, true_mean_test),
                "test_slope_rmse": rmse(slope, true_slope_test, mask_test),
                "grid_slope_rmse": rmse(slope_grid, true_slope_grid, mask_grid),
            })
    return pd.DataFrame(rows)


def mark_selectors(df: pd.DataFrame, tolerances: Sequence[float] = (0.005, 0.01, 0.02)) -> pd.DataFrame:
    """Mark observable tuning rules plus explicitly labelled simulation oracles."""
    out = df.copy()
    for c in [
        "selected_single_split", "selected_cv5", "selected_repeated_cv",
        "oracle_slope", "oracle_mean",
    ]:
        out[c] = False
    for tol in tolerances:
        tag = str(tol).replace("0.", "0p").replace(".", "p")
        out[f"selected_rcv_tol_{tag}_stability"] = False
        out[f"oracle_best_within_rcv_tol_{tag}"] = False

    def _idxmin(col: str, subset: pd.DataFrame | None = None) -> int:
        g = out if subset is None else subset
        # Deterministic ties: favour lower epsilon then lower radius.
        return int(g.sort_values([col, "eps", "r"], kind="mergesort").index[0])

    out.loc[_idxmin("single_split_rmse"), "selected_single_split"] = True
    out.loc[_idxmin("cv5_rmse"), "selected_cv5"] = True
    out.loc[_idxmin("repeated_cv_rmse"), "selected_repeated_cv"] = True
    out.loc[_idxmin("test_slope_rmse"), "oracle_slope"] = True
    out.loc[_idxmin("test_mean_rmse"), "oracle_mean"] = True

    best = float(out["repeated_cv_rmse"].min())
    for tol in tolerances:
        tag = str(tol).replace("0.", "0p").replace(".", "p")
        eligible = out[out["repeated_cv_rmse"] <= best * (1.0 + float(tol)) + 1e-15]
        # Observable stability selector. Prediction risk is primary; instability
        # only chooses among prediction-admissible configurations.
        idx = int(eligible.sort_values(
            ["coefficient_instability", "repeated_cv_rmse", "eps", "r"],
            kind="mergesort",
        ).index[0])
        out.loc[idx, f"selected_rcv_tol_{tag}_stability"] = True
        oracle_idx = int(eligible.sort_values(
            ["test_slope_rmse", "repeated_cv_rmse", "eps", "r"],
            kind="mergesort",
        ).index[0])
        out.loc[oracle_idx, f"oracle_best_within_rcv_tol_{tag}"] = True
    return out


def selector_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return one compact row per selector/oracle for a marked tuning grid."""
    flag_cols = [c for c in df.columns if c.startswith("selected_") or c.startswith("oracle_") or c.startswith("llr_selected_")]
    rows = []
    oracle_slope = float(df.loc[df["oracle_slope"], "test_slope_rmse"].iloc[0])
    for flag in flag_cols:
        g = df[df[flag].astype(bool)]
        if len(g) != 1:
            continue
        row = g.iloc[0].to_dict()
        row["selector"] = flag
        row["slope_regret_ratio"] = float(row["test_slope_rmse"] / oracle_slope) if oracle_slope > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
