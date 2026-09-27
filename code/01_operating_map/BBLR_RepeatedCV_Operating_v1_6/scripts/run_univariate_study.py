#!/usr/bin/env python3
"""Monte Carlo laboratory for one-dimensional BBLR.

Example quick pilot:
python scripts/run_univariate_study.py --reps 10 --n 400 \
  --dgps U01,U02,U04,U08,U10,U11 --designs uniform,normal,gap

The script evaluates a grid of Stage-1 radii r and Stage-2 radii eps.  It also
runs a single-stage local-linear estimator using each r, plus global OLS.  The
primary outcome is derivative RMSE against the analytical m'(x); function RMSE
is reported separately.  For kinks/jumps, derivative metrics exclude a guard
band around non-differentiable points.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bblr.univariate import (
    BBLR1D, DGP_REGISTRY, LocalLinear1D, LocalPolynomial1D, boundary_mask,
    differentiable_mask, generate_sample, rmse,
)


def parse_float_list(s: str):
    return [float(v) for v in s.split(",") if v.strip()]


def parse_str_list(s: str):
    return [v.strip() for v in s.split(",") if v.strip()]


def score(method, dgp_code, design, rep, n, noise_sd, r, eps,
          pred, slope, x_eval, diag, guard_width, min_stage1, min_stage2, aggregation):
    dgp = DGP_REGISTRY[dgp_code]
    mu = dgp.mean(x_eval)
    slope_true = dgp.derivative(x_eval)
    diff = differentiable_mask(x_eval, dgp, guard_width)
    bnd = boundary_mask(x_eval, dgp.support)
    interior = ~bnd
    valid_bnd = diff & bnd
    valid_int = diff & interior

    out = {
        "dgp": dgp_code,
        "dgp_label": dgp.label,
        "design": design,
        "rep": rep,
        "n": n,
        "noise_sd": noise_sd,
        "method": method,
        "r": r,
        "eps": eps,
        "min_stage1": min_stage1,
        "min_stage2": min_stage2,
        "aggregation": aggregation,
        "function_rmse": rmse(pred, mu),
        "slope_rmse": rmse(slope, slope_true, diff),
        "slope_bias": float(np.mean((slope-slope_true)[diff])),
        "slope_rmse_interior": rmse(slope, slope_true, valid_int) if valid_int.any() else np.nan,
        "slope_rmse_boundary": rmse(slope, slope_true, valid_bnd) if valid_bnd.any() else np.nan,
        "n_slope_scored": int(diff.sum()),
    }
    if diag:
        if "fallback" in diag:
            out["query_fallback_share"] = float(np.mean(diag["fallback"]))
            out["mean_query_neighbours"] = float(np.mean(diag["neighbour_count"]))
        if "stage2_fallback" in diag:
            out["stage2_fallback_share"] = float(np.mean(diag["stage2_fallback"]))
            out["mean_stage2_centres"] = float(np.mean(diag["stage2_centre_count"]))
            out["stage1_fallback_share"] = float(np.mean(diag["stage1_fallback"]))
            out["mean_stage1_neighbours"] = float(np.mean(diag["stage1_count"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgps", default="U01,U02,U03,U04,U05,U06,U07,U08,U09,U10,U11,U12")
    ap.add_argument("--designs", default="uniform,normal,skewed,gap")
    ap.add_argument("--n", type=int, default=500, help="single sample size; ignored when --n-grid is supplied")
    ap.add_argument("--n-grid", default=None, help="optional comma-separated sample sizes")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--noise-sd", type=float, default=0.25, help="single noise SD; ignored when --noise-grid is supplied")
    ap.add_argument("--noise-grid", default=None, help="optional comma-separated noise SDs")
    ap.add_argument("--r-grid", default="0.15,0.25,0.40,0.60")
    ap.add_argument("--eps-grid", default="0.10,0.20,0.35,0.55")
    ap.add_argument("--min-stage1", type=int, default=15)
    ap.add_argument("--min-stage2", type=int, default=1)
    ap.add_argument("--min-stage2-grid", default=None, help="optional comma-separated Stage-2 minimum centre counts")
    ap.add_argument("--n-eval", type=int, default=301)
    ap.add_argument("--guard-width", type=float, default=0.10)
    ap.add_argument("--aggregation", choices=["mean", "median"], default="mean")
    ap.add_argument("--seed-offset", type=int, default=17000)
    ap.add_argument("--jobs", type=int, default=1, help="outer Monte Carlo workers; use -1 for all cores")
    ap.add_argument("--backend", choices=["loky", "multiprocessing", "threading"], default="loky")
    ap.add_argument("--output-dir", default=str(ROOT / "results_univariate"))
    args = ap.parse_args()

    dgps = parse_str_list(args.dgps)
    designs = parse_str_list(args.designs)
    r_grid = parse_float_list(args.r_grid)
    eps_grid = parse_float_list(args.eps_grid)
    n_grid = [int(v) for v in parse_float_list(args.n_grid)] if args.n_grid else [args.n]
    noise_grid = parse_float_list(args.noise_grid) if args.noise_grid else [args.noise_sd]
    min_stage2_grid = [int(v) for v in parse_float_list(args.min_stage2_grid)] if args.min_stage2_grid else [args.min_stage2]
    for code in dgps:
        if code not in DGP_REGISTRY:
            raise ValueError(f"Unknown DGP {code}")

    def one_replicate(dgp_i, dgp_code, design_i, design, n_i, n, noise_i, noise_sd, rep):
        dgp = DGP_REGISTRY[dgp_code]
        lo, hi = dgp.support
        x_eval = np.linspace(lo, hi, args.n_eval)
        seed = (args.seed_offset + 1000000*dgp_i + 100000*design_i
                + 10000*n_i + 1000*noise_i + rep)
        x, y, _ = generate_sample(dgp_code, n, noise_sd, seed, design)
        local_rows = []
        local_curves = []

        # Global OLS diagnostic benchmark.
        lr = LinearRegression().fit(x.reshape(-1, 1), y)
        pred = lr.predict(x_eval.reshape(-1, 1))
        slope = np.full_like(x_eval, lr.coef_[0], dtype=float)
        local_rows.append(score("OLS", dgp_code, design, rep, n, noise_sd,
                                np.nan, np.nan, pred, slope, x_eval, None, args.guard_width,
                                args.min_stage1, np.nan, args.aggregation))

        # Single-stage local linear and quadratic comparators.
        for r in r_grid:
            ll = LocalLinear1D(r=r, min_points=args.min_stage1).fit(x, y)
            pred_ll, slope_ll, diag_ll = ll.predict_with_slope(x_eval)
            local_rows.append(score("LLR", dgp_code, design, rep, n, noise_sd,
                                    r, 0.0, pred_ll, slope_ll, x_eval, diag_ll, args.guard_width,
                                    args.min_stage1, np.nan, args.aggregation))
            if rep == 0 and n_i == 0 and noise_i == 0:
                truth_mu = dgp.mean(x_eval)
                truth_slope = dgp.derivative(x_eval)
                for xx, pp, ss, tm, ts in zip(x_eval, pred_ll, slope_ll, truth_mu, truth_slope):
                    local_curves.append({
                        "dgp": dgp_code, "design": design, "method": "LLR",
                        "n": n, "noise_sd": noise_sd, "min_stage2": np.nan,
                        "r": r, "eps": 0.0, "x": xx, "mean_true": tm,
                        "mean_est": pp, "slope_true": ts, "slope_est": ss,
                    })

            lp2 = LocalPolynomial1D(r=r, degree=2, min_points=max(args.min_stage1, 3)).fit(x, y)
            pred_lp2, slope_lp2, diag_lp2 = lp2.predict_with_slope(x_eval)
            local_rows.append(score("LP2", dgp_code, design, rep, n, noise_sd,
                                    r, 0.0, pred_lp2, slope_lp2, x_eval, diag_lp2, args.guard_width,
                                    args.min_stage1, np.nan, args.aggregation))
            if rep == 0 and n_i == 0 and noise_i == 0:
                truth_mu = dgp.mean(x_eval)
                truth_slope = dgp.derivative(x_eval)
                for xx, pp, ss, tm, ts in zip(x_eval, pred_lp2, slope_lp2, truth_mu, truth_slope):
                    local_curves.append({
                        "dgp": dgp_code, "design": design, "method": "LP2",
                        "n": n, "noise_sd": noise_sd, "min_stage2": np.nan,
                        "r": r, "eps": 0.0, "x": xx, "mean_true": tm,
                        "mean_est": pp, "slope_true": ts, "slope_est": ss,
                    })

            # Stage 1 depends on r but not eps.  Fit it once and reuse it across
            # the Stage-2 parameter grid.
            bblr = BBLR1D(
                r=r, eps=eps_grid[0],
                min_points_stage1=args.min_stage1,
                min_centres_stage2=min_stage2_grid[0],
                aggregation=args.aggregation,
            ).fit(x, y)
            for min_stage2 in min_stage2_grid:
                bblr.min_centres_stage2 = min_stage2
                for eps in eps_grid:
                    bblr.eps = eps
                    pred_b, slope_b, diag_b = bblr.predict_with_slope(x_eval)
                    local_rows.append(score("BBLR", dgp_code, design, rep, n, noise_sd,
                                            r, eps, pred_b, slope_b, x_eval, diag_b, args.guard_width,
                                            args.min_stage1, min_stage2, args.aggregation))

                    if rep == 0 and n_i == 0 and noise_i == 0:
                        truth_mu = dgp.mean(x_eval)
                        truth_slope = dgp.derivative(x_eval)
                        for xx, pp, ss, tm, ts in zip(x_eval, pred_b, slope_b, truth_mu, truth_slope):
                            local_curves.append({
                                "dgp": dgp_code, "design": design, "method": "BBLR",
                                "n": n, "noise_sd": noise_sd, "min_stage2": min_stage2,
                                "r": r, "eps": eps, "x": xx, "mean_true": tm,
                                "mean_est": pp, "slope_true": ts, "slope_est": ss,
                            })
        return local_rows, local_curves

    tasks = [
        (dgp_i, dgp_code, design_i, design, n_i, n, noise_i, noise_sd, rep)
        for dgp_i, dgp_code in enumerate(dgps)
        for design_i, design in enumerate(designs)
        for n_i, n in enumerate(n_grid)
        for noise_i, noise_sd in enumerate(noise_grid)
        for rep in range(args.reps)
    ]

    if args.jobs == 1:
        pieces = [one_replicate(*task) for task in tasks]
    else:
        pieces = Parallel(n_jobs=args.jobs, backend=args.backend, verbose=5)(
            delayed(one_replicate)(*task) for task in tasks
        )

    rows = [row for part, _ in pieces for row in part]
    curve_rows = [row for _, curves in pieces for row in curves]

    out = pd.DataFrame(rows)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_file = outdir / "univariate_metrics.csv"
    out.to_csv(raw_file, index=False)

    # Monte Carlo mean performance for each parameter combination.
    group_cols = ["dgp", "dgp_label", "design", "method", "r", "eps", "n", "noise_sd",
                  "min_stage1", "min_stage2", "aggregation"]
    metric_cols = [c for c in out.columns if c not in group_cols + ["rep"]]
    numeric = [c for c in metric_cols if pd.api.types.is_numeric_dtype(out[c])]
    summary = out.groupby(group_cols, dropna=False)[numeric].mean().reset_index()
    summary.to_csv(outdir / "univariate_summary.csv", index=False)

    # Oracle diagnostics: these are NOT feasible tuning rules.  They answer
    # whether the (r, eps) that is best for prediction is also best for slope
    # recovery, which is central to the manuscript's estimand question.
    b = out[out.method == "BBLR"].copy()
    idcols = ["dgp", "design", "rep", "n", "noise_sd", "min_stage1", "min_stage2", "aggregation"]
    best_slope = b.loc[b.groupby(idcols)["slope_rmse"].idxmin()].copy()
    best_slope["oracle_target"] = "slope"
    best_function = b.loc[b.groupby(idcols)["function_rmse"].idxmin()].copy()
    best_function["oracle_target"] = "function"
    pd.concat([best_slope, best_function], ignore_index=True).to_csv(
        outdir / "univariate_oracle_choices.csv", index=False
    )

    if curve_rows:
        pd.DataFrame(curve_rows).to_csv(outdir / "univariate_rep0_curves.csv", index=False)

    print(f"Wrote {raw_file}")
    print(f"Rows: {len(out):,}")


if __name__ == "__main__":
    main()
