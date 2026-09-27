#!/usr/bin/env python3
"""Summarize sharded BBLR support-regime Monte Carlo output.

The routine keeps the complete configuration summary on the source machine, but
also creates compact selected-estimator, oracle, robust-summary and Pareto files
needed for analysis handbacks.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

DGP_META = [
    "dgp","dgp_label","dgp_family","dgp_parameter_name","dgp_parameter_value",
    "dgp_secondary_parameter_name","dgp_secondary_parameter_value",
]
DESIGN_META = [
    "design","design_label","design_family","design_parameter_name","design_parameter_value",
    "design_secondary_parameter_name","design_secondary_parameter_value",
]
CELL = DGP_META + DESIGN_META + ["n_train","noise_sd"]
TUNE = ["method","r","eps","k","gap_factor","tuning_name","tuning_value","min_stage1","min_stage2"]

METRICS = [
    "validation_y_rmse","test_y_rmse","test_mean_rmse","test_slope_rmse",
    "grid_mean_rmse","grid_slope_rmse","grid_slope_rmse_interior",
    "grid_slope_rmse_boundary","test_slope_bias",
    "test_slope_rmse_near_feature","test_slope_rmse_away_feature",
    "test_slope_rmse_left_regime","test_slope_rmse_right_regime",
    "selected_by_validation_y","selected_by_validation_y_1se_low_smoothing",
    "selected_by_validation_y_tol_0p5_low_smoothing","selected_by_validation_y_tol_1p0_low_smoothing",
    "oracle_slope_within_method","oracle_mean_within_method",
    "pareto_all_configurations","pareto_selected_methods",
]

KEEP = CELL + [
    "rep","method","validation_y_rmse","test_y_rmse","test_mean_rmse",
    "test_slope_rmse","grid_slope_rmse","test_slope_bias",
    "test_slope_rmse_near_feature","test_slope_rmse_away_feature",
    "test_slope_rmse_left_regime","test_slope_rmse_right_regime",
    "n_test_near_feature","n_test_away_feature","n_test_left_regime","n_test_right_regime",
    "pareto_selected_methods","r","eps","k","gap_factor","tuning_name","tuning_value",
    "selected_by_validation_y_1se_low_smoothing",
    "selected_by_validation_y_tol_0p5_low_smoothing","selected_by_validation_y_tol_1p0_low_smoothing",
    "validation_y_mse","validation_y_mse_se","n_validation",
    "query_fallback_share","mean_query_neighbours",
    "mean_adaptive_bandwidth","median_adaptive_bandwidth",
    "stage2_fallback_share","mean_stage2_centres",
    "stage1_fallback_share","mean_stage1_neighbours",
    "n_support_components","support_spacing_baseline","support_max_observed_gap",
    "support_max_gap_ratio","support_split_points",
]


def q10(x): return x.quantile(0.10)
def q90(x): return x.quantile(0.90)
q10.__name__ = "q10"
q90.__name__ = "q90"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()

    run = Path(args.run_dir)
    shards = sorted((run/"shards").glob("*.csv.gz"))
    if not shards:
        raise SystemExit("No shards found")

    group_cols = CELL + TUNE
    partial = []
    selected = []
    oracle_slope = []
    oracle_mean = []
    pareto = []

    for p in shards:
        df = pd.read_csv(p)
        num = [c for c in METRICS if c in df.columns]
        g = df.groupby(group_cols, dropna=False)[num].agg(["sum","count"])
        g.columns = [f"{a}__{b}" for a,b in g.columns]
        partial.append(g.reset_index())

        keep = [c for c in KEEP if c in df.columns]
        selected.append(df[df["selected_by_validation_y"].astype(bool)][keep])
        oracle_slope.append(df[df["oracle_slope_within_method"].astype(bool)][keep])
        oracle_mean.append(df[df["oracle_mean_within_method"].astype(bool)][keep])
        pareto.append(df[df["pareto_all_configurations"].astype(bool)][keep])

    pp = pd.concat(partial, ignore_index=True)
    sum_cols = [c for c in pp if c.endswith("__sum") or c.endswith("__count")]
    total = pp.groupby(group_cols, dropna=False)[sum_cols].sum().reset_index()
    for m in METRICS:
        s = f"{m}__sum"; c = f"{m}__count"
        if s in total.columns and c in total.columns:
            total[f"{m}__mean"] = total[s] / total[c].replace(0, np.nan)
    total.to_csv(run/"HPC_CONFIGURATION_SUMMARY.csv", index=False)

    sel = pd.concat(selected, ignore_index=True)
    ora_s = pd.concat(oracle_slope, ignore_index=True)
    ora_m = pd.concat(oracle_mean, ignore_index=True)
    par = pd.concat(pareto, ignore_index=True)

    sel.to_csv(run/"HPC_SELECTED_BY_Y.csv.gz", index=False, compression="gzip")
    ora_s.to_csv(run/"HPC_ORACLE_SLOPE_BY_METHOD.csv.gz", index=False, compression="gzip")
    ora_m.to_csv(run/"HPC_ORACLE_MEAN_BY_METHOD.csv.gz", index=False, compression="gzip")
    par.to_csv(run/"HPC_PARETO_CONFIGURATIONS.csv.gz", index=False, compression="gzip")

    # Mean summary retained for backwards compatibility.
    mean_summary = sel.groupby(CELL+["method"], dropna=False).agg(
        reps=("rep","count"),
        validation_y_rmse=("validation_y_rmse","mean"),
        test_y_rmse=("test_y_rmse","mean"),
        test_mean_rmse=("test_mean_rmse","mean"),
        test_slope_rmse=("test_slope_rmse","mean"),
        grid_slope_rmse=("grid_slope_rmse","mean"),
        pareto_selected_share=("pareto_selected_methods","mean"),
    ).reset_index()
    mean_summary.to_csv(run/"HPC_SELECTED_METHOD_SUMMARY.csv", index=False)

    # Robust summaries guard against a small number of catastrophic smooth fits.
    metrics = [m for m in [
        "validation_y_rmse","test_y_rmse","test_mean_rmse","test_slope_rmse",
        "grid_slope_rmse","test_slope_rmse_near_feature","test_slope_rmse_away_feature",
        "test_slope_rmse_left_regime","test_slope_rmse_right_regime",
    ] if m in sel.columns]

    agg = {}
    for m in metrics:
        agg[m] = ["mean","median",q10,q90]
    robust = sel.groupby(CELL+["method"], dropna=False).agg(agg)
    robust.columns = [f"{a}__{b}" for a,b in robust.columns]
    robust = robust.reset_index()
    pareto_share = sel.groupby(CELL+["method"], dropna=False)["pareto_selected_methods"].mean().reset_index(
        name="pareto_selected_share")
    robust = robust.merge(pareto_share, on=CELL+["method"], how="left")
    robust.to_csv(run/"HPC_SELECTED_METHOD_ROBUST_SUMMARY.csv", index=False)

    print(f"Summarized {len(shards)} shards")
    print(run/"HPC_SELECTED_METHOD_ROBUST_SUMMARY.csv")


if __name__ == "__main__":
    main()
