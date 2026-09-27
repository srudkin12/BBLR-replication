#!/usr/bin/env python3
"""Summarize BBLR repeated-CV landmark shards into compact analysis objects."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()
    run = args.run_dir.resolve()
    sel_files = sorted((run / "selector_shards").glob("*_selectors.csv.gz"))
    grid_files = sorted((run / "grid_shards").glob("*_grid.csv.gz"))
    if not sel_files or not grid_files:
        raise SystemExit("missing repeated-CV shards")
    sel = pd.concat((pd.read_csv(f) for f in sel_files), ignore_index=True)
    sel.to_csv(run / "RCV_SELECTOR_ROWS.csv.gz", index=False, compression="gzip")

    keys = ["cell_id", "landmark_label", "panel_block", "expected_regime", "dgp", "design", "n_train_legacy", "n_development", "noise_sd", "selector"]
    summary = sel.groupby(keys, dropna=False).agg(
        reps=("rep", "nunique"),
        mean_test_y_rmse=("test_y_rmse", "mean"),
        median_test_y_rmse=("test_y_rmse", "median"),
        mean_test_slope_rmse=("test_slope_rmse", "mean"),
        median_test_slope_rmse=("test_slope_rmse", "median"),
        mean_slope_regret_ratio=("slope_regret_ratio", "mean"),
        median_slope_regret_ratio=("slope_regret_ratio", "median"),
        mean_r=("r", "mean"),
        mean_eps=("eps", "mean"),
        eps_zero_share=("eps", lambda x: float(np.mean(np.isclose(x.astype(float), 0.0)))),
        mean_coefficient_instability=("coefficient_instability", "mean"),
    ).reset_index()
    summary.to_csv(run / "RCV_SELECTOR_SUMMARY.csv", index=False)

    # Read only columns needed for tuning geometry from grid shards.
    cols = [
        "cell_id", "landmark_label", "panel_block", "expected_regime", "dgp", "design", "rep", "r", "eps",
        "single_split_rmse", "cv5_rmse", "repeated_cv_rmse", "coefficient_instability",
        "test_y_rmse", "test_slope_rmse", "oracle_slope",
    ]
    grids = []
    for f in grid_files:
        z = pd.read_csv(f, usecols=lambda c: c in cols)
        grids.append(z)
    grid = pd.concat(grids, ignore_index=True)
    # Per-replication risk/slope rank correlations and admissible-set sizes.
    rows = []
    for keys2, g in grid.groupby(["cell_id", "landmark_label", "panel_block", "expected_regime", "dgp", "design", "rep"], sort=False):
        best = float(g.repeated_cv_rmse.min())
        oracle = float(g.loc[g.oracle_slope.astype(bool), "test_slope_rmse"].iloc[0])
        rec = dict(zip(["cell_id", "landmark_label", "panel_block", "expected_regime", "dgp", "design", "rep"], keys2))
        rec["spearman_rcv_vs_slope"] = float(g[["repeated_cv_rmse", "test_slope_rmse"]].corr(method="spearman").iloc[0,1])
        for tol in (0.005, 0.01, 0.02):
            tag = str(tol).replace("0.", "0p")
            e = g[g.repeated_cv_rmse <= best*(1+tol)+1e-15]
            rec[f"admissible_n_{tag}"] = int(len(e))
            rec[f"best_admissible_slope_regret_{tag}"] = float(e.test_slope_rmse.min()/oracle) if oracle > 0 else np.nan
        rows.append(rec)
    geom = pd.DataFrame(rows)
    geom.to_csv(run / "RCV_TUNING_GEOMETRY.csv.gz", index=False, compression="gzip")
    geom.groupby(["cell_id", "landmark_label", "panel_block", "expected_regime", "dgp", "design"], dropna=False).agg(
        reps=("rep", "nunique"),
        median_spearman_rcv_vs_slope=("spearman_rcv_vs_slope", "median"),
        mean_spearman_rcv_vs_slope=("spearman_rcv_vs_slope", "mean"),
        median_admissible_n_0p005=("admissible_n_0p005", "median"),
        median_best_admissible_slope_regret_0p005=("best_admissible_slope_regret_0p005", "median"),
        median_admissible_n_0p01=("admissible_n_0p01", "median"),
        median_best_admissible_slope_regret_0p01=("best_admissible_slope_regret_0p01", "median"),
        median_admissible_n_0p02=("admissible_n_0p02", "median"),
        median_best_admissible_slope_regret_0p02=("best_admissible_slope_regret_0p02", "median"),
    ).reset_index().to_csv(run / "RCV_TUNING_GEOMETRY_SUMMARY.csv", index=False)
    print(f"Summarized {len(sel_files)} selector shards and {len(grid_files)} grid shards")

if __name__ == "__main__":
    main()
