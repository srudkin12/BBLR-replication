#!/usr/bin/env python3
"""Create analysis tables that answer the main 1D BBLR questions."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_csv")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    path = Path(args.metrics_csv)
    outdir = Path(args.output_dir) if args.output_dir else path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)

    keys = ["dgp", "design", "rep", "n", "noise_sd", "r"]
    b = df[df.method == "BBLR"].copy()
    llr = df[df.method == "LLR"][keys + ["slope_rmse", "function_rmse"]].copy()
    llr = llr.rename(columns={"slope_rmse": "llr_slope_rmse", "function_rmse": "llr_function_rmse"})
    gain = b.merge(llr, on=keys, how="left")
    gain["slope_rmse_gain"] = gain["llr_slope_rmse"] - gain["slope_rmse"]
    gain["slope_rmse_gain_pct"] = 100 * gain["slope_rmse_gain"] / gain["llr_slope_rmse"]
    gain["function_rmse_gain"] = gain["llr_function_rmse"] - gain["function_rmse"]
    gain.to_csv(outdir / "second_stage_gain_vs_same_r_llr.csv", index=False)

    group = ["dgp", "design", "rep", "n", "noise_sd", "method"]
    best_rows = []
    for _, g in df.groupby(group, dropna=False):
        row = g.loc[g.slope_rmse.idxmin()].copy()
        best_rows.append(row)
    best = pd.DataFrame(best_rows)
    best.to_csv(outdir / "best_slope_configuration_by_rep.csv", index=False)

    agg = best.groupby(["dgp", "design", "n", "noise_sd", "method"], dropna=False).agg(
        mean_best_slope_rmse=("slope_rmse", "mean"),
        sd_best_slope_rmse=("slope_rmse", "std"),
        mean_function_rmse_at_slope_best=("function_rmse", "mean"),
        mean_r=("r", "mean"),
        mean_eps=("eps", "mean"),
    ).reset_index()
    agg.to_csv(outdir / "best_slope_method_summary.csv", index=False)

    # BBLR function-optimal vs slope-optimal parameter mismatch.
    idcols = ["dgp", "design", "rep", "n", "noise_sd", "min_stage1", "min_stage2", "aggregation"]
    slope_best = b.loc[b.groupby(idcols, dropna=False).slope_rmse.idxmin(), idcols + ["r", "eps", "slope_rmse", "function_rmse"]]
    slope_best = slope_best.rename(columns={"r":"slope_best_r", "eps":"slope_best_eps", "slope_rmse":"slope_best_slope_rmse", "function_rmse":"slope_best_function_rmse"})
    function_best = b.loc[b.groupby(idcols, dropna=False).function_rmse.idxmin(), idcols + ["r", "eps", "slope_rmse", "function_rmse"]]
    function_best = function_best.rename(columns={"r":"function_best_r", "eps":"function_best_eps", "slope_rmse":"function_best_slope_rmse", "function_rmse":"function_best_function_rmse"})
    mismatch = slope_best.merge(function_best, on=idcols, how="inner")
    mismatch["same_parameter_pair"] = (mismatch.slope_best_r == mismatch.function_best_r) & (mismatch.slope_best_eps == mismatch.function_best_eps)
    mismatch["slope_penalty_from_function_choice"] = mismatch.function_best_slope_rmse - mismatch.slope_best_slope_rmse
    mismatch.to_csv(outdir / "function_vs_slope_optimal_parameters.csv", index=False)

    mismatch.groupby(["dgp", "design", "n", "noise_sd"], dropna=False).agg(
        share_same_parameter_pair=("same_parameter_pair", "mean"),
        mean_slope_penalty_from_function_choice=("slope_penalty_from_function_choice", "mean"),
        median_slope_penalty_from_function_choice=("slope_penalty_from_function_choice", "median"),
    ).reset_index().to_csv(outdir / "function_vs_slope_optimal_summary.csv", index=False)

    print(f"Wrote summaries to {outdir}")


if __name__ == "__main__":
    main()
