#!/usr/bin/env python3
"""Support-regime analysis for BBLR univariate Monte Carlo experiments."""
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
REPKEY = ["dgp","design","n_train","noise_sd","rep"]


def safe_ratio(a, b):
    return np.where(np.isfinite(a) & np.isfinite(b) & (b != 0), a/b, np.nan)


def add_alignment(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    centre = out["dgp_secondary_parameter_value"].astype(float)
    out["transition_centre_abs"] = np.abs(centre)
    out["transition_width"] = out["dgp_parameter_value"].astype(float)

    hard = out["design_family"].eq("hard_gap")
    gap = out["design_parameter_value"].astype(float)
    out["hard_gap_halfwidth"] = np.where(hard, gap, np.nan)
    # Negative => feature centre lies inside the empty central gap.
    out["centre_minus_gap_edge"] = np.where(hard, np.abs(centre)-gap, np.nan)
    out["centre_inside_hard_gap"] = np.where(
        hard, (gap > 0) & (np.abs(centre) < gap), False
    )

    soft = out["design_family"].eq("soft_gap")
    retention = out["design_parameter_value"].astype(float)
    window = out["design_secondary_parameter_value"].astype(float)
    out["soft_central_retention"] = np.where(soft, retention, np.nan)
    out["centre_inside_soft_window"] = np.where(
        soft, np.abs(centre) <= window, False
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)

    selected = pd.read_csv(run/"HPC_SELECTED_BY_Y.csv.gz")
    oracle = pd.read_csv(run/"HPC_ORACLE_SLOPE_BY_METHOD.csv.gz")
    selected = add_alignment(selected)

    # Pairwise replication-level comparisons against outcome-selected BBLR.
    bblr = selected[selected["method"]=="BBLR"].copy()
    comparators = sorted(m for m in selected["method"].unique() if m != "BBLR")
    pair_frames = []
    for comp in comparators:
        c = selected[selected["method"]==comp].copy()
        keep_c = REPKEY + [
            "test_y_rmse","test_slope_rmse","test_mean_rmse",
            "test_slope_rmse_near_feature","test_slope_rmse_away_feature",
            "pareto_selected_methods",
        ]
        keep_c = [x for x in keep_c if x in c.columns]
        merged = bblr.merge(
            c[keep_c],
            on=REPKEY,
            how="inner",
            suffixes=("_bblr","_comp"),
            validate="one_to_one",
        )
        merged["comparator"] = comp
        merged["y_ratio_bblr_over_comp"] = safe_ratio(
            merged["test_y_rmse_bblr"], merged["test_y_rmse_comp"])
        merged["slope_ratio_bblr_over_comp"] = safe_ratio(
            merged["test_slope_rmse_bblr"], merged["test_slope_rmse_comp"])
        merged["bblr_y_win"] = merged["test_y_rmse_bblr"] < merged["test_y_rmse_comp"]
        merged["bblr_slope_win"] = merged["test_slope_rmse_bblr"] < merged["test_slope_rmse_comp"]
        merged["bblr_dominates"] = merged["bblr_y_win"] & merged["bblr_slope_win"]
        merged["comp_dominates"] = (
            (merged["test_y_rmse_comp"] < merged["test_y_rmse_bblr"])
            & (merged["test_slope_rmse_comp"] < merged["test_slope_rmse_bblr"])
        )
        if "test_slope_rmse_near_feature_bblr" in merged:
            merged["near_feature_slope_ratio_bblr_over_comp"] = safe_ratio(
                merged["test_slope_rmse_near_feature_bblr"],
                merged["test_slope_rmse_near_feature_comp"],
            )
        if "test_slope_rmse_away_feature_bblr" in merged:
            merged["away_feature_slope_ratio_bblr_over_comp"] = safe_ratio(
                merged["test_slope_rmse_away_feature_bblr"],
                merged["test_slope_rmse_away_feature_comp"],
            )
        pair_frames.append(merged)

    pair = pd.concat(pair_frames, ignore_index=True)
    pair.to_csv(run/"SUPPORT_PAIRWISE_REPLICATION.csv.gz", index=False, compression="gzip")

    group_cols = CELL + ["comparator"]
    agg_dict = {
        "rep": ("rep","count"),
        "mean_y_ratio": ("y_ratio_bblr_over_comp","mean"),
        "median_y_ratio": ("y_ratio_bblr_over_comp","median"),
        "mean_slope_ratio": ("slope_ratio_bblr_over_comp","mean"),
        "median_slope_ratio": ("slope_ratio_bblr_over_comp","median"),
        "bblr_y_win_share": ("bblr_y_win","mean"),
        "bblr_slope_win_share": ("bblr_slope_win","mean"),
        "bblr_dominance_share": ("bblr_dominates","mean"),
        "comp_dominance_share": ("comp_dominates","mean"),
    }
    if "near_feature_slope_ratio_bblr_over_comp" in pair:
        agg_dict["mean_near_feature_slope_ratio"] = (
            "near_feature_slope_ratio_bblr_over_comp","mean")
        agg_dict["median_near_feature_slope_ratio"] = (
            "near_feature_slope_ratio_bblr_over_comp","median")
    if "away_feature_slope_ratio_bblr_over_comp" in pair:
        agg_dict["mean_away_feature_slope_ratio"] = (
            "away_feature_slope_ratio_bblr_over_comp","mean")
        agg_dict["median_away_feature_slope_ratio"] = (
            "away_feature_slope_ratio_bblr_over_comp","median")

    cell = pair.groupby(group_cols, dropna=False).agg(**agg_dict).reset_index()
    cell = add_alignment(cell)
    cell.to_csv(run/"SUPPORT_PAIRWISE_CELL_SUMMARY.csv", index=False)

    # Explicit phase tables for the key competitors.
    key_comps = [m for m in ["SPLINE","AKLLR","AKLP2","LP2","KLLR","LLR"] if m in comparators]
    phase = cell[cell["comparator"].isin(key_comps)].copy()
    phase.to_csv(run/"SUPPORT_PHASE_MAP.csv", index=False)

    # Outcome-selected slope regret relative to each method's infeasible slope oracle.
    ok = REPKEY + ["method","test_slope_rmse"]
    oracle_small = oracle[ok].rename(columns={"test_slope_rmse":"oracle_test_slope_rmse"})
    regret = selected.merge(
        oracle_small,
        on=REPKEY+["method"],
        how="left",
        validate="one_to_one",
    )
    regret["slope_selection_regret_ratio"] = safe_ratio(
        regret["test_slope_rmse"], regret["oracle_test_slope_rmse"])
    regret_summary = regret.groupby(CELL+["method"], dropna=False).agg(
        reps=("rep","count"),
        mean_selected_slope_rmse=("test_slope_rmse","mean"),
        median_selected_slope_rmse=("test_slope_rmse","median"),
        mean_oracle_slope_rmse=("oracle_test_slope_rmse","mean"),
        mean_regret_ratio=("slope_selection_regret_ratio","mean"),
        median_regret_ratio=("slope_selection_regret_ratio","median"),
    ).reset_index()
    regret_summary.to_csv(run/"SUPPORT_SELECTION_REGRET_SUMMARY.csv", index=False)

    # Tuning-boundary audit for every method.
    audits = []
    for method, g in selected.groupby("method", sort=False):
        tune = g["tuning_value"].astype(float)
        finite = tune[np.isfinite(tune)]
        if finite.empty:
            continue
        lo = finite.min(); hi = finite.max()
        gg = g.copy()
        gg["at_tuning_min"] = np.isclose(tune, lo, rtol=0, atol=1e-12)
        gg["at_tuning_max"] = np.isclose(tune, hi, rtol=0, atol=1e-12)
        a = gg.groupby(CELL, dropna=False).agg(
            reps=("rep","count"),
            tuning_min_share=("at_tuning_min","mean"),
            tuning_max_share=("at_tuning_max","mean"),
            mean_tuning=("tuning_value","mean"),
            median_tuning=("tuning_value","median"),
        ).reset_index()
        a["method"] = method
        a["global_tuning_min"] = lo
        a["global_tuning_max"] = hi
        audits.append(a)

    # BBLR has two dimensions, so audit r and eps separately.
    b = selected[selected["method"]=="BBLR"].copy()
    for var in ["r","eps"]:
        vals = b[var].dropna().astype(float)
        lo = vals.min(); hi = vals.max()
        b[f"{var}_at_min"] = np.isclose(b[var], lo, rtol=0, atol=1e-12)
        b[f"{var}_at_max"] = np.isclose(b[var], hi, rtol=0, atol=1e-12)
    ba = b.groupby(CELL, dropna=False).agg(
        reps=("rep","count"),
        r_at_min_share=("r_at_min","mean"),
        r_at_max_share=("r_at_max","mean"),
        eps_at_min_share=("eps_at_min","mean"),
        eps_at_max_share=("eps_at_max","mean"),
        mean_selected_r=("r","mean"),
        median_selected_r=("r","median"),
        mean_selected_eps=("eps","mean"),
        median_selected_eps=("eps","median"),
    ).reset_index()
    ba.to_csv(run/"SUPPORT_BBLR_TUNING_BOUNDARY_AUDIT.csv", index=False)

    if audits:
        pd.concat(audits, ignore_index=True).to_csv(
            run/"SUPPORT_METHOD_TUNING_BOUNDARY_AUDIT.csv", index=False)

    # Selected-method Pareto shares by support/transition geometry.
    pareto = selected.groupby(CELL+["method"], dropna=False).agg(
        pareto_selected_share=("pareto_selected_methods","mean")
    ).reset_index()
    pareto.to_csv(run/"SUPPORT_SELECTED_PARETO_SUMMARY.csv", index=False)

    print("Wrote support-regime analysis outputs")
    for fn in [
        "SUPPORT_PAIRWISE_REPLICATION.csv.gz",
        "SUPPORT_PAIRWISE_CELL_SUMMARY.csv",
        "SUPPORT_PHASE_MAP.csv",
        "SUPPORT_SELECTION_REGRET_SUMMARY.csv",
        "SUPPORT_BBLR_TUNING_BOUNDARY_AUDIT.csv",
        "SUPPORT_METHOD_TUNING_BOUNDARY_AUDIT.csv",
        "SUPPORT_SELECTED_PARETO_SUMMARY.csv",
    ]:
        print(run/fn)


if __name__ == "__main__":
    main()
