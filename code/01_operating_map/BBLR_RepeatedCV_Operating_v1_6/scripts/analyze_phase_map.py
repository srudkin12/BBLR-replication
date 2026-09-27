#!/usr/bin/env python3
"""Create design-level BBLR phase-map diagnostics from summarized HPC output."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

KEYS = [
    "dgp","dgp_label","dgp_family","dgp_parameter_name","dgp_parameter_value",
    "dgp_secondary_parameter_name","dgp_secondary_parameter_value",
    "design","n_train","noise_sd"
]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",required=True)
    args=ap.parse_args()
    run=Path(args.run_dir)
    selected=pd.read_csv(run/"HPC_SELECTED_METHOD_SUMMARY.csv")
    chosen=pd.read_csv(run/"HPC_SELECTED_BY_Y.csv.gz")
    oracle=pd.read_csv(run/"HPC_ORACLE_SLOPE_BY_METHOD.csv.gz")

    y=selected.pivot_table(index=KEYS,columns="method",values="test_y_rmse")
    s=selected.pivot_table(index=KEYS,columns="method",values="test_slope_rmse")
    methods=[m for m in s.columns if m!="BBLR" and m in y.columns]
    rows=[]
    for idx in s.index:
        base=dict(zip(KEYS,idx if isinstance(idx,tuple) else (idx,)))
        for m in methods:
            if not (np.isfinite(s.loc[idx,"BBLR"]) and np.isfinite(s.loc[idx,m])):
                continue
            rows.append(base | {
                "comparator":m,
                "bblr_test_y_rmse":float(y.loc[idx,"BBLR"]),
                "comparator_test_y_rmse":float(y.loc[idx,m]),
                "bblr_slope_rmse":float(s.loc[idx,"BBLR"]),
                "comparator_slope_rmse":float(s.loc[idx,m]),
                "y_rmse_ratio_bblr_over_comparator":float(y.loc[idx,"BBLR"]/y.loc[idx,m]),
                "slope_rmse_ratio_bblr_over_comparator":float(s.loc[idx,"BBLR"]/s.loc[idx,m]),
                "bblr_dominates":bool((y.loc[idx,"BBLR"]<y.loc[idx,m]) and (s.loc[idx,"BBLR"]<s.loc[idx,m])),
                "comparator_dominates":bool((y.loc[idx,m]<y.loc[idx,"BBLR"]) and (s.loc[idx,m]<s.loc[idx,"BBLR"])),
            })
    gains=pd.DataFrame(rows)
    gains.to_csv(run/"PHASE_BBLR_VS_COMPARATORS.csv",index=False)

    # Prediction-selected slope regret relative to each method's infeasible
    # slope-oracle tuning choice.  Truth is used only for evaluation.
    c=chosen.merge(
        oracle[["dgp","design","n_train","noise_sd","rep","method","test_slope_rmse"]].rename(
            columns={"test_slope_rmse":"oracle_test_slope_rmse"}),
        on=["dgp","design","n_train","noise_sd","rep","method"],how="left"
    )
    c["slope_selection_regret_ratio"]=c["test_slope_rmse"]/c["oracle_test_slope_rmse"]
    c.to_csv(run/"PHASE_PREDICTION_SELECTION_REGRET.csv.gz",index=False,compression="gzip")

    regret=c.groupby(KEYS+["method"],dropna=False).agg(
        reps=("rep","count"),
        mean_selected_slope_rmse=("test_slope_rmse","mean"),
        mean_oracle_slope_rmse=("oracle_test_slope_rmse","mean"),
        mean_regret_ratio=("slope_selection_regret_ratio","mean"),
        median_regret_ratio=("slope_selection_regret_ratio","median"),
    ).reset_index()
    regret.to_csv(run/"PHASE_SELECTION_REGRET_SUMMARY.csv",index=False)

    # Tuning-boundary audit.  Boundary choices are warning flags that a tuning
    # grid may be truncating the empirical optimum.
    b=chosen[chosen["method"]=="BBLR"].copy()
    rvals=np.sort(b["r"].dropna().unique()); evals=np.sort(b["eps"].dropna().unique())
    b["r_at_min"]=b["r"].eq(rvals.min()); b["r_at_max"]=b["r"].eq(rvals.max())
    b["eps_at_min"]=b["eps"].eq(evals.min()); b["eps_at_max"]=b["eps"].eq(evals.max())
    audit=b.groupby(KEYS,dropna=False).agg(
        reps=("rep","count"),
        r_at_min_share=("r_at_min","mean"),
        r_at_max_share=("r_at_max","mean"),
        eps_at_min_share=("eps_at_min","mean"),
        eps_at_max_share=("eps_at_max","mean"),
        mean_selected_r=("r","mean"),
        mean_selected_eps=("eps","mean"),
    ).reset_index()
    audit.to_csv(run/"PHASE_BBLR_TUNING_BOUNDARY_AUDIT.csv",index=False)

    # Family/parameter summaries preserve the controlled ladder interpretation.
    family=gains.groupby(
        ["dgp_family","dgp_parameter_name","dgp_parameter_value","comparator"],
        dropna=False
    ).agg(
        cells=("dgp","count"),
        mean_y_ratio=("y_rmse_ratio_bblr_over_comparator","mean"),
        mean_slope_ratio=("slope_rmse_ratio_bblr_over_comparator","mean"),
        bblr_dominance_share=("bblr_dominates","mean"),
        comparator_dominance_share=("comparator_dominates","mean"),
    ).reset_index()
    family.to_csv(run/"PHASE_FAMILY_SUMMARY.csv",index=False)

    print("Wrote phase-map diagnostics:")
    for fn in [
        "PHASE_BBLR_VS_COMPARATORS.csv",
        "PHASE_PREDICTION_SELECTION_REGRET.csv.gz",
        "PHASE_SELECTION_REGRET_SUMMARY.csv",
        "PHASE_BBLR_TUNING_BOUNDARY_AUDIT.csv",
        "PHASE_FAMILY_SUMMARY.csv",
    ]:
        print(run/fn)

if __name__=="__main__":
    main()
