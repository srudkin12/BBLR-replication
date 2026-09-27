#!/usr/bin/env python3
"""Analyze the v1.6 1000-replication BBLR repeated-CV operating panel."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

KEY=["cell_id","landmark_label","panel_block","expected_regime","dgp","design","n_train_legacy","noise_sd"]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True,type=Path)
    args=ap.parse_args(); run=args.run_dir.resolve()
    s=pd.read_csv(run/'RCV_SELECTOR_ROWS.csv.gz')
    # Core selector table.
    keep=s[s.selector.isin(['selected_single_split','selected_cv5','selected_repeated_cv','llr_selected_repeated_cv','oracle_slope'])].copy()
    piv=keep.pivot_table(index=KEY+['rep'],columns='selector',values=['test_y_rmse','test_slope_rmse','slope_regret_ratio','r','eps'],aggfunc='first')
    piv.columns=['__'.join(map(str,c)) for c in piv.columns]; piv=piv.reset_index()
    def col(metric,sel): return f'{metric}__{sel}'
    # Repeated-CV gains relative to single split and nested LLR.
    piv['rcv_vs_single_slope_ratio']=piv[col('test_slope_rmse','selected_repeated_cv')]/piv[col('test_slope_rmse','selected_single_split')]
    piv['rcv_vs_cv5_slope_ratio']=piv[col('test_slope_rmse','selected_repeated_cv')]/piv[col('test_slope_rmse','selected_cv5')]
    piv['bblr_vs_llr_slope_ratio']=piv[col('test_slope_rmse','selected_repeated_cv')]/piv[col('test_slope_rmse','llr_selected_repeated_cv')]
    piv['bblr_vs_llr_y_ratio']=piv[col('test_y_rmse','selected_repeated_cv')]/piv[col('test_y_rmse','llr_selected_repeated_cv')]
    piv['stage2_used']=~np.isclose(piv[col('eps','selected_repeated_cv')],0.0)
    piv.to_csv(run/'RCV1000_REPLICATION_DIAGNOSTICS.csv.gz',index=False,compression='gzip')

    metrics={
      'reps':('rep','nunique'),
      'median_rcv_vs_single_slope_ratio':('rcv_vs_single_slope_ratio','median'),
      'mean_rcv_vs_single_slope_ratio':('rcv_vs_single_slope_ratio','mean'),
      'rcv_beats_single_slope_share':('rcv_vs_single_slope_ratio',lambda x:float(np.mean(x<1))),
      'median_rcv_vs_cv5_slope_ratio':('rcv_vs_cv5_slope_ratio','median'),
      'rcv_beats_cv5_slope_share':('rcv_vs_cv5_slope_ratio',lambda x:float(np.mean(x<1))),
      'median_bblr_vs_llr_slope_ratio':('bblr_vs_llr_slope_ratio','median'),
      'mean_bblr_vs_llr_slope_ratio':('bblr_vs_llr_slope_ratio','mean'),
      'bblr_beats_llr_slope_share':('bblr_vs_llr_slope_ratio',lambda x:float(np.mean(x<1))),
      'median_bblr_vs_llr_y_ratio':('bblr_vs_llr_y_ratio','median'),
      'stage2_use_share':('stage2_used','mean'),
    }
    cell=piv.groupby(KEY,dropna=False).agg(**metrics).reset_index()
    cell.to_csv(run/'RCV1000_CELL_OPERATING_SUMMARY.csv',index=False)
    block=piv.groupby(['panel_block'],dropna=False).agg(**metrics).reset_index()
    block.to_csv(run/'RCV1000_BLOCK_SUMMARY.csv',index=False)
    regime=piv.groupby(['expected_regime'],dropna=False).agg(**metrics).reset_index()
    regime.to_csv(run/'RCV1000_EXPECTED_REGIME_SUMMARY.csv',index=False)

    # Merge tuning geometry to the cell summary for a single analysis table.
    geom=pd.read_csv(run/'RCV_TUNING_GEOMETRY_SUMMARY.csv')
    merge_keys=['cell_id','landmark_label','panel_block','expected_regime','dgp','design']
    master=cell.merge(geom,on=merge_keys,how='left',validate='one_to_one')
    master.to_csv(run/'RCV1000_MASTER_CELL_SUMMARY.csv',index=False)
    print('Wrote v1.6 operating-characteristics summaries')

if __name__=='__main__': main()
