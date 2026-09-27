#!/usr/bin/env python3
"""Summarise landmark runs, including support-aware methods and feasible BBLR selectors."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

REPKEY=['dgp','design','n_train','noise_sd','rep']
RULES={
 'minY':'selected_by_validation_y',
 '1SE_low_smoothing':'selected_by_validation_y_1se_low_smoothing',
 'tol0p5_low_smoothing':'selected_by_validation_y_tol_0p5_low_smoothing',
 'tol1p0_low_smoothing':'selected_by_validation_y_tol_1p0_low_smoothing',
 'slope_oracle':'oracle_slope_within_method',
}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True,type=Path); args=ap.parse_args()
    run=args.run_dir.resolve(); shards=sorted((run/'shards').glob('*.csv.gz'))
    frames=[]
    for i,p in enumerate(shards,1):
        df=pd.read_csv(p)
        keep=np.zeros(len(df),dtype=bool)
        for col in RULES.values():
            if col in df: keep |= df[col].fillna(False).astype(bool).to_numpy()
        frames.append(df.loc[keep].copy())
        if i%100==0 or i==len(shards): print(f'Read {i}/{len(shards)} shards')
    d=pd.concat(frames,ignore_index=True)
    long=[]
    for name,col in RULES.items():
        if col not in d: continue
        z=d[d[col].fillna(False).astype(bool)].copy(); z['selection_rule']=name; long.append(z)
    out=pd.concat(long,ignore_index=True)
    out.to_csv(run/'POWER_SELECTED_RULES.csv.gz',index=False,compression='gzip')
    keys=['dgp','dgp_label','design','design_label','n_train','noise_sd','method','selection_rule']
    summ=out.groupby(keys,dropna=False).agg(
        reps=('rep','count'),
        mean_test_y_rmse=('test_y_rmse','mean'),median_test_y_rmse=('test_y_rmse','median'),
        mean_slope_rmse=('test_slope_rmse','mean'),median_slope_rmse=('test_slope_rmse','median'),
        mean_r=('r','mean'),mean_eps=('eps','mean'),mean_k=('k','mean'),
        mean_gap_factor=('gap_factor','mean'),
        mean_components=('n_support_components','mean') if 'n_support_components' in out else ('rep','count'),
    ).reset_index()
    summ.to_csv(run/'POWER_LANDMARK_SUMMARY.csv',index=False)

    # BBLR Stage-2 power: compare each feasible BBLR selection with eps=0
    # configuration having the same r when available, and with selected LLR.
    b=out[out.method.eq('BBLR') & ~out.selection_rule.eq('slope_oracle')].copy()
    ll=out[out.method.eq('LLR') & out.selection_rule.eq('minY')][REPKEY+['test_y_rmse','test_slope_rmse']].rename(
        columns={'test_y_rmse':'llr_test_y_rmse','test_slope_rmse':'llr_slope_rmse'})
    b=b.merge(ll,on=REPKEY,how='left')
    b['slope_ratio_bblr_over_selected_llr']=b.test_slope_rmse/b.llr_slope_rmse
    b['y_ratio_bblr_over_selected_llr']=b.test_y_rmse/b.llr_test_y_rmse
    b.to_csv(run/'POWER_BBLR_STAGE2_SELECTED.csv.gz',index=False,compression='gzip')

    # Component-aware comparison under the ordinary min-Y rule.
    m=out[out.selection_rule.eq('minY')]
    piv=m.pivot_table(index=REPKEY,columns='method',values=['test_y_rmse','test_slope_rmse'])
    rows=[]
    for comp in ['CSPLINE','CAKLLR','CAKLP2','SPLINE','AKLLR','AKLP2']:
        if ('test_slope_rmse','BBLR') not in piv or ('test_slope_rmse',comp) not in piv: continue
        x=pd.DataFrame({
            'bblr_slope':piv[('test_slope_rmse','BBLR')], 'comp_slope':piv[('test_slope_rmse',comp)],
            'bblr_y':piv[('test_y_rmse','BBLR')], 'comp_y':piv[('test_y_rmse',comp)]}).dropna()
        x['comparator']=comp; x['slope_ratio']=x.bblr_slope/x.comp_slope; x['y_ratio']=x.bblr_y/x.comp_y
        x=x.reset_index(); rows.append(x)
    if rows:
        pw=pd.concat(rows,ignore_index=True); pw.to_csv(run/'POWER_SUPPORT_AWARE_PAIRWISE.csv.gz',index=False,compression='gzip')
        ps=pw.groupby(['dgp','design','n_train','noise_sd','comparator']).agg(
            reps=('rep','count'),mean_slope_ratio=('slope_ratio','mean'),median_slope_ratio=('slope_ratio','median'),
            bblr_slope_win_share=('slope_ratio',lambda z:float(np.mean(z<1))),mean_y_ratio=('y_ratio','mean'),
            bblr_y_win_share=('y_ratio',lambda z:float(np.mean(z<1))),
        ).reset_index(); ps.to_csv(run/'POWER_SUPPORT_AWARE_PAIRWISE_SUMMARY.csv',index=False)
    print('Wrote landmark power analysis outputs')

if __name__=='__main__': main()
