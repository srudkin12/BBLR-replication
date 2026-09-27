#!/usr/bin/env python3
"""Summarize the repeated-CV strong-comparator confirmation run."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

KEY=['cell_id','label','panel_block','expected_regime','dgp','design','n_train_legacy','noise_sd']

def _pareto(x,y):
    n=len(x); out=np.ones(n,dtype=bool)
    for i in range(n):
        if not (np.isfinite(x[i]) and np.isfinite(y[i])): out[i]=False; continue
        dom=(x<=x[i])&(y<=y[i])&((x<x[i])|(y<y[i]))
        dom[i]=False
        if dom.any(): out[i]=False
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True,type=Path); args=ap.parse_args()
    run=args.run_dir.resolve(); audit=json.loads((run/'COMPLETION_AUDIT.json').read_text())
    if audit.get('status')!='PASS': raise SystemExit('completion audit is not PASS')
    paths=sorted((run/'selected_shards').glob('*_selected.csv.gz'))
    if not paths: raise SystemExit('no selected shards')
    frames=[pd.read_csv(p) for p in paths]; df=pd.concat(frames,ignore_index=True)
    df.to_csv(run/'SC_SELECTED_ALL.csv.gz',index=False,compression='gzip')

    cell_method=df.groupby(KEY+['method'],dropna=False).agg(
        reps=('rep','count'),
        mean_cv_rmse=('repeated_cv_rmse','mean'), median_cv_rmse=('repeated_cv_rmse','median'),
        mean_test_y_rmse=('test_y_rmse','mean'), median_test_y_rmse=('test_y_rmse','median'),
        mean_test_slope_rmse=('test_slope_rmse','mean'), median_test_slope_rmse=('test_slope_rmse','median'),
        mean_grid_slope_rmse=('grid_slope_rmse','mean'), median_grid_slope_rmse=('grid_slope_rmse','median'),
    ).reset_index()
    cell_method.to_csv(run/'SC_CELL_METHOD_SUMMARY.csv',index=False)

    # Paired BBLR comparisons use identical simulated datasets and CV splits.
    b=df[df.method.eq('BBLR')].copy()
    pair_rows=[]
    for comp in sorted(m for m in df.method.unique() if m!='BBLR'):
        c=df[df.method.eq(comp)].copy()
        m=b.merge(c,on=KEY+['rep'],suffixes=('_bblr','_comp'),how='inner')
        if not len(m): continue
        m['slope_ratio']=m.test_slope_rmse_bblr/m.test_slope_rmse_comp
        m['y_ratio']=m.test_y_rmse_bblr/m.test_y_rmse_comp
        m['bblr_slope_win']=m.test_slope_rmse_bblr<m.test_slope_rmse_comp
        m['bblr_y_win']=m.test_y_rmse_bblr<m.test_y_rmse_comp
        m['bblr_dominates']=m.bblr_slope_win&m.bblr_y_win
        g=m.groupby(KEY,dropna=False).agg(
            reps=('rep','count'),median_slope_ratio=('slope_ratio','median'),mean_slope_ratio=('slope_ratio','mean'),
            bblr_slope_win_share=('bblr_slope_win','mean'),median_y_ratio=('y_ratio','median'),
            bblr_y_win_share=('bblr_y_win','mean'),bblr_dominance_share=('bblr_dominates','mean'),
        ).reset_index(); g['comparator']=comp; pair_rows.append(g)
    pair=pd.concat(pair_rows,ignore_index=True) if pair_rows else pd.DataFrame()
    pair.to_csv(run/'SC_BBLR_PAIRWISE_CELL_SUMMARY.csv',index=False)

    # Pareto membership among outcome-selected methods in each replication.
    prow=[]
    for keys,g in df.groupby(KEY+['rep'],dropna=False):
        gg=g.copy(); mask=_pareto(gg.test_y_rmse.to_numpy(float),gg.test_slope_rmse.to_numpy(float))
        for method,on in zip(gg.method,mask):
            prow.append(dict(zip(KEY+['rep'],keys))|{'method':method,'pareto':bool(on)})
    pareto=pd.DataFrame(prow)
    ps=pareto.groupby(KEY+['method'],dropna=False).pareto.mean().reset_index(name='pareto_share')
    ps.to_csv(run/'SC_PARETO_CELL_SUMMARY.csv',index=False)

    # Block-level pairwise picture.
    if len(pair):
        block=pair.groupby(['panel_block','comparator'],dropna=False).agg(
            cells=('label','count'),median_cell_slope_ratio=('median_slope_ratio','median'),
            mean_cell_slope_ratio=('median_slope_ratio','mean'),mean_bblr_slope_win_share=('bblr_slope_win_share','mean'),
            mean_bblr_dominance_share=('bblr_dominance_share','mean'),
        ).reset_index()
        block.to_csv(run/'SC_BLOCK_PAIRWISE_SUMMARY.csv',index=False)

    # Per-cell best methods by median outcome and median slope, without pretending
    # the two objectives are interchangeable.
    best=[]
    for keys,g in cell_method.groupby(KEY,dropna=False):
        gy=g.sort_values(['median_test_y_rmse','method']); gs=g.sort_values(['median_test_slope_rmse','method'])
        best.append(dict(zip(KEY,keys))|{
            'best_outcome_method':gy.iloc[0].method,'best_outcome_rmse':gy.iloc[0].median_test_y_rmse,
            'best_slope_method':gs.iloc[0].method,'best_slope_rmse':gs.iloc[0].median_test_slope_rmse,
            'bblr_slope_rank':int(gs.reset_index(drop=True).index[gs.method.eq('BBLR')][0]+1) if (gs.method=='BBLR').any() else np.nan,
            'n_methods':len(g),
        })
    pd.DataFrame(best).to_csv(run/'SC_CELL_BEST_METHODS.csv',index=False)
    print(f'Summarized {len(paths)} selected shards and {df.rep.nunique()} replication IDs across {df.label.nunique()} cells')
if __name__=='__main__': main()
