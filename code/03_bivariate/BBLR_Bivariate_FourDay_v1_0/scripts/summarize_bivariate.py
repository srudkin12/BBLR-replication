#!/usr/bin/env python3
import argparse,glob,json
from pathlib import Path
import pandas as pd,numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True,type=Path); a=ap.parse_args(); run=a.run_dir
    files=sorted((run/'selected_shards').glob('*.csv.gz')); 
    if not files: raise SystemExit('no selected shards')
    df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    df.to_csv(run/'BIV_SELECTED_ALL.csv.gz',index=False,compression='gzip')
    keys=['cell_id','label','block','expected_regime','dgp','design','n_train_legacy','noise_sd','method']
    s=df.groupby(keys,dropna=False).agg(reps=('rep','count'),median_test_y_rmse=('test_y_rmse','median'),mean_test_y_rmse=('test_y_rmse','mean'),median_gradient_rmse=('test_gradient_rmse','median'),mean_gradient_rmse=('test_gradient_rmse','mean'),median_slope1_rmse=('test_slope1_rmse','median'),median_slope2_rmse=('test_slope2_rmse','median'),median_grid_gradient_rmse=('grid_gradient_rmse','median'),median_cv_rmse=('repeated_cv_rmse','median')).reset_index()
    s.to_csv(run/'BIV_METHOD_CELL_SUMMARY.csv',index=False)
    # Best method and BBLR pairwise by cell
    best=[]; pair=[]
    for cid,g in s.groupby('cell_id'):
        g=g.copy(); bs=g.loc[g.median_gradient_rmse.idxmin()]; by=g.loc[g.median_test_y_rmse.idxmin()]
        best.append({'cell_id':cid,'label':bs.label,'block':bs.block,'best_gradient_method':bs.method,'best_gradient_rmse':bs.median_gradient_rmse,'best_outcome_method':by.method,'best_outcome_rmse':by.median_test_y_rmse})
        if 'BBLR' in set(g.method):
            b=g[g.method=='BBLR'].iloc[0]
            for _,r in g[g.method!='BBLR'].iterrows():
                pair.append({'cell_id':cid,'label':b.label,'block':b.block,'dgp':b.dgp,'design':b.design,'comparator':r.method,'bblr_gradient_ratio':b.median_gradient_rmse/r.median_gradient_rmse,'bblr_y_ratio':b.median_test_y_rmse/r.median_test_y_rmse,'bblr_slope1_ratio':b.median_slope1_rmse/r.median_slope1_rmse,'bblr_slope2_ratio':b.median_slope2_rmse/r.median_slope2_rmse})
    pd.DataFrame(best).to_csv(run/'BIV_CELL_BEST_METHODS.csv',index=False); pd.DataFrame(pair).to_csv(run/'BIV_BBLR_PAIRWISE.csv',index=False)
    print(f'Summarized {len(files)} shards; selected rows={len(df)}')
if __name__=='__main__': main()
