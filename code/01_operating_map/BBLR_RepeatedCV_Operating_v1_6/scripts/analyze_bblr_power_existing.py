#!/usr/bin/env python3
"""Intensive analysis of BBLR tuning clouds from an existing raw-shard run.

No new Monte Carlo is performed.  The script streams the authoritative shard
archive, retains BBLR configuration rows, and quantifies:
- alignment of validation-Y error with slope error;
- prediction-surface flatness;
- infeasible slope opportunity inside prediction-near-optimal sets;
- feasible low-smoothing selectors at 0.5%, 1%, and one-SE tolerances;
- Stage-2 gain relative to outcome-selected LLR where available.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from joblib import Parallel, delayed, parallel_config

REPKEY=['dgp','design','n_train','noise_sd','rep']
META=['dgp_label','dgp_family','dgp_parameter_value','dgp_secondary_parameter_value',
      'design_label','design_family','design_parameter_value','design_secondary_parameter_value']


def safe_ratio(a,b):
    return float(a/b) if np.isfinite(a) and np.isfinite(b) and b!=0 else np.nan


def choose_low_smoothing(g):
    return g.sort_values(['eps','r','min_stage2'],na_position='last',kind='mergesort').iloc[0]


def one_rep(g, llr=None):
    g=g[np.isfinite(g.validation_y_rmse) & np.isfinite(g.test_slope_rmse)].copy()
    if g.empty: return None
    ysel=g.sort_values(['validation_y_rmse','eps','r'],kind='mergesort').iloc[0]
    oracle=g.loc[g.test_slope_rmse.idxmin()]
    rho=spearmanr(g.validation_y_rmse,g.test_slope_rmse,nan_policy='omit').statistic if len(g)>2 else np.nan
    base={c:ysel[c] for c in REPKEY+META if c in ysel.index}
    base.update({
        'configs':len(g),'spearman_validationY_vs_slope':rho,
        'selected_r':ysel.r,'selected_eps':ysel.eps,
        'selected_validation_y_rmse':ysel.validation_y_rmse,
        'selected_test_y_rmse':ysel.test_y_rmse,
        'selected_slope_rmse':ysel.test_slope_rmse,
        'oracle_r':oracle.r,'oracle_eps':oracle.eps,
        'oracle_slope_rmse':oracle.test_slope_rmse,
        'selected_slope_regret':safe_ratio(ysel.test_slope_rmse,oracle.test_slope_rmse),
    })
    if llr is not None and len(llr):
        lr=llr.sort_values(['validation_y_rmse','r'],kind='mergesort').iloc[0]
        base['selected_llr_test_y_rmse']=lr.test_y_rmse
        base['selected_llr_slope_rmse']=lr.test_slope_rmse
        base['stage2_slope_ratio_bblr_over_llr']=safe_ratio(ysel.test_slope_rmse,lr.test_slope_rmse)
        base['stage2_y_ratio_bblr_over_llr']=safe_ratio(ysel.test_y_rmse,lr.test_y_rmse)
    ymin=float(ysel.validation_y_rmse)
    for tol,tag in [(0.001,'0p1'),(0.005,'0p5'),(0.010,'1p0')]:
        z=g[g.validation_y_rmse <= ymin*(1+tol)+1e-15]
        low=choose_low_smoothing(z)
        best=z.loc[z.test_slope_rmse.idxmin()]
        base[f'n_configs_within_{tag}pct']=len(z)
        base[f'share_configs_within_{tag}pct']=len(z)/len(g)
        base[f'low_smoothing_eps_{tag}pct']=low.eps
        base[f'low_smoothing_r_{tag}pct']=low.r
        base[f'low_smoothing_slope_rmse_{tag}pct']=low.test_slope_rmse
        base[f'low_smoothing_slope_regret_{tag}pct']=safe_ratio(low.test_slope_rmse,oracle.test_slope_rmse)
        base[f'best_slope_within_{tag}pct']=best.test_slope_rmse
        base[f'latent_slope_opportunity_ratio_{tag}pct']=safe_ratio(best.test_slope_rmse,ysel.test_slope_rmse)
    return base


def process_shard(path):
    df=pd.read_csv(path)
    b=df[df.method.eq('BBLR')]
    l=df[df.method.eq('LLR')]
    lmap={k:g for k,g in l.groupby(REPKEY,sort=False)}
    rows=[]
    for key,g in b.groupby(REPKEY,sort=False):
        r=one_rep(g,lmap.get(key))
        if r is not None: rows.append(r)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',required=True,type=Path)
    ap.add_argument('--output-dir',type=Path,default=None)
    ap.add_argument('--jobs',type=int,default=1)
    args=ap.parse_args()
    run=args.run_dir.resolve(); shards=sorted((run/'shards').glob('*.csv.gz'))
    if not shards: raise SystemExit(f'No shards found under {run}/shards')
    out=(args.output_dir or run/'power_analysis').resolve(); out.mkdir(parents=True,exist_ok=True)
    if args.jobs==1:
        parts=[process_shard(p) for p in shards]
    else:
        with parallel_config(backend='loky',n_jobs=args.jobs,inner_max_num_threads=1):
            parts=Parallel(verbose=10,batch_size=2,pre_dispatch='2*n_jobs')(
                delayed(process_shard)(p) for p in shards)
    rows=[r for part in parts for r in part]
    rep=pd.DataFrame(rows)
    rep.to_csv(out/'POWER_BBLR_REPLICATION_DIAGNOSTICS.csv.gz',index=False,compression='gzip')
    cellkeys=[c for c in REPKEY[:-1]+META if c in rep.columns]
    metrics=[c for c in rep.columns if c not in cellkeys+['rep'] and pd.api.types.is_numeric_dtype(rep[c])]
    agg={}
    for c in metrics:
        agg[f'mean_{c}']=(c,'mean'); agg[f'median_{c}']=(c,'median')
    summary=rep.groupby(cellkeys,dropna=False).agg(reps=('rep','count'),**agg).reset_index()
    summary.to_csv(out/'POWER_BBLR_CELL_SUMMARY.csv',index=False)
    # Compact global summary for immediate interpretation.
    glob={
        'replications':int(len(rep)),
        'median_selected_slope_regret':float(rep.selected_slope_regret.median()),
        'mean_selected_slope_regret':float(rep.selected_slope_regret.mean()),
        'median_stage2_slope_ratio_bblr_over_llr':float(rep.stage2_slope_ratio_bblr_over_llr.median()),
        'median_spearman_validationY_vs_slope':float(rep.spearman_validationY_vs_slope.median()),
    }
    for tag in ['0p1','0p5','1p0']:
        glob[f'median_share_configs_within_{tag}pct']=float(rep[f'share_configs_within_{tag}pct'].median())
        glob[f'median_low_smoothing_slope_regret_{tag}pct']=float(rep[f'low_smoothing_slope_regret_{tag}pct'].median())
        glob[f'median_latent_slope_opportunity_ratio_{tag}pct']=float(rep[f'latent_slope_opportunity_ratio_{tag}pct'].median())
    (out/'POWER_GLOBAL_SUMMARY.json').write_text(json.dumps(glob,indent=2)+'\n')
    print(json.dumps(glob,indent=2))
    print(f'Outputs: {out}')

if __name__=='__main__': main()
