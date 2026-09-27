#!/usr/bin/env python3
"""Assemble and analyse the matched-H_lambda mechanism audit."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def _bootstrap_mean_ci(x, reps=2000, seed=12345):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)==0: return (np.nan,np.nan,np.nan,np.nan)
    mean=float(np.mean(x)); se=float(np.std(x,ddof=1)/np.sqrt(len(x))) if len(x)>1 else np.nan
    rng=np.random.default_rng(seed)
    vals=np.empty(reps,float)
    n=len(x)
    for b in range(reps):
        vals[b]=np.mean(x[rng.integers(0,n,n)])
    lo,hi=np.quantile(vals,[0.025,0.975])
    return mean,se,float(lo),float(hi)


def _classify(lo,hi):
    if not np.isfinite(lo) or not np.isfinite(hi): return 'unclassified'
    if hi < 0: return 'BBLR_better'
    if lo > 0: return 'W_better'
    return 'unclear'


def pair_summary(df, group_cols, bcol, wcol, pair_name, bootstrap_reps=2000):
    rows=[]
    grouped=df.groupby(group_cols,dropna=False,sort=True) if group_cols else [((),df)]
    for key,g in grouped:
        if not isinstance(key,tuple): key=(key,)
        z=g[[bcol,wcol]].dropna()
        d=(z[bcol]-z[wcol]).to_numpy(float)
        mean,se,lo,hi=_bootstrap_mean_ci(d,bootstrap_reps,seed=9173+len(rows))
        rec={c:v for c,v in zip(group_cols,key)}
        rec.update({
            'pair':pair_name,'n_pairs':len(d),'mean_bblr_mse':float(z[bcol].mean()) if len(z) else np.nan,
            'mean_w_mse':float(z[wcol].mean()) if len(z) else np.nan,
            'median_bblr_rmse':float(np.sqrt(z[bcol]).median()) if len(z) else np.nan,
            'median_w_rmse':float(np.sqrt(z[wcol]).median()) if len(z) else np.nan,
            'bblr_win_share':float(np.mean(z[bcol].to_numpy()<z[wcol].to_numpy())) if len(z) else np.nan,
            'mean_delta_bblr_minus_w':mean,'mc_se_delta':se,'bootstrap_ci_low':lo,'bootstrap_ci_high':hi,
            'classification':_classify(lo,hi),
            'mse_ratio_bblr_over_w':float(z[bcol].mean()/z[wcol].mean()) if len(z) and z[wcol].mean()>0 else np.nan,
        })
        for q in (0.5,0.9,0.95,0.99):
            tag=str(q).replace('.','p')
            rec[f'bblr_mse_q{tag}']=float(z[bcol].quantile(q)) if len(z) else np.nan
            rec[f'w_mse_q{tag}']=float(z[wcol].quantile(q)) if len(z) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--bootstrap-reps',type=int,default=2000)
    args=ap.parse_args()
    run=args.run_dir.resolve(); shards=sorted((run/'replication_shards').glob('*.csv.gz'))
    if not shards: raise SystemExit('no replication shards found')
    df=pd.concat([pd.read_csv(p) for p in shards],ignore_index=True)
    df=df.sort_values(['cell_id','rep']).reset_index(drop=True)
    df.to_csv(run/'MATCHED_HLAMBDA_REPLICATION_METRICS.csv.gz',index=False,compression='gzip')

    # Four deliberately distinct comparisons.  Strict W on common evaluation
    # locations is primary; safeguarded W is secondary.
    pairs=[
        ('strict_test_common','bblr_wstrict_common_test_mse','wstrict_common_test_mse'),
        ('strict_supported_grid_common','bblr_wstrict_common_supported_grid_mse','wstrict_common_supported_grid_mse'),
        ('safe_test','bblr_test_mse','wsafe_test_mse'),
        ('safe_supported_grid','bblr_supported_grid_mse','wsafe_supported_grid_mse'),
    ]
    group_cols=['cell_id','landmark_label','panel_block','dgp','design','n_train_legacy','noise_sd']
    cell=[]
    pos=df[df.eps>0].copy()
    for name,b,w in pairs:
        a=pair_summary(df,group_cols,b,w,name,args.bootstrap_reps); a['subset']='all_selected'; cell.append(a)
        a=pair_summary(pos,group_cols,b,w,name,args.bootstrap_reps); a['subset']='positive_stage2'; cell.append(a)
    cell=pd.concat(cell,ignore_index=True)
    cell.to_csv(run/'CELL_PAIRWISE_SUMMARY.csv',index=False)

    # Theory-facing strata: eps=0, lambda below/equal/above 1, and Gamma bands.
    strata=[]
    for name,b,w in pairs[:2]:
        a=pair_summary(df,['lambda_stratum'],b,w,name,args.bootstrap_reps); a['stratification']='lambda'; strata.append(a)
        temp=df.copy()
        temp['gamma_band']=np.select(
            [temp['eps'].eq(0),temp['Gamma_lambda'].abs()<0.01,temp['Gamma_lambda'].abs()<0.05],
            ['eps0','absGamma_lt_0p01','absGamma_0p01_to_0p05'],default='absGamma_ge_0p05')
        a=pair_summary(temp,['gamma_band'],b,w,name,args.bootstrap_reps); a['stratification']='Gamma'; strata.append(a)
        a=pair_summary(df,['panel_block'],b,w,name,args.bootstrap_reps); a['stratification']='panel'; strata.append(a)
        a=pair_summary(df,['design'],b,w,name,args.bootstrap_reps); a['stratification']='design'; strata.append(a)
    pd.concat(strata,ignore_index=True).to_csv(run/'THEORY_STRATUM_SUMMARY.csv',index=False)

    # Diagnostics and implementation identity checks.
    eps0=df[df.eps.eq(0)]
    audit={
        'rows':int(len(df)),'cells':int(df.cell_id.nunique()),
        'min_reps_per_cell':int(df.groupby('cell_id').size().min()),'max_reps_per_cell':int(df.groupby('cell_id').size().max()),
        'positive_stage2_rows':int((df.eps>0).sum()),
        'archive_test_rmse_max_abs_diff':float(df.archive_test_rmse_abs_diff.max()),
        'archive_grid_rmse_max_abs_diff':float(df.archive_grid_rmse_abs_diff.max()),
        'eps0_rows':int(len(eps0)),
        'eps0_bblr_wsafe_test_max_abs_diff':float(eps0.eps0_max_abs_bblr_wsafe_test.max()) if len(eps0) else np.nan,
        'eps0_bblr_wsafe_grid_max_abs_diff':float(eps0.eps0_max_abs_bblr_wsafe_grid.max()) if len(eps0) else np.nan,
        'strict_test_valid_share_mean':float(df.wstrict_test_valid_share.mean()),
        'strict_supported_grid_valid_share_mean':float(df.wstrict_supported_grid_valid_share.mean()),
        'safe_test_fallback_share_mean':float(df.wsafe_test_fallback_share.mean()),
        'safe_supported_grid_fallback_share_mean':float(df.wsafe_supported_grid_fallback_share.mean()),
    }
    used = pd.read_csv(run/'FROZEN_SELECTED_CONFIGS_USED.csv.gz') if (run/'FROZEN_SELECTED_CONFIGS_USED.csv.gz').exists() else None
    expected_rows = int(len(used)) if used is not None else int(len(df))
    expected_cells = int(used.cell_id.nunique()) if used is not None else int(df.cell_id.nunique())
    audit['expected_rows']=expected_rows; audit['expected_cells']=expected_cells
    audit['status']='PASS' if (len(df)==expected_rows and df.cell_id.nunique()==expected_cells and audit['archive_test_rmse_max_abs_diff']<1e-9 and audit['archive_grid_rmse_max_abs_diff']<1e-9 and audit['eps0_bblr_wsafe_test_max_abs_diff']<1e-9 and audit['eps0_bblr_wsafe_grid_max_abs_diff']<1e-9) else 'CHECK'
    (run/'ANALYSIS_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')

    # Key-case extract requested by the v0.9 hostile reviewer plus controls.
    key_labels=[
        'support_sharp_center_D02','support_sharp_center_D10','noise_highfreq_0.1',
        'shape_quad_high','density_transition_normal','density_transition_skewed',
        'support_sharp_center_G00','shape_linear'
    ]
    cell[cell.landmark_label.isin(key_labels)].to_csv(run/'KEY_CASE_PAIRWISE_SUMMARY.csv',index=False)

    # Compact human-readable result note, intentionally descriptive only.
    strict=cell[(cell['pair']=='strict_test_common')&(cell['subset']=='positive_stage2')]
    counts=strict.classification.value_counts().to_dict()
    lines=[
        '# Matched-H-lambda audit automated summary','',
        f'- Rows assembled: {len(df):,}.', f'- Cells: {df.cell_id.nunique()}.',
        f'- Positive-Stage-2 rows: {(df.eps>0).sum():,}.',
        f'- Strict-W test-support cell classifications (positive Stage 2): {counts}.',
        f'- Mean strict-W test valid share: {df.wstrict_test_valid_share.mean():.6f}.',
        f'- Mean safeguard-W test fallback share: {df.wsafe_test_fallback_share.mean():.6f}.',
        f'- Archive reconstruction max |test RMSE difference|: {df.archive_test_rmse_abs_diff.max():.3e}.',
        f'- Archive reconstruction max |grid RMSE difference|: {df.archive_grid_rmse_abs_diff.max():.3e}.',
        '', 'Interpret classifications using paired MSE differences (BBLR minus W); negative means lower BBLR risk.',
        'Hard-gap full-grid recovery is not a regular derivative target; use supported-grid summaries and fallback diagnostics there.'
    ]
    (run/'AUTOMATED_SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__': main()
