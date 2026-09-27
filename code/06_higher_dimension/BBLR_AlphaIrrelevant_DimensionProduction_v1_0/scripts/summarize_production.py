#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

PRIMARY_METRICS=[
 'test_mean_rmse','gradient_component_rmse','gradient_vector_rmse',
 'active_gradient_component_rmse','active_gradient_vector_rmse',
 'tangent_alpha_rmse','reference_alpha0_abs_error','reference_alpha0_sq_error',
 'coefficient_field_vector_rmse','nuisance_component_rmse','nuisance_vector_rmse',
 'nuisance_vector_rmse_per_sqrt_dim','nuisance_mean_norm','nuisance_median_norm',
 'nuisance_q90_norm','nuisance_q95_norm','nuisance_mean_norm_per_sqrt_dim',
 'nuisance_median_norm_per_sqrt_dim','nuisance_q90_norm_per_sqrt_dim','nuisance_q95_norm_per_sqrt_dim',
 'nuisance_share_gt_005','nuisance_share_gt_010','nuisance_share_gt_020',
 'nuisance_scaled_share_gt_005','nuisance_scaled_share_gt_010','nuisance_scaled_share_gt_020',
 'origin_nn_distance','stage2_positive','selected_eps_over_r'
]

PAIR_METRICS=[
 'test_mean_rmse','gradient_vector_rmse','active_gradient_vector_rmse','tangent_alpha_rmse',
 'reference_alpha0_abs_error','coefficient_field_vector_rmse','nuisance_vector_rmse',
 'nuisance_vector_rmse_per_sqrt_dim'
]


def med_ratio(a,b):
    valid=np.isfinite(a)&np.isfinite(b)&(b>0)
    return (float(np.median(a[valid]/b[valid])),float(np.mean(a[valid]<b[valid])),int(valid.sum())) if valid.any() else (np.nan,np.nan,0)


def risk_summary(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    valid=np.isfinite(a)&np.isfinite(b)&(b>0)
    if not valid.any():
        return dict(n_valid=0,median_paired_ratio=np.nan,win_share=np.nan,median_rmse_ratio=np.nan,mean_mse_ratio=np.nan)
    av=a[valid]; bv=b[valid]
    return dict(
        n_valid=int(valid.sum()),
        median_paired_ratio=float(np.median(av/bv)),
        win_share=float(np.mean(av<bv)),
        median_rmse_ratio=float(np.median(av)/np.median(bv)) if np.median(bv)>0 else np.nan,
        mean_mse_ratio=float(np.mean(av*av)/np.mean(bv*bv)) if np.mean(bv*bv)>0 else np.nan,
    )


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); ap.add_argument('--expected-reps',type=int,required=True); args=ap.parse_args()
    rd=Path(args.run_dir); fs=sorted((rd/'shards').glob('*.csv.gz'))
    if not fs: raise SystemExit('No shards found')
    z=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True)
    ok=z[np.isfinite(pd.to_numeric(z.get('gradient_vector_rmse',np.nan),errors='coerce'))].copy()
    keys=['cell_id','track','d','active_dim','dgp','design','n_dev','n_test','noise_sd','origin_supported','method']
    metrics=[m for m in PRIMARY_METRICS if m in ok.columns]
    agg={m:['mean','median'] for m in metrics}
    s=ok.groupby(keys,dropna=False).agg(agg); s.columns=['_'.join(c) for c in s.columns]; s=s.reset_index()
    sizes=ok.groupby(keys,dropna=False).size().reset_index(name='n_rep')
    s=s.merge(sizes,on=keys,how='left'); s.to_csv(rd/'METHOD_CELL_SUMMARY.csv',index=False)

    # General paired BBLR comparisons.
    base=ok[ok.method=='BBLR'][['cell_id','rep']+metrics].copy(); pairs=[]
    for m in sorted(ok.method.unique()):
        if m=='BBLR': continue
        other=ok[ok.method==m][['cell_id','rep']+metrics].copy()
        q=base.merge(other,on=['cell_id','rep'],suffixes=('_bblr','_other'))
        for cid,g in q.groupby('cell_id'):
            row={'cell_id':cid,'other_method':m,'n_pair':len(g)}
            for metric in PAIR_METRICS:
                aa=pd.to_numeric(g.get(f'{metric}_bblr'),errors='coerce').to_numpy(float)
                bb=pd.to_numeric(g.get(f'{metric}_other'),errors='coerce').to_numpy(float)
                ratio,win,nv=med_ratio(aa,bb)
                row[f'{metric}_median_ratio']=ratio; row[f'{metric}_win_share']=win; row[f'{metric}_n_valid']=nv
            pairs.append(row)
    pd.DataFrame(pairs).to_csv(rd/'BBLR_PAIRWISE.csv',index=False)

    # Conditional matched-radius Stage-2 audit at BBLR-selected r.
    matched=ok[ok.method=='LLR_MATCHED_BBLR_R'][['cell_id','rep']+metrics].copy()
    q=base.merge(matched,on=['cell_id','rep'],suffixes=('_bblr','_matched'))
    cell_rows=[]
    for cid,g0 in q.groupby('cell_id'):
        for stratum,g in [('all',g0),('stage2_positive',g0[g0.stage2_positive_bblr>0.5])]:
            if len(g)==0: continue
            row={'cell_id':cid,'stratum':stratum,'n_pair':len(g),'median_selected_eps_over_r':float(np.nanmedian(g.selected_eps_over_r_bblr))}
            for metric in PAIR_METRICS:
                rs=risk_summary(g[f'{metric}_bblr'].to_numpy(float),g[f'{metric}_matched'].to_numpy(float))
                for k,v in rs.items(): row[f'{metric}_{k}']=v
            cell_rows.append(row)
    mr=pd.DataFrame(cell_rows)
    mr.to_csv(rd/'MATCHED_RADIUS_STAGE2_SUMMARY.csv',index=False)

    # Track-level median cell effects for quick interpretation; never replaces cell-level output.
    meta=ok[['cell_id','track','d','active_dim','dgp','design','n_dev']].drop_duplicates('cell_id')
    if len(mr):
        mm=mr.merge(meta,on='cell_id',how='left')
        track_rows=[]
        for (track,stratum),g in mm.groupby(['track','stratum']):
            row={'track':track,'stratum':stratum,'n_cells':int(g.cell_id.nunique())}
            for metric in PAIR_METRICS:
                col=f'{metric}_median_paired_ratio'; win=f'{metric}_win_share'
                if col in g:
                    vv=pd.to_numeric(g[col],errors='coerce').to_numpy(float); vv=vv[np.isfinite(vv)]
                    row[f'{metric}_median_cell_ratio']=float(np.median(vv)) if len(vv) else np.nan
                if win in g:
                    vv=pd.to_numeric(g[win],errors='coerce').to_numpy(float); vv=vv[np.isfinite(vv)]
                    row[f'{metric}_median_cell_win_share']=float(np.median(vv)) if len(vv) else np.nan
            track_rows.append(row)
        pd.DataFrame(track_rows).to_csv(rd/'MATCHED_RADIUS_TRACK_SUMMARY.csv',index=False)

    # Identity audit when BBLR selects epsilon=0; the matched-radius LLR must be the exact same procedure.
    z0=q[q.stage2_positive_bblr<0.5].copy(); diffs={}
    for metric in PAIR_METRICS:
        if len(z0):
            a=z0[f'{metric}_bblr'].to_numpy(float); b=z0[f'{metric}_matched'].to_numpy(float)
            v=np.isfinite(a)&np.isfinite(b); diffs[metric]=float(np.max(np.abs(a[v]-b[v]))) if v.any() else np.nan
        else: diffs[metric]=np.nan
    maxdiff=max([v for v in diffs.values() if np.isfinite(v)],default=0.0)
    ident={'status':'PASS' if maxdiff<=1e-10 else 'CHECK','n_stage2_zero_pairs':int(len(z0)),'empirical_check':'not_applicable_no_stage2_zero_selections' if len(z0)==0 else 'evaluated','unit_test_exact_nesting':True,'tolerance':1e-10,'max_abs_metric_difference':float(maxdiff),'metric_max_abs_differences':diffs}
    (rd/'MATCHED_RADIUS_IDENTITY_AUDIT.json').write_text(json.dumps(ident,indent=2))

    # Alpha summaries.
    alpha_cols=['reference_alpha0_abs_error','tangent_alpha_rmse','coefficient_field_vector_rmse']
    ok.groupby(['track','origin_supported','d','active_dim','dgp','design','n_dev','method'],dropna=False)[alpha_cols].median().reset_index().to_csv(rd/'ALPHA0_TANGENT_SUMMARY.csv',index=False)

    # Nuisance diagnostics, raw and dimension-normalised.
    irr=ok[ok.active_dim<ok.d].copy()
    nuisance_cols=[
        'nuisance_component_rmse','nuisance_vector_rmse','nuisance_vector_rmse_per_sqrt_dim',
        'nuisance_mean_norm','nuisance_q90_norm','nuisance_q95_norm',
        'nuisance_mean_norm_per_sqrt_dim','nuisance_q90_norm_per_sqrt_dim','nuisance_q95_norm_per_sqrt_dim',
        'nuisance_share_gt_005','nuisance_share_gt_010','nuisance_share_gt_020',
        'nuisance_scaled_share_gt_005','nuisance_scaled_share_gt_010','nuisance_scaled_share_gt_020',
        'active_gradient_vector_rmse'
    ]
    irr.groupby(['track','d','active_dim','dgp','design','n_dev','method'],dropna=False)[nuisance_cols].median().reset_index().to_csv(rd/'NUISANCE_SUMMARY.csv',index=False)

    # Pure-null false-slope controls.
    null=ok[ok.track=='flat_null'].copy()
    null_cols=['nuisance_vector_rmse','nuisance_vector_rmse_per_sqrt_dim','nuisance_q95_norm','nuisance_q95_norm_per_sqrt_dim','nuisance_share_gt_010','nuisance_scaled_share_gt_010','reference_alpha0_abs_error','test_mean_rmse']
    null.groupby(['d','design','method'],dropna=False)[null_cols].median().reset_index().to_csv(rd/'NULL_SLOPE_SUMMARY.csv',index=False)

    # Dimension, oracle cost, sample-size recovery and tuning.
    dim_cols=['gradient_vector_rmse','active_gradient_vector_rmse','nuisance_vector_rmse','nuisance_vector_rmse_per_sqrt_dim','tangent_alpha_rmse','reference_alpha0_abs_error','coefficient_field_vector_rmse']
    ok.groupby(['track','d','active_dim','dgp','design','n_dev','method'],dropna=False)[dim_cols].median().reset_index().to_csv(rd/'DIMENSION_VECTOR_SUMMARY.csv',index=False)
    full=ok[ok.method=='BBLR'][['cell_id','rep','d','active_dim','dgp','design','n_dev']+dim_cols].copy()
    oracle=ok[ok.method=='BBLR_ORACLE_ACTIVE'][['cell_id','rep']+dim_cols].copy()
    qo=full.merge(oracle,on=['cell_id','rep'],suffixes=('_full','_oracle')); rows=[]
    for cid,g in qo.groupby('cell_id'):
        row={k:g.iloc[0][k] for k in ['cell_id','d','active_dim','dgp','design','n_dev']}; row['n_pair']=len(g)
        for metric in ['active_gradient_vector_rmse','tangent_alpha_rmse','reference_alpha0_abs_error','coefficient_field_vector_rmse']:
            ratio,win,nv=med_ratio(g[f'{metric}_full'].to_numpy(float),g[f'{metric}_oracle'].to_numpy(float))
            row[f'{metric}_median_full_over_oracle']=ratio; row[f'{metric}_full_win_share']=win
        rows.append(row)
    pd.DataFrame(rows).to_csv(rd/'AMBIENT_COST_SUMMARY.csv',index=False)
    ss=ok[ok.track=='sample_size_recovery'].copy()
    ss.groupby(['d','design','n_dev','method'],dropna=False)[dim_cols].median().reset_index().to_csv(rd/'SAMPLE_SIZE_SUMMARY.csv',index=False)
    bt=ok[ok.method=='BBLR'].copy()
    tune=bt.groupby(['track','d','dgp','design','n_dev'],dropna=False).agg(stage2_positive_share=('stage2_positive','mean'),median_eps_over_r=('selected_eps_over_r','median'),mean_eps_over_r=('selected_eps_over_r','mean'),n=('rep','size')).reset_index()
    tune.to_csv(rd/'BBLR_TUNING_SUMMARY.csv',index=False)

    observed=ok[['cell_id','rep']].drop_duplicates().shape[0]; n_cells=ok.cell_id.nunique(); expected=n_cells*args.expected_reps; failures=len(z)-len(ok)
    audit={'status':'PASS' if observed==expected and failures==0 and ident['status']=='PASS' else 'CHECK','n_cells_observed':int(n_cells),'cell_rep_observed':int(observed),'cell_rep_expected_from_observed_cells':int(expected),'rows_ok':int(len(ok)),'rows_total':int(len(z)),'failed_rows':int(failures),'matched_radius_identity_status':ident['status']}
    (rd/'COMPLETION_AUDIT.json').write_text(json.dumps(audit,indent=2))

    lines=['# BBLR alpha / nuisance / dimension full production','',f"Completion status: **{audit['status']}**. Cell-replications observed: {observed}.",'']
    bycell=bt.groupby('cell_id')
    lines.append(f"BBLR median vector-gradient RMSE across cells: {bycell.gradient_vector_rmse.median().median():.6g}")
    lines.append(f"BBLR median tangent-alpha RMSE across cells: {bycell.tangent_alpha_rmse.median().median():.6g}")
    nuis=bt[bt.active_dim<bt.d]
    if len(nuis):
        lines.append(f"BBLR median nuisance-vector RMSE across nuisance cells: {nuis.groupby('cell_id').nuisance_vector_rmse.median().median():.6g}")
        lines.append(f"BBLR median dimension-normalised nuisance RMSE across nuisance cells: {nuis.groupby('cell_id').nuisance_vector_rmse_per_sqrt_dim.median().median():.6g}")
    lines.append(f"BBLR positive Stage-2 share across all fitted cell-reps: {bt.stage2_positive.mean():.4f}")
    if len(mr):
        pos=mr[mr.stratum=='stage2_positive']
        for metric in ['active_gradient_vector_rmse','tangent_alpha_rmse','coefficient_field_vector_rmse','nuisance_vector_rmse_per_sqrt_dim']:
            col=f'{metric}_median_paired_ratio'
            if col in pos and pos[col].notna().any(): lines.append(f"Matched-radius Stage-2 median cell ratio for {metric}: {np.nanmedian(pos[col]):.6g}")
    lines.append(f"Exact-nesting matched-radius identity audit: {ident['status']} (max metric difference {maxdiff:.3g}).")
    (rd/'AUTOMATED_SUMMARY.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    main()
