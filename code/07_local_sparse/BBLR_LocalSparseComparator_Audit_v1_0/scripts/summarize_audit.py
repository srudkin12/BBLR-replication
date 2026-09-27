#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

METRICS=['test_mean_rmse','gradient_vector_rmse','active_gradient_vector_rmse','tangent_alpha_rmse','reference_alpha0_abs_error','joint_coordinate_rmse_scale_specific','nuisance_vector_rmse','nuisance_vector_rmse_per_sqrt_dim','nuisance_zero_share','active_zero_share','slope_zero_share']
RISK_METRICS=['test_mean_rmse','gradient_vector_rmse','active_gradient_vector_rmse','tangent_alpha_rmse','reference_alpha0_abs_error','joint_coordinate_rmse_scale_specific','nuisance_vector_rmse_per_sqrt_dim']

def _pair(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); v=np.isfinite(a)&np.isfinite(b)&(b>0)
    if not v.any(): return {'n_valid':0,'median_paired_ratio':np.nan,'win_share':np.nan,'median_rmse_ratio':np.nan,'mean_mse_ratio':np.nan}
    a=a[v]; b=b[v]
    return {'n_valid':int(len(a)),'median_paired_ratio':float(np.median(a/b)),'win_share':float(np.mean(a<b)),'median_rmse_ratio':float(np.median(a)/np.median(b)) if np.median(b)>0 else np.nan,'mean_mse_ratio':float(np.mean(a*a)/np.mean(b*b)) if np.mean(b*b)>0 else np.nan}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); ap.add_argument('--expected-reps',type=int,required=True); args=ap.parse_args()
    rd=Path(args.run_dir); fs=sorted((rd/'shards').glob('*.csv.gz'))
    if not fs: raise SystemExit('No shards found')
    z=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True)
    ok=z[np.isfinite(pd.to_numeric(z.get('gradient_vector_rmse',np.nan),errors='coerce'))].copy()
    keys=['cell_id','original_cell_id','audit_role','d','active_dim','dgp','design','n_dev','method']
    metrics=[m for m in METRICS if m in ok]
    agg={m:['mean','median'] for m in metrics}; s=ok.groupby(keys,dropna=False).agg(agg); s.columns=['_'.join(x) for x in s.columns]; s=s.reset_index()
    s=s.merge(ok.groupby(keys,dropna=False).size().reset_index(name='n_rep'),on=keys,how='left')
    s.to_csv(rd/'METHOD_CELL_SUMMARY.csv',index=False)

    base=ok[ok.method=='BBLR'][['cell_id','rep']+metrics].copy(); rows=[]
    for method in ['LL_ALASSO','LLR','LASSO','OLS']:
        oth=ok[ok.method==method][['cell_id','rep']+metrics].copy(); q=base.merge(oth,on=['cell_id','rep'],suffixes=('_bblr','_other'))
        for cid,g in q.groupby('cell_id'):
            meta=ok[ok.cell_id==cid].iloc[0]
            row={'cell_id':cid,'original_cell_id':meta.original_cell_id,'audit_role':meta.audit_role,'other_method':method,'n_pair':len(g)}
            for metric in RISK_METRICS:
                rs=_pair(g[f'{metric}_bblr'],g[f'{metric}_other'])
                for k,v in rs.items(): row[f'{metric}_{k}']=v
            rows.append(row)
    pd.DataFrame(rows).to_csv(rd/'BBLR_PAIRWISE.csv',index=False)

    # Direct sparse benchmark frontier: LL-ALASSO relative to BBLR and LLR.
    frontier=[]
    for cid,g0 in ok.groupby('cell_id'):
        meta=g0.iloc[0]
        for left,right in [('BBLR','LL_ALASSO'),('LL_ALASSO','LLR'),('LL_ALASSO','LASSO')]:
            a=g0[g0.method==left][['rep']+metrics]; b=g0[g0.method==right][['rep']+metrics]
            q=a.merge(b,on='rep',suffixes=('_left','_right'))
            row={'cell_id':cid,'original_cell_id':meta.original_cell_id,'audit_role':meta.audit_role,'left_method':left,'right_method':right,'n_pair':len(q)}
            for metric in RISK_METRICS:
                rs=_pair(q[f'{metric}_left'],q[f'{metric}_right'])
                for k,v in rs.items(): row[f'{metric}_{k}']=v
            frontier.append(row)
    pd.DataFrame(frontier).to_csv(rd/'LOCAL_SPARSE_FRONTIER.csv',index=False)

    al=ok[ok.method=='LL_ALASSO'].copy()
    tune=al.groupby(['cell_id','original_cell_id','audit_role','d','dgp','design'],dropna=False).agg(
        n=('rep','size'),positive_penalty_share=('selected_positive_local_penalty','mean'),median_lambda_frac=('lambda_frac','median'),mean_lambda_frac=('lambda_frac','mean'),median_cv_rmse=('cv_rmse','median'),median_slope_zero_share=('slope_zero_share','median'),median_active_zero_share=('active_zero_share','median'),median_nuisance_zero_share=('nuisance_zero_share','median')
    ).reset_index()
    tune.to_csv(rd/'LOCAL_SPARSE_TUNING_SUMMARY.csv',index=False)

    # Negative-control and nonlinear-frontier compact summaries.
    roles=ok[['cell_id','audit_role','d','dgp','design','active_dim']].drop_duplicates('cell_id')
    pair=pd.read_csv(rd/'BBLR_PAIRWISE.csv').merge(roles,on=['cell_id','audit_role'],how='left')
    pair[pair.other_method=='LL_ALASSO'].to_csv(rd/'BBLR_VS_LOCAL_SPARSE.csv',index=False)

    observed=ok[['cell_id','rep']].drop_duplicates().shape[0]; n_cells=ok.cell_id.nunique(); expected=n_cells*args.expected_reps; failures=len(z)-len(ok)
    counts=ok.groupby(['cell_id','rep']).method.nunique(); expected_methods=5
    method_complete=bool(len(counts)==expected and (counts==expected_methods).all())
    audit={'status':'PASS' if observed==expected and failures==0 and method_complete else 'CHECK','n_cells_observed':int(n_cells),'cell_rep_observed':int(observed),'cell_rep_expected':int(expected),'rows_ok':int(len(ok)),'rows_total':int(len(z)),'failed_rows':int(failures),'expected_methods_per_cell_rep':expected_methods,'all_cell_reps_have_all_methods':method_complete}
    (rd/'COMPLETION_AUDIT.json').write_text(json.dumps(audit,indent=2))

    lines=['# Targeted local-sparse comparator audit','',f"Completion: **{audit['status']}**; cell-replications {observed}/{expected}; failures {failures}.",'']
    if len(al):
        lines.append(f"LL-ALASSO positive-penalty selection share: {al.selected_positive_local_penalty.mean():.4f}.")
        lines.append(f"LL-ALASSO median nuisance exact-zero share across nuisance cells: {al[al.active_dim<al.d].groupby('cell_id').nuisance_zero_share.median().median():.4f}.")
    bv=pair[pair.other_method=='LL_ALASSO']
    for metric in ['active_gradient_vector_rmse','nuisance_vector_rmse_per_sqrt_dim','tangent_alpha_rmse','joint_coordinate_rmse_scale_specific']:
        col=f'{metric}_median_paired_ratio'
        if col in bv and bv[col].notna().any():
            lines.append(f"Median cell BBLR / LL-ALASSO paired ratio for {metric}: {np.nanmedian(bv[col]):.6g}.")
    lines.append('Interpret cell-by-cell and preserve null/linear/high-frequency negative controls; no universal-dominance criterion is predeclared.')
    (rd/'AUTOMATED_SUMMARY.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__': main()
