#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

def safe_ratio(a,b):return float(a/b) if np.isfinite(a) and np.isfinite(b) and b!=0 else np.nan

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);a=ap.parse_args();run=a.run_dir.resolve();files=sorted((run/'shards').glob('*.csv.gz'))
 if not files:raise SystemExit('No replay shards found')
 frames=[pd.read_csv(f) for f in files];df=pd.concat(frames,ignore_index=True);df.to_csv(run/'REPLAY_RESULTS_FULL.csv.gz',index=False,compression='gzip')
 ok=df[df.status.eq('ok')].copy(); fail=df[~df.status.eq('ok')].copy()

 # compact replication handback
 compact_cols=[c for c in ['suite','cell_id','rep','label','block','expected_regime','dgp','design','method','selection_role','r','eps','k','lam','n_knots','alpha','gap_factor','origin_support_class','test_tangent_alpha_rmse','test_tangent_alpha_bias','grid_tangent_alpha_rmse_supported','alpha0_true','alpha0_hat','alpha0_signed_error','alpha0_abs_error','origin_nearest_distance_dev','test_slope_rmse_replay','test_gradient_rmse_replay'] if c in ok.columns]
 ok[compact_cols].to_csv(run/'LEGACY_COEFFICIENT_FIELD_REPLICATIONS.csv.gz',index=False,compression='gzip')
 # support-stratified reference-alpha summary
 sup=ok.groupby(['suite','origin_support_class','method'],dropna=False).agg(n=('rep','size'),alpha0_abs_error_median=('alpha0_abs_error','median'),alpha0_rmse=('alpha0_sq_error',lambda x:float(np.sqrt(np.mean(x)))),tangent_alpha_rmse_median=('test_tangent_alpha_rmse','median')).reset_index()
 sup.to_csv(run/'REFERENCE_ALPHA_SUPPORT_SUMMARY.csv',index=False)
 # replay integrity
 checks=[]
 pairs=[('old_test_y_rmse','test_y_rmse_replay'),('old_test_mean_rmse','test_mean_rmse_replay'),('old_test_slope_rmse','test_slope_rmse_replay'),('old_grid_slope_rmse','grid_slope_rmse_replay'),('old_test_gradient_rmse','test_gradient_rmse_replay'),('old_test_slope1_rmse','test_slope1_rmse_replay'),('old_test_slope2_rmse','test_slope2_rmse_replay'),('old_grid_gradient_rmse','grid_gradient_rmse_replay')]
 maxerr=0.0
 for old,new in pairs:
  if old in ok.columns and new in ok.columns:
   z=ok[[old,new]].dropna(); err=float(np.max(np.abs(z[old]-z[new]))) if len(z) else np.nan
   checks.append({'old_metric':old,'replay_metric':new,'n':len(z),'max_abs_error':err});
   if np.isfinite(err):maxerr=max(maxerr,err)
 pd.DataFrame(checks).to_csv(run/'REPLAY_INTEGRITY_METRICS.csv',index=False)
 # method-cell summaries
 groupcols=['suite','cell_id','label','block','expected_regime','dgp','design','method','origin_support_class']
 agg=ok.groupby(groupcols,dropna=False).agg(n=('rep','size'),test_tangent_alpha_rmse_median=('test_tangent_alpha_rmse','median'),test_tangent_alpha_rmse_mean=('test_tangent_alpha_rmse','mean'),test_tangent_alpha_bias_mean=('test_tangent_alpha_bias','mean'),grid_tangent_alpha_rmse_supported_median=('grid_tangent_alpha_rmse_supported','median'),alpha0_abs_error_median=('alpha0_abs_error','median'),alpha0_rmse=('alpha0_sq_error',lambda x:float(np.sqrt(np.mean(x)))),alpha0_bias=('alpha0_signed_error','mean'),origin_nearest_distance_median=('origin_nearest_distance_dev','median')).reset_index()
 agg.to_csv(run/'LEGACY_ALPHA_METHOD_CELL_SUMMARY.csv',index=False)
 # matched-radius BBLR vs LLR at BBLR r
 mr=[]
 for keys,g in agg.groupby(['suite','cell_id'],sort=False):
  b=g[g.method.eq('BBLR')];m=g[g.method.eq('LLR_MATCHED_R')]
  if len(b)==1 and len(m)==1:
   b=b.iloc[0];m=m.iloc[0]
   mr.append({'suite':keys[0],'cell_id':keys[1],'label':b.label,'block':b.block,'dgp':b.dgp,'design':b.design,'origin_support_class':b.origin_support_class,
     'tangent_alpha_ratio_BBLR_over_matchedLLR':safe_ratio(b.test_tangent_alpha_rmse_median,m.test_tangent_alpha_rmse_median),
     'alpha0_ratio_BBLR_over_matchedLLR':safe_ratio(b.alpha0_abs_error_median,m.alpha0_abs_error_median),
     'grid_supported_tangent_alpha_ratio':safe_ratio(b.grid_tangent_alpha_rmse_supported_median,m.grid_tangent_alpha_rmse_supported_median)})
 pd.DataFrame(mr).to_csv(run/'MATCHED_RADIUS_ALPHA_SUMMARY.csv',index=False)
 # BBLR pairwise by cell against every method, based on medians.
 pw=[]
 for keys,g in agg.groupby(['suite','cell_id'],sort=False):
  b=g[g.method.eq('BBLR')]
  if len(b)!=1:continue
  b=b.iloc[0]
  for _,m in g.iterrows():
   if m.method=='BBLR':continue
   pw.append({'suite':keys[0],'cell_id':keys[1],'label':b.label,'block':b.block,'competitor':m.method,'origin_support_class':b.origin_support_class,
    'tangent_alpha_ratio_BBLR_over_comp':safe_ratio(b.test_tangent_alpha_rmse_median,m.test_tangent_alpha_rmse_median),
    'alpha0_ratio_BBLR_over_comp':safe_ratio(b.alpha0_abs_error_median,m.alpha0_abs_error_median)})
 pd.DataFrame(pw).to_csv(run/'BBLR_ALPHA_PAIRWISE_CELL.csv',index=False)
 # track summaries
 track=[]
 p=pd.DataFrame(pw)
 for (suite,comp),g in p.groupby(['suite','competitor']):
  track.append({'suite':suite,'competitor':comp,'cells':len(g),'BBLR_better_tangent_alpha_cells':int((g.tangent_alpha_ratio_BBLR_over_comp<1).sum()),'median_tangent_alpha_ratio':float(g.tangent_alpha_ratio_BBLR_over_comp.median()),
   'BBLR_better_alpha0_cells_all':int((g.alpha0_ratio_BBLR_over_comp<1).sum()),'median_alpha0_ratio_all':float(g.alpha0_ratio_BBLR_over_comp.median())})
 pd.DataFrame(track).to_csv(run/'LEGACY_TRACK_ALPHA_SUMMARY.csv',index=False)
 audit={'rows':int(len(df)),'ok_rows':int(len(ok)),'failed_rows':int(len(fail)),'replay_max_abs_error':maxerr,'replay_tolerance':1e-8,'status':'PASS' if len(fail)==0 and maxerr<=1e-8 else 'CHECK'}
 (run/'ANALYSIS_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
 # summary md
 mrdf=pd.DataFrame(mr); lines=['# Legacy coefficient-field replay summary','',f'- Rows: {len(df):,}',f'- Failures: {len(fail):,}',f'- Maximum archived-risk replay discrepancy: {maxerr:.3g}',f'- Audit: **{audit["status"]}**','']
 if len(mrdf):
  lines+=['## Matched-radius Stage-2 alpha','']
  for suite,g in mrdf.groupby('suite'):
   lines.append(f'- {suite}: median BBLR/matched-LLR tangent-alpha ratio = {g.tangent_alpha_ratio_BBLR_over_matchedLLR.median():.3f}; BBLR lower in {(g.tangent_alpha_ratio_BBLR_over_matchedLLR<1).sum()}/{len(g)} cells.')
 (run/'AUTOMATED_SUMMARY.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
