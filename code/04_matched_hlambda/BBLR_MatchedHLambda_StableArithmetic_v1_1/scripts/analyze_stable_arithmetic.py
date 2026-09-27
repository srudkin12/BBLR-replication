#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

def boot(d,B,seed):
    d=np.asarray(d,float);d=d[np.isfinite(d)];rng=np.random.default_rng(seed);n=len(d);m=np.empty(B)
    for i in range(B):m[i]=d[rng.integers(0,n,n)].mean()
    return np.quantile(m,[.025,.975]) if n else (np.nan,np.nan)
def summ(g,wcol,B,seed):
    b=g.bblr_test_mse.to_numpy(float);w=g[wcol].to_numpy(float);ok=np.isfinite(b)&np.isfinite(w);b=b[ok];w=w[ok];d=b-w;lo,hi=boot(d,B,seed);cl='BBLR_better' if hi<0 else ('W_better' if lo>0 else 'unclear');return {'n_pairs':len(d),'mean_bblr_mse':b.mean(),'mean_w_mse':w.mean(),'median_bblr_rmse':np.median(np.sqrt(b)),'median_w_rmse':np.median(np.sqrt(w)),'bblr_win_share':np.mean(b<w),'mean_delta_bblr_minus_w':d.mean(),'bootstrap_ci_low':lo,'bootstrap_ci_high':hi,'mse_ratio_bblr_over_w':b.mean()/w.mean(),'classification':cl}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--bootstrap-reps',type=int,default=1000);a=ap.parse_args();rd=a.run_dir;files=sorted((rd/'replication_shards').glob('*.csv.gz'));df=pd.concat([pd.read_csv(f) for f in files],ignore_index=True).sort_values(['cell_id','rep']).reset_index(drop=True);df.to_csv(rd/'STABLE_ARITHMETIC_REPLICATION_METRICS.csv.gz',index=False,compression='gzip');pos=df[df.eps>0];rows=[]
    for meth,wcol in [('normal','wnormal_test_mse'),('centered','wcenter_test_mse')]:
        for cid,g in pos.groupby('cell_id'):
            meta=g.iloc[0];rows.append({'method':meth,'cell_id':int(cid),'landmark_label':meta.landmark_label,'panel_block':meta.panel_block,'dgp':meta.dgp,'design':meta.design,**summ(g,wcol,a.bootstrap_reps,20260912+int(cid)+(0 if meth=='normal' else 10000))})
    cell=pd.DataFrame(rows);cell.to_csv(rd/'CELL_STABLE_ARITHMETIC_SUMMARY.csv',index=False)
    strata=[];masks=[('all_positive',np.ones(len(pos),bool)),('lambda_eq1',pos.lambda_stratum.eq('eq1').to_numpy()),('lambda_lt1',pos.lambda_stratum.eq('lt1').to_numpy()),('lambda_gt1',pos.lambda_stratum.eq('gt1').to_numpy()),('absGamma_lt_0p01',pos.gamma_abs_lt_0p01.astype(bool).to_numpy())]
    for meth,wcol in [('normal','wnormal_test_mse'),('centered','wcenter_test_mse')]:
        for j,(lab,mask) in enumerate(masks):strata.append({'method':meth,'stratum':lab,**summ(pos.loc[mask],wcol,a.bootstrap_reps,20270000+j+(0 if meth=='normal' else 100))})
    st=pd.DataFrame(strata);st.to_csv(rd/'THEORY_STABLE_ARITHMETIC_SUMMARY.csv',index=False)
    counts=cell.groupby('method').classification.value_counts().unstack(fill_value=0);audit={'rows':len(df),'cells':int(df.cell_id.nunique()),'positive_stage2_rows':len(pos),'archive_test_rmse_max_abs_diff':float(df.archive_test_rmse_abs_diff.max()),'archive_grid_rmse_max_abs_diff':float(df.archive_grid_rmse_abs_diff.max()),'normal_counts':counts.loc['normal'].to_dict(),'centered_counts':counts.loc['centered'].to_dict(),'max_test_slope_abs_diff':float(df.normal_center_test_max_abs_slope_diff.max()),'q99_test_slope_abs_diff':float(df.normal_center_test_max_abs_slope_diff.quantile(.99)),'max_fallback_mask_disagreement_share':float(df.fallback_mask_disagreement_test_share.max()),'status':'PASS'};exp={'BBLR_better':73,'W_better':14,'unclear':5}
    
    is_full=(len(df)==92000 and df.cell_id.nunique()==92)
    basic_ok=(audit['archive_test_rmse_max_abs_diff']<1e-9 and audit['archive_grid_rmse_max_abs_diff']<1e-9 and audit['max_fallback_mask_disagreement_share']<1e-12 and audit['max_test_slope_abs_diff']<1e-6)
    count_ok=(all(int(audit['normal_counts'].get(k,0))==v for k,v in exp.items()) if is_full else True)
    if not (basic_ok and count_ok): audit['status']='CHECK'
    (rd/'ANALYSIS_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n');lines=['# Stable-arithmetic matched-H-lambda audit','',f'- Analysis audit: **{audit["status"]}**.',f'- Normal-form counts: {audit["normal_counts"]}.',f'- Centered-form counts: {audit["centered_counts"]}.',f'- Maximum per-replication max absolute test-slope difference: {audit["max_test_slope_abs_diff"]:.6g}.',f'- 99th percentile of that diagnostic: {audit["q99_test_slope_abs_diff"]:.6g}.','', '## Theory-facing aggregates']
    for _,r in st.iterrows():lines.append(f'- {r.method}, {r.stratum}: ratio={r.mse_ratio_bblr_over_w:.4f}, win={r.bblr_win_share:.3f}, delta={r.mean_delta_bblr_minus_w:.6g} [{r.bootstrap_ci_low:.6g},{r.bootstrap_ci_high:.6g}], {r.classification}.')
    (rd/'AUTOMATED_SUMMARY.md').write_text('\n'.join(lines)+'\n');print(json.dumps(audit,indent=2));print('\n'.join(lines))
if __name__=='__main__':main()
