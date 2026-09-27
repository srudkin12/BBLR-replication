from __future__ import annotations
from pathlib import Path
import zlib
import numpy as np
import pandas as pd
from .config import NOISE_GRID,BASE_SEED

def _qs(s):
    a=np.asarray(s,float); a=a[np.isfinite(a)]
    mean=float(np.mean(a)); sd=float(np.std(a,ddof=1)); mcse=float(sd/np.sqrt(len(a)))
    return dict(mean=mean,sd=sd,mcse_mean=mcse,mean_mc_ci_low=float(mean-1.96*mcse),mean_mc_ci_high=float(mean+1.96*mcse),
                q025=float(np.quantile(a,.025)),q10=float(np.quantile(a,.10)),q25=float(np.quantile(a,.25)),median=float(np.quantile(a,.50)),
                q75=float(np.quantile(a,.75)),q90=float(np.quantile(a,.90)),q975=float(np.quantile(a,.975)),n=int(len(a)))

def bootstrap_median_ci(a,seed,n_boot=2000):
    a=np.asarray(a,float); a=a[np.isfinite(a)]
    rng=np.random.default_rng(seed); n=len(a)
    vals=np.empty(n_boot)
    for b in range(n_boot): vals[b]=np.median(a[rng.integers(0,n,n)])
    return float(np.quantile(vals,.025)),float(np.quantile(vals,.975))

def assemble(outdir:str):
    out=Path(outdir); files=sorted((out/"raw_batches").glob("sigma_*/*.csv"))
    if not files: raise RuntimeError("No raw batch files found")
    raw=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    raw=raw.sort_values(["sigma","rep","method"]).reset_index(drop=True)
    (out/"tables").mkdir(parents=True,exist_ok=True)
    raw.to_csv(out/"tables"/"noise_replication_level.csv",index=False)
    summ=[]
    for (sig,meth),g in raw.groupby(["sigma","method"]):
        for metric in ["outcome_rmse","deriv_rmse_test","deriv_rmse_grid"]:
            d=_qs(g[metric]); d.update(sigma=sig,method=meth,metric=metric)
            ci=bootstrap_median_ci(g[metric],int(BASE_SEED+10000*sig+zlib.crc32(meth.encode())%10000))
            d["median_mc_ci_low"],d["median_mc_ci_high"]=ci
            summ.append(d)
    method_summary=pd.DataFrame(summ)
    method_summary.to_csv(out/"tables"/"noise_method_distribution_summary.csv",index=False)
    ratios=[]
    for sig,g in raw.groupby("sigma"):
        p=g.pivot(index="rep",columns="method",values=["deriv_rmse_test","deriv_rmse_grid"])
        for metric in ["deriv_rmse_test","deriv_rmse_grid"]:
            for comp in [c for c in p[metric].columns if c!="BBLR"]:
                a=(p[(metric,"BBLR")]/p[(metric,comp)]).dropna()
                d=_qs(a); d.update(sigma=sig,metric=metric,comparator=comp,ratio_definition="BBLR_RMSE/comparator_RMSE",bblr_win_share=float(np.mean(a<1)))
                lo,hi=bootstrap_median_ci(a,int(BASE_SEED+20000*sig+zlib.crc32((comp+metric).encode())%10000)); d["median_mc_ci_low"]=lo; d["median_mc_ci_high"]=hi
                ratios.append(d)
    ratio=pd.DataFrame(ratios); ratio.to_csv(out/"tables"/"noise_paired_ratio_distribution_summary.csv",index=False)
    # BBLR tuning distribution
    b=raw[raw.method=="BBLR"].copy()
    tune=(b.groupby("sigma").agg(stage2_positive_share=("stage2_active","mean"),median_r=("param1","median"),median_eps=("param2","median"),n=("rep","nunique")).reset_index())
    tune.to_csv(out/"tables"/"noise_bblr_tuning_summary.csv",index=False)
    # anchor continuity diagnostics against manuscript-reported median-RMSE values (selected entries only)
    anchors=[
      (0.10,"BBLR","deriv_rmse_test",0.2356),(0.10,"SPLINE","deriv_rmse_test",0.1210),
      (0.25,"BBLR","deriv_rmse_test",0.4128),(0.25,"SPLINE","deriv_rmse_test",0.2338),
      (0.50,"BBLR","deriv_rmse_test",0.4726),(0.50,"AKLP2","deriv_rmse_test",0.3923),
      (0.10,"BBLR","deriv_rmse_grid",0.2364),(0.10,"SPLINE","deriv_rmse_grid",0.1223),
      (0.25,"BBLR","deriv_rmse_grid",0.4134),(0.25,"SPLINE","deriv_rmse_grid",0.2348),
      (0.50,"BBLR","deriv_rmse_grid",0.4755),(0.50,"AKLP2","deriv_rmse_grid",0.3944),
    ]
    rows=[]
    for sig,meth,metric,legacy in anchors:
        s=method_summary.query("sigma==@sig and method==@meth and metric==@metric")
        if len(s):
            new=float(s.iloc[0]["median"]); rel=(new-legacy)/legacy
            rows.append(dict(sigma=sig,method=meth,metric=metric,legacy_value=legacy,new_median=new,relative_difference=rel,within_10pct=abs(rel)<=.10))
    pd.DataFrame(rows).to_csv(out/"tables"/"anchor_continuity_check.csv",index=False)
    return raw,method_summary,ratio,tune
