from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.stats import binom
from .config import QUANTILES


def median_order_ci(x, alpha=0.05):
    x=np.sort(np.asarray(x,float)); x=x[np.isfinite(x)]; n=len(x)
    if n==0: return (np.nan,np.nan)
    # Distribution-free equal-tail order-statistic interval for population median.
    lo=int(max(0, math.floor(binom.ppf(alpha/2,n,0.5))))
    hi=int(min(n-1, math.ceil(binom.ppf(1-alpha/2,n,0.5))))
    return float(x[lo]), float(x[hi])


def summarize_vector(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]; n=len(x)
    if n==0:
        return {k:np.nan for k in ["n","mean","sd","mcse","mean_ci_low","mean_ci_high","q025","q10","q25","median","q75","q90","q975","median_ci_low","median_ci_high"]}
    q=np.quantile(x,QUANTILES)
    mean=float(np.mean(x)); sd=float(np.std(x,ddof=1)) if n>1 else 0.0; mcse=sd/math.sqrt(n) if n else np.nan
    ml,mh=mean-1.96*mcse,mean+1.96*mcse
    medlo,medhi=median_order_ci(x)
    return {"n":n,"mean":mean,"sd":sd,"mcse":mcse,"mean_ci_low":ml,"mean_ci_high":mh,
            "q025":q[0],"q10":q[1],"q25":q[2],"median":q[3],"q75":q[4],"q90":q[5],"q975":q[6],
            "median_ci_low":medlo,"median_ci_high":medhi}


def summarize_long(x: pd.DataFrame):
    idcols={"study_id","source","cell_id","rep","method"}
    metrics=[c for c in x.columns if c not in idcols and pd.api.types.is_numeric_dtype(x[c])]
    rows=[]
    for (study,cell,method),g in x.groupby(["study_id","cell_id","method"],dropna=False,sort=False):
        for metric in metrics:
            s=summarize_vector(g[metric].to_numpy())
            rows.append({"study_id":study,"cell_id":cell,"method":method,"metric":metric,**s})
    return pd.DataFrame(rows)


def paired_bblr(x: pd.DataFrame):
    idcols={"study_id","source","cell_id","rep","method"}
    metrics=[c for c in x.columns if c not in idcols and pd.api.types.is_numeric_dtype(x[c])]
    out=[]
    for (study,cell),g in x.groupby(["study_id","cell_id"],dropna=False,sort=False):
        methods=set(g.method.dropna().astype(str))
        if "BBLR" not in methods: continue
        for metric in metrics:
            p=g.pivot_table(index="rep",columns="method",values=metric,aggfunc="first")
            if "BBLR" not in p: continue
            for comp in sorted(c for c in p.columns if c!="BBLR"):
                z=p[["BBLR",comp]].dropna()
                if len(z)<20: continue
                b=z["BBLR"].to_numpy(float); c=z[comp].to_numpy(float)
                diff=b-c
                ratio=np.divide(b,c,out=np.full_like(b,np.nan),where=np.isfinite(c)&(c>0)&np.isfinite(b)&(b>=0))
                ds=summarize_vector(diff); rs=summarize_vector(ratio)
                mean_class="unclear"
                if ds["mean_ci_high"]<0: mean_class="BBLR_better"
                elif ds["mean_ci_low"]>0: mean_class="comparator_better"
                median_class="unclear"
                if rs["median_ci_high"]<1: median_class="BBLR_better"
                elif rs["median_ci_low"]>1: median_class="comparator_better"
                out.append({
                    "study_id":study,"cell_id":cell,"metric":metric,"comparator":comp,"paired_n":len(z),
                    "bblr_win_share":float(np.mean(b<c)),
                    **{f"diff_{k}":v for k,v in ds.items()},
                    **{f"ratio_{k}":v for k,v in rs.items()},
                    "mean_difference_class":mean_class,"median_ratio_class":median_class,
                })
    return pd.DataFrame(out)
