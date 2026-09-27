from __future__ import annotations
import os
os.environ.setdefault("MPLBACKEND","Agg")
import matplotlib
matplotlib.use("Agg",force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def make_plots(outdir:str,method_summary:pd.DataFrame,ratio:pd.DataFrame,tune:pd.DataFrame):
    out=Path(outdir); figdir=out/"figures"; figdir.mkdir(parents=True,exist_ok=True)
    # main paired-ratio plot: three panels; central 95% replication interval and IQR expose between-replication dispersion.
    comps=["LLR","AKLP2","SPLINE"]
    fig,axs=plt.subplots(1,3,figsize=(12.0,3.5),sharey=True)
    for ax,comp in zip(axs,comps):
        d=ratio[(ratio.metric=="deriv_rmse_test")&(ratio.comparator==comp)].sort_values("sigma")
        x=d.sigma.to_numpy(); med=d["median"].to_numpy(); q025=d.q025.to_numpy(); q975=d.q975.to_numpy(); q25=d.q25.to_numpy(); q75=d.q75.to_numpy()
        ax.fill_between(x,q025,q975,alpha=.12); ax.fill_between(x,q25,q75,alpha=.25); ax.plot(x,med,marker="o",lw=1.5)
        ax.axhline(1.0,lw=1,ls="--")
        ax.set_xlabel("Noise standard deviation")
        ax.set_title(comp)
    axs[0].set_ylabel("BBLR / comparator derivative RMSE")
    fig.tight_layout(); fig.savefig(figdir/"fig_noise_paired_ratio_uncertainty_main.pdf",bbox_inches="tight"); fig.savefig(figdir/"fig_noise_paired_ratio_uncertainty_main.png",dpi=220,bbox_inches="tight"); plt.close(fig)
    # Absolute derivative RMSE distributions.
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    for meth in ["BBLR","LLR","AKLP2","SPLINE"]:
        d=method_summary[(method_summary.metric=="deriv_rmse_test")&(method_summary.method==meth)].sort_values("sigma")
        x=d.sigma.to_numpy(); med=d["median"].to_numpy()
        ax.plot(x,med,marker="o",lw=1.3,label=meth)
    ax.set_xlabel("Noise standard deviation"); ax.set_ylabel("Median derivative RMSE"); ax.legend(frameon=False,ncol=2)
    fig.tight_layout(); fig.savefig(figdir/"fig_noise_absolute_derivative_rmse.pdf",bbox_inches="tight"); fig.savefig(figdir/"fig_noise_absolute_derivative_rmse.png",dpi=220,bbox_inches="tight"); plt.close(fig)
    # Stage2 selection share.
    fig,ax=plt.subplots(figsize=(6.2,3.8)); ax.plot(tune.sigma,tune.stage2_positive_share,marker="o"); ax.set_ylim(0,1); ax.set_xlabel("Noise standard deviation"); ax.set_ylabel("Share selecting positive Stage 2"); fig.tight_layout(); fig.savefig(figdir/"fig_noise_stage2_selection.pdf",bbox_inches="tight"); fig.savefig(figdir/"fig_noise_stage2_selection.png",dpi=220,bbox_inches="tight"); plt.close(fig)
