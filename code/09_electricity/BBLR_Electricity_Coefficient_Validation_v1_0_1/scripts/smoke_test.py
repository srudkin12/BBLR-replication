#!/usr/bin/env python3
from __future__ import annotations
import sys, tempfile
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_coefficient_validation as a

rng=np.random.default_rng(123)
n=120
x=np.sort(rng.uniform(12,34,n)); y=4.0+0.015*(x-22)**2+rng.normal(0,.25,n)
groups=np.array([f'W{j//5:02d}' for j in range(n)])
params={'BBLR':(2.5,2.0),'LLR':2.5,'LP2':6.0,'PSpline':(5,0.1),'Hinge':(20.0,28.0)}
lo,hi=np.quantile(x,[.05,.95])
with tempfile.TemporaryDirectory() as td:
    od=Path(td)
    splits=a.random_splits(n,2,321)
    df,rep,summ=a.run_crossfit(od,'smoke','random',splits,x,y,params,lo,hi,2)
    assert len(df)>0 and len(rep)>0 and len(summ)>0
    grid,curves=a.agreement(od,x,y,params,lo,hi)
    assert all(len(v)==len(grid) for v in curves.values())
    bp,bf=a.run_bootstrap(od,'smoke',groups,x,y,params,lo,hi,5,2)
    assert len(bp)>0 and len(bf)==5
    a.make_fig(od,grid,curves,bp,'Smoke')
    assert (od/'fig_all_method_derivatives.png').is_file()
print('PASS_COEFFICIENT_VALIDATION_SMOKE')
