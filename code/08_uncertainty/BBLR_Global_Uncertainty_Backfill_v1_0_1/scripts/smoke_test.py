#!/usr/bin/env python3
import tempfile
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from bblr_uncertainty.normalize import normalize_long
from bblr_uncertainty.stats import summarize_long, paired_bblr


def main():
    rng=np.random.default_rng(1)
    rows=[]
    for suite in ['UV92','BIV101']:
        for rep in range(100):
            for m,shift in [('BBLR',0),('LLR',0.1)]:
                rows.append([suite,0,rep,m,abs(rng.normal(.5+shift,.1))])
    df=pd.DataFrame(rows,columns=['suite','cell_id','rep','method','test_slope_rmse'])
    x,meta=normalize_long(df,'legacy_coefficient_replay','synthetic.csv')
    assert meta['status']=='ready' and x.cell_id.nunique()==2
    sm=summarize_long(x); pr=paired_bblr(x)
    assert len(pr)==2 and set(sm.n.dropna().astype(int))=={100}

    # Wide matched-H-lambda methods must become pairable estimators.
    rows=[]
    for rep in range(100):
        rows.append([0,rep,1+0.01*rep,1.1+0.01*rep,1.2+0.01*rep,1.05+0.01*rep])
    w=pd.DataFrame(rows,columns=['cell_id','rep','bblr_test_mse','wsafe_test_mse','wstrict_test_mse','llr_rB_test_mse'])
    x2,meta2=normalize_long(w,'matched_hlambda','synthetic_wide.csv')
    assert meta2['status']=='ready'
    assert {'BBLR','WSAFE','WSTRICT','LLR'}.issubset(set(x2.method))
    pr2=paired_bblr(x2)
    assert {'WSAFE','WSTRICT','LLR'}.issubset(set(pr2.comparator))
    assert {'q025','q975','median_ci_low','median_ci_high'}.issubset(sm.columns)
    print('PASS_GLOBAL_UNCERTAINTY_SMOKE_V1_0_1')
    print(f'legacy_cells={x.cell_id.nunique()} matched_comparators={sorted(pr2.comparator.unique())}')
if __name__=='__main__': main()
