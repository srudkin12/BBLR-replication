#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from bblr_uncertainty.stats import summarize_vector

def main():
    pkg=Path(__file__).resolve().parents[1]
    exp=pd.read_csv(pkg/'config/expected_studies.csv')
    assert len(exp)>=10
    x=[1,2,3,4,5,6,7,8,9,10]*100
    s=summarize_vector(x)
    assert s['n']==1000 and s['q025']<=s['median']<=s['q975'] and s['mean_ci_low']<s['mean_ci_high']
    print('PASS_FOR_GLOBAL_UNCERTAINTY_BACKFILL')
    print(f"expected_studies={len(exp)} q025={s['q025']:.3f} median={s['median']:.3f} q975={s['q975']:.3f}")
if __name__=='__main__': main()
