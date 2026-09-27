#!/usr/bin/env python3
from pathlib import Path
import ast, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
req=[ROOT/'source_data/peak_workdays.csv',ROOT/'source_data/evening1800_workdays.csv',ROOT/'frozen_grouped_selection.csv']
for p in req:
    if not p.exists(): raise SystemExit(f'MISSING {p}')
sel=pd.read_csv(ROOT/'frozen_grouped_selection.csv')
assert len(sel)==10 and set(sel.outcome)=={'peak','evening1800'}
for s in sel.param: ast.literal_eval(str(s))
for name in ['peak','evening1800']:
    d=pd.read_csv(ROOT/f'source_data/{name}_workdays.csv')
    assert len(d)==251
    assert {'Temperature','week_group','y'}.issubset(d.columns)
print('PASS_GROUPED_PRIMARY_PREFLIGHT')
