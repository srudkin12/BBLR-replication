#!/usr/bin/env python3
import argparse,hashlib,zipfile
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--out-dir',type=Path,default=Path('.'));a=ap.parse_args();files=['COMPLETION_AUDIT.json','ANALYSIS_AUDIT.json','RUN_CONFIG.json','FROZEN_SELECTED_CONFIGS_USED.csv.gz','STABLE_ARITHMETIC_REPLICATION_METRICS.csv.gz','CELL_STABLE_ARITHMETIC_SUMMARY.csv','THEORY_STABLE_ARITHMETIC_SUMMARY.csv','AUTOMATED_SUMMARY.md'];out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=True);z=out/'BBLR_MatchedHLambda_StableArithmetic_v1_1_HANDback.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as zz:
    for f in files:
        p=a.run_dir/f
        if p.exists():zz.write(p,arcname=f)
h=hashlib.sha256(z.read_bytes()).hexdigest();s=out/(z.name+'.sha256');s.write_text(f'{h}  {z.name}\n');print(z);print(s);print(h)
