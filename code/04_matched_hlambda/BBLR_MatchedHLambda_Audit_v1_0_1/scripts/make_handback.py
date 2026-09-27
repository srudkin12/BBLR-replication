#!/usr/bin/env python3
import argparse, hashlib, json, zipfile
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',type=Path,required=True); ap.add_argument('--out-dir',type=Path,default=Path('.')); a=ap.parse_args()
r=a.run_dir.resolve(); out=a.out_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
required=['COMPLETION_AUDIT.json','ANALYSIS_AUDIT.json','RUN_CONFIG.json','FROZEN_SELECTED_CONFIGS_USED.csv.gz','MATCHED_HLAMBDA_REPLICATION_METRICS.csv.gz','CELL_PAIRWISE_SUMMARY.csv','THEORY_STRATUM_SUMMARY.csv','KEY_CASE_PAIRWISE_SUMMARY.csv','AUTOMATED_SUMMARY.md']
missing=[f for f in required if not (r/f).exists()]
if missing: raise SystemExit('missing handback files: '+', '.join(missing))
name='BBLR_MatchedHLambda_Audit_v1_0_1_HANDback.zip'; zp=out/name
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for f in required: z.write(r/f,arcname=f)
h=hashlib.sha256(zp.read_bytes()).hexdigest(); sha=out/(name+'.sha256'); sha.write_text(f'{h}  {name}\n')
print(zp); print(sha); print(h)
