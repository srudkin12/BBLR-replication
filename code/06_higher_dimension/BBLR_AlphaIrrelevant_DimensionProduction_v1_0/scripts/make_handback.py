#!/usr/bin/env python3
import argparse,hashlib,zipfile
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); ap.add_argument('--out-dir',default='.'); args=ap.parse_args()
rd=Path(args.run_dir); od=Path(args.out_dir)
files=[
 'RUN_CONFIG.json','COMPLETION_AUDIT.json','MATCHED_RADIUS_IDENTITY_AUDIT.json','AUTOMATED_SUMMARY.md','METHOD_CELL_SUMMARY.csv','BBLR_PAIRWISE.csv',
 'MATCHED_RADIUS_STAGE2_SUMMARY.csv','MATCHED_RADIUS_TRACK_SUMMARY.csv','ALPHA0_TANGENT_SUMMARY.csv','NUISANCE_SUMMARY.csv','NULL_SLOPE_SUMMARY.csv',
 'DIMENSION_VECTOR_SUMMARY.csv','AMBIENT_COST_SUMMARY.csv','SAMPLE_SIZE_SUMMARY.csv','BBLR_TUNING_SUMMARY.csv'
]
name='BBLR_AlphaIrrelevant_DimensionProduction_v1_0_HANDback.zip'; zp=od/name
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
    for fn in files:
        p=rd/fn
        if p.exists(): z.write(p,arcname=fn)
h=hashlib.sha256(zp.read_bytes()).hexdigest(); (od/(name+'.sha256')).write_text(f'{h}  {name}\n'); print(zp); print(h)
