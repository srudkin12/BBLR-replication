#!/usr/bin/env python3
import argparse,hashlib,zipfile
from pathlib import Path
ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True); ap.add_argument('--out-dir',default='.'); args=ap.parse_args()
rd=Path(args.run_dir); od=Path(args.out_dir); name='BBLR_LocalSparseComparator_Audit_v1_0_HANDback.zip'; zp=od/name
files=['RUN_CONFIG.json','COMPLETION_AUDIT.json','AUTOMATED_SUMMARY.md','METHOD_CELL_SUMMARY.csv','BBLR_PAIRWISE.csv','LOCAL_SPARSE_FRONTIER.csv','LOCAL_SPARSE_TUNING_SUMMARY.csv','BBLR_VS_LOCAL_SPARSE.csv']
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED) as z:
    for fn in files:
        p=rd/fn
        if p.exists(): z.write(p,arcname=fn)
h=hashlib.sha256(zp.read_bytes()).hexdigest(); (od/(name+'.sha256')).write_text(f'{h}  {name}\n'); print(zp); print(h)
