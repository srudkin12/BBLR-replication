#!/usr/bin/env python3
import argparse,hashlib,zipfile
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--out-dir',type=Path,default=Path('.'));a=ap.parse_args();run=a.run_dir.resolve();out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
files=['ANALYSIS_AUDIT.json','AUTOMATED_SUMMARY.md','REPLAY_INTEGRITY_METRICS.csv','LEGACY_ALPHA_METHOD_CELL_SUMMARY.csv','MATCHED_RADIUS_ALPHA_SUMMARY.csv','BBLR_ALPHA_PAIRWISE_CELL.csv','LEGACY_TRACK_ALPHA_SUMMARY.csv','REFERENCE_ALPHA_SUPPORT_SUMMARY.csv','LEGACY_COEFFICIENT_FIELD_REPLICATIONS.csv.gz']
files += [p.name for p in run.glob('COMPLETION_AUDIT_*.json')]
name='BBLR_LegacyCoefficientFieldReplay_v1_0_HANDback.zip';zp=out/name
with zipfile.ZipFile(zp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in files:
  p=run/f
  if p.exists():z.write(p,arcname=p.name)
h=hashlib.sha256(zp.read_bytes()).hexdigest();(out/(name+'.sha256')).write_text(f'{h}  {name}\n');print(zp);print(h)
