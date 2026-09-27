#!/usr/bin/env python3
import argparse,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);a=ap.parse_args();run=a.run_dir
for track in ('uv92','uv27','biv101'):
 p=run/f'RUN_CONFIG_{track}.json'
 if not p.exists():continue
 cfg=json.loads(p.read_text()); present=len(list((run/'shards').glob(f'{track}_chunk_*.csv.gz')))
 print(f'{track}: {present}/{cfg["expected_chunks"]} chunks ({100*present/max(cfg["expected_chunks"],1):.1f}%)')
