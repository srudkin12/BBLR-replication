#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True);a=ap.parse_args();e=pd.read_csv(a.run_dir/'EXPECTED_CHUNKS.csv');fs=list((a.run_dir/'replication_shards').glob('*.csv.gz'));rows=0
for f in fs:
    try:rows+=len(pd.read_csv(f,usecols=['cell_id']))
    except:pass
print(f'Chunks: {len(fs)}/{len(e)} ({100*len(fs)/len(e):.1f}%)');print(f'Rows: {rows}/{int(e.stop.max())} ({100*rows/int(e.stop.max()):.1f}%)')
