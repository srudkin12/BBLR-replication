#!/usr/bin/env python3
from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from bblr_noise_audit.legacy import inventory
p=argparse.ArgumentParser(); p.add_argument('--root',default='..'); p.add_argument('--output',default='outputs/audits/legacy_archive_inventory.csv'); a=p.parse_args()
df=inventory(a.root,a.output); print(f"legacy_csv_members={len(df)}"); print(a.output)
