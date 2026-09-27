#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, pandas as pd
from bblr_uncertainty.scan import scan_root
from bblr_uncertainty.studies import load_expected

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--out",default="outputs/archive_inventory.csv"); args=ap.parse_args()
    expected=load_expected(Path(__file__).resolve().parents[1]/"config/expected_studies.csv")
    df=scan_root(args.root,expected)
    p=Path(args.out); p.parent.mkdir(parents=True,exist_ok=True); df.to_csv(p,index=False)
    print(f"SCANNED {len(df)} data members/files; replication candidates={(df.kind=='replication_candidate').sum()}")
    print(p)
if __name__=="__main__": main()
