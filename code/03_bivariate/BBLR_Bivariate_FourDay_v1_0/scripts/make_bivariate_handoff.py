#!/usr/bin/env python3
import argparse,hashlib,zipfile,json
from pathlib import Path
from datetime import datetime
CORE=['RUN_CONFIG.json','COMPLETION_AUDIT.json','BIVARIATE_CELLS.csv','BIV_METHOD_CELL_SUMMARY.csv','BIV_CELL_BEST_METHODS.csv','BIV_BBLR_PAIRWISE.csv','BIV_SELECTED_ALL.csv.gz']
def sha(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True,type=Path); a=ap.parse_args(); run=a.run_dir.resolve()
    miss=[x for x in CORE if not (run/x).exists()]
    if miss: raise SystemExit('missing '+str(miss))
    out=run.parent/'handbacks'; out.mkdir(exist_ok=True); stamp=datetime.now().strftime('%Y%m%d_%H%M%S'); base=f'{run.name}_bivariate_handback_{stamp}'; zp=out/f'{base}.zip'; sp=out/f'{base}.zip.sha256'
    with zipfile.ZipFile(zp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for x in CORE: z.write(run/x,arcname=x)
    dig=sha(zp); sp.write_text(f'{dig}  {zp.name}\n'); print(json.dumps({'zip':str(zp),'sha':str(sp),'mb':zp.stat().st_size/1024/1024},indent=2))
if __name__=='__main__':main()
