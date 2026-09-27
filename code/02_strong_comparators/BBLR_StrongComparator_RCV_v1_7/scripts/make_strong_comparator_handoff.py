#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

FILES=['RUN_CONFIG.json','COMPLETION_AUDIT.json','EXPECTED_CHUNKS.csv','CHUNK_MANIFEST.csv','STRONG_COMPARATOR_CELLS.csv',
       'SC_CELL_METHOD_SUMMARY.csv','SC_BBLR_PAIRWISE_CELL_SUMMARY.csv','SC_PARETO_CELL_SUMMARY.csv',
       'SC_BLOCK_PAIRWISE_SUMMARY.csv','SC_CELL_BEST_METHODS.csv','SC_SELECTED_ALL.csv.gz']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True,type=Path); args=ap.parse_args()
    run=args.run_dir.resolve(); audit=json.loads((run/'COMPLETION_AUDIT.json').read_text())
    if audit.get('status')!='PASS': raise SystemExit('refusing handoff: completion audit not PASS')
    missing=[f for f in FILES if not (run/f).exists()]
    if missing: raise SystemExit('missing handoff files: '+', '.join(missing))
    out=run.parent/'handbacks'; out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S'); base=f'{run.name}_strong_comparator_handback_{stamp}'
    zp=out/f'{base}.zip'; sp=out/f'{base}.zip.sha256'
    manifest=[]
    with ZipFile(zp,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for f in FILES:
            p=run/f; z.write(p,arcname=f); manifest.append({'filename':f,'bytes':p.stat().st_size,'sha256':sha(p)})
        z.writestr('HANDOFF_MANIFEST.json',json.dumps({'schema':'BBLR_STRONG_COMPARATOR_HANDOFF_V1','files':manifest},indent=2)+'\n')
    digest=sha(zp); sp.write_text(f'{digest}  {zp.name}\n')
    print(json.dumps({'status':'PASS','zip':str(zp),'sha256_file':str(sp),'zip_mb':round(zp.stat().st_size/1024/1024,3)},indent=2))
if __name__=='__main__': main()
