#!/usr/bin/env python3
"""Create a compact handback for BBLR power/landmark analysis."""
from __future__ import annotations
import argparse,csv,hashlib,json
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

CANDIDATES=[
 'RUN_CONFIG.json','COMPLETION_AUDIT.json','LANDMARK_CELLS.csv','EXPECTED_CHUNKS.csv','CHUNK_MANIFEST.csv',
 'HPC_SELECTED_BY_Y.csv.gz','HPC_SELECTED_METHOD_SUMMARY.csv','HPC_SELECTED_METHOD_ROBUST_SUMMARY.csv',
 'POWER_SELECTED_RULES.csv.gz','POWER_LANDMARK_SUMMARY.csv','POWER_BBLR_STAGE2_SELECTED.csv.gz',
 'POWER_SUPPORT_AWARE_PAIRWISE.csv.gz','POWER_SUPPORT_AWARE_PAIRWISE_SUMMARY.csv',
]

def sha(path,bs=1024*1024):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(bs),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-dir',required=True,type=Path)
    ap.add_argument('--extra-analysis-dir',type=Path,default=None)
    ap.add_argument('--output-dir',type=Path,default=None); ap.add_argument('--skip-shard-hashes',action='store_true')
    a=ap.parse_args(); run=a.run_dir.resolve(); audit=json.loads((run/'COMPLETION_AUDIT.json').read_text())
    if audit.get('status')!='PASS': raise SystemExit('run completion audit is not PASS')
    files=[run/f for f in CANDIDATES if (run/f).exists()]
    if a.extra_analysis_dir:
        extra=a.extra_analysis_dir.resolve()
        files += [p for p in sorted(extra.iterdir()) if p.is_file()]
    shards=sorted((run/'shards').glob('*.csv.gz'))
    if len(shards)!=int(audit['expected_chunks']): raise SystemExit('shard count mismatch')
    out=(a.output_dir or run.parent/'handbacks').resolve(); out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S'); base=f'{run.name}_power_handback_{stamp}'
    zp=out/f'{base}.zip'; sp=out/f'{base}.zip.sha256'
    src=[]
    for p in files: src.append({'filename':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
    sh=[]
    for i,p in enumerate(shards,1):
        sh.append({'filename':p.name,'bytes':p.stat().st_size,'sha256':'' if a.skip_shard_hashes else sha(p)})
        if not a.skip_shard_hashes and (i%100==0 or i==len(shards)): print(f'Hashed {i}/{len(shards)} shards')
    import io
    b1=io.StringIO(); w=csv.DictWriter(b1,fieldnames=['filename','bytes','sha256']); w.writeheader(); w.writerows(src)
    b2=io.StringIO(); w=csv.DictWriter(b2,fieldnames=['filename','bytes','sha256']); w.writeheader(); w.writerows(sh)
    manifest={'schema':'BBLR_POWER_HANDBACK_V1','run_name':run.name,'status':'PASS','shards':len(shards),
              'raw_shards_included':False,'per_shard_hashes_included':not a.skip_shard_hashes,
              'extra_analysis_dir':str(a.extra_analysis_dir.resolve()) if a.extra_analysis_dir else None}
    with ZipFile(zp,'w',compression=ZIP_DEFLATED,compresslevel=6) as z:
        seen=set()
        for p in files:
            arc=p.name if p.name not in seen else f'extra_{p.name}'; seen.add(p.name); z.write(p,arcname=arc)
        z.writestr('SOURCE_FILE_SHA256.csv',b1.getvalue()); z.writestr('SHARD_SHA256.csv',b2.getvalue())
        z.writestr('HANDOFF_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    digest=sha(zp); sp.write_text(f'{digest}  {zp.name}\n')
    print(json.dumps({'status':'PASS','zip':str(zp),'sha256_file':str(sp),'zip_mb':round(zp.stat().st_size/2**20,3)},indent=2))

if __name__=='__main__': main()
