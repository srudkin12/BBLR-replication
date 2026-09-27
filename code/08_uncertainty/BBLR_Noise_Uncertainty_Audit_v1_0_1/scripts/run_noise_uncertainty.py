#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys,os,argparse,json,time,platform,zipfile,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
# Set process-wide non-GUI/non-nested-thread environment before worker imports.
os.environ.setdefault("MPLBACKEND","Agg")
for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]: os.environ[k]="1"
import numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from bblr_noise_audit.config import NOISE_GRID,BASE_SEED
from bblr_noise_audit.simulate import run_batch,batch_path
from bblr_noise_audit.summarize import assemble
from bblr_noise_audit.plots import make_plots
from bblr_noise_audit.legacy import inventory

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=80)
    ap.add_argument('--reps',type=int,default=1000)
    ap.add_argument('--batch-size',type=int,default=5)
    ap.add_argument('--seed',type=int,default=BASE_SEED)
    ap.add_argument('--output',default='outputs')
    ap.add_argument('--noise',nargs='*',type=float,default=None,help='Optional subset; default all nine frozen levels')
    ap.add_argument('--legacy-root',default='..')
    args=ap.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    sigmas=np.array(args.noise if args.noise else NOISE_GRID,dtype=float)
    if args.workers<1 or args.workers>80: raise SystemExit('--workers must be 1..80 for this production package')
    cfg=dict(workers=args.workers,reps=args.reps,batch_size=args.batch_size,seed=args.seed,noise=sigmas.tolist(),python=platform.python_version(),started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    (out/'RUN_CONFIG.json').write_text(json.dumps(cfg,indent=2))
    tasks=[]
    for s in sigmas:
        for st in range(0,args.reps,args.batch_size):
            en=min(args.reps,st+args.batch_size)
            p=batch_path(out,float(s),st,en)
            if not p.exists(): tasks.append((float(s),st,en))
    print(f"NOISE_UNCERTAINTY_PRODUCTION levels={len(sigmas)} reps={args.reps} methods=5 cv=5x5 workers={args.workers} pending_batches={len(tasks)}",flush=True)
    done=0
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs={ex.submit(run_batch,str(out),s,st,en,args.seed):(s,st,en) for s,st,en in tasks}
            for fut in as_completed(futs):
                fut.result(); done+=1
                if done%20==0 or done==len(tasks): print(f"completed_batches={done}/{len(tasks)}",flush=True)
    raw,ms,ratio,tune=assemble(str(out)); make_plots(str(out),ms,ratio,tune)
    try: inventory(args.legacy_root,str(out/'audits'/'legacy_archive_inventory.csv'))
    except Exception as e: (out/'LEGACY_INVENTORY_WARNING.txt').write_text(str(e))
    # completion checks
    counts=raw.groupby(['sigma','method']).rep.nunique().reset_index(name='n_reps')
    counts.to_csv(out/'tables'/'completion_counts.csv',index=False)
    expected=len(sigmas)*5
    ok=(len(counts)==expected and counts.n_reps.min()==args.reps and counts.n_reps.max()==args.reps)
    audit={'complete':bool(ok),'rows':int(len(raw)),'expected_method_cells':expected,'min_reps':int(counts.n_reps.min()),'max_reps':int(counts.n_reps.max())}
    (out/'COMPLETION_AUDIT.json').write_text(json.dumps(audit,indent=2))
    # compact handback: summaries + figures + config/audits, exclude raw batch shards; include replication-level CSV (45k rows at production)
    hand=Path('BBLR_Noise_Uncertainty_Audit_v1_0_1_HANDback.zip')
    with zipfile.ZipFile(hand,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in [out/'RUN_CONFIG.json',out/'COMPLETION_AUDIT.json']:
            if p.exists(): z.write(p,p.relative_to(out.parent))
        for sub in ['tables','figures','audits']:
            d=out/sub
            if d.exists():
                for p in d.rglob('*'):
                    if p.is_file(): z.write(p,p.relative_to(out.parent))
    digest=sha256(hand); Path(str(hand)+'.sha256').write_text(f"{digest}  {hand.name}\n")
    print(json.dumps(audit,indent=2)); print(f"HANDback={hand}"); print(f"SHA256={digest}")
    if not ok: raise SystemExit('COMPLETION_AUDIT_FAILED')
if __name__=='__main__': main()
