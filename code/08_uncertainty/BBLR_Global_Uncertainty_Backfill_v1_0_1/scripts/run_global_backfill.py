#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, shutil, zipfile, hashlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from bblr_uncertainty.scan import scan_root
from bblr_uncertainty.studies import load_expected
from bblr_uncertainty.io import read_source
from bblr_uncertainty.normalize import normalize_long
from bblr_uncertainty.stats import summarize_long, paired_bblr
from bblr_uncertainty.report import coverage, write_markdown


def atomic_csv(df,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp'); df.to_csv(tmp,index=False); os.replace(tmp,p)

def process_one(root,row):
    source=row.source; sid=row.study_id
    try:
        df=read_source(root,source)
        x,meta=normalize_long(df,sid,source)
        meta.update({"study_id":sid,"source":source,"input_rows":len(df),"input_columns":"|".join(map(str,df.columns))})
        return x,meta
    except Exception as e:
        return None,{"status":"review","reason":f"read_or_normalize_error:{type(e).__name__}:{e}","study_id":sid,"source":source}


def merge_identity_rows(x, sid, out):
    idc=["study_id","cell_id","rep","method"]
    value_cols=[c for c in x.columns if c not in idc+["source"]]
    rows=[]; conflicts=[]
    for key,g in x.groupby(idc,dropna=False,sort=False):
        row=dict(zip(idc,key)); src=[]; bad=False
        for source in g.get("source",pd.Series(dtype=str)).dropna().astype(str).unique(): src.append(source)
        row["source"]="|".join(src)
        for c in value_cols:
            vals=g[c].dropna()
            if vals.empty:
                row[c]=float("nan")
                continue
            if pd.api.types.is_numeric_dtype(vals):
                arr=vals.astype(float).to_numpy()
                ref=arr[0]
                if not ((abs(arr-ref) <= 1e-12*(1+abs(ref))) | (pd.isna(arr)&pd.isna(ref))).all():
                    bad=True; conflicts.append({**dict(zip(idc,key)),"column":c,"values":"|".join(map(str,vals.unique()[:20])),"sources":row["source"]})
                row[c]=ref
            else:
                u=vals.astype(str).unique()
                if len(u)>1:
                    bad=True; conflicts.append({**dict(zip(idc,key)),"column":c,"values":"|".join(u[:20]),"sources":row["source"]})
                row[c]=u[0]
        if not bad: rows.append(row)
    if conflicts:
        pd.DataFrame(conflicts).to_csv(out/f"conflicts_{sid}.csv",index=False)
    return pd.DataFrame(rows), len(conflicts)


def select_preferred_candidates(cand: pd.DataFrame):
    """Prefer authoritative production aggregates/shards for studies with known preflight copies."""
    parts=[]; audit=[]
    for sid,g in cand.groupby("study_id",sort=False):
        before=len(g); chosen=g.copy(); rule="all_candidates"
        src=g.source.astype(str)
        if sid == "legacy_coefficient_replay":
            mask=src.str.contains(r"output/production/(REPLAY_RESULTS_FULL|LEGACY_COEFFICIENT_FIELD_REPLICATIONS)\.csv\.gz$",case=False,regex=True)
            if mask.any(): chosen=g[mask].copy(); rule="production_consolidated_replay_files"
        elif sid == "matched_hlambda":
            mask=src.str.contains(r"output/production/MATCHED_HLAMBDA_REPLICATION_METRICS\.csv\.gz$",case=False,regex=True)
            if mask.any(): chosen=g[mask].copy(); rule="production_consolidated_matched_hlambda"
        elif sid == "matched_hlambda_stable":
            mask=src.str.contains(r"output/production/STABLE_ARITHMETIC_REPLICATION_METRICS\.csv\.gz$",case=False,regex=True)
            if mask.any(): chosen=g[mask].copy(); rule="production_consolidated_stable_arithmetic"
        elif sid == "higher_dimension":
            mask=src.str.contains(r"output/production1000/",case=False,regex=True)
            if mask.any(): chosen=g[mask].copy(); rule="production1000_only"
        parts.append(chosen)
        audit.append({"study_id":sid,"candidates_before":before,"candidates_selected":len(chosen),"selection_rule":rule})
    return pd.concat(parts,ignore_index=True) if parts else cand.iloc[0:0].copy(), pd.DataFrame(audit)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True)
    ap.add_argument("--io-workers",type=int,default=12,help="Archive readers; keep modest to avoid disk thrash")
    ap.add_argument("--compute-workers",type=int,default=80,help="Recorded for provenance; current aggregation uses vectorised NumPy/pandas")
    ap.add_argument("--outdir",default="outputs")
    ap.add_argument("--include-unclassified",action="store_true")
    ap.add_argument("--max-candidate-mb",type=float,default=4096)
    args=ap.parse_args()
    root=Path(args.root).resolve(); out=Path(args.outdir).resolve(); out.mkdir(parents=True,exist_ok=True)
    pkg=Path(__file__).resolve().parents[1]
    expected=load_expected(pkg/"config/expected_studies.csv")
    inv=scan_root(root,expected); atomic_csv(inv,out/"archive_inventory.csv")
    cand=inv[inv.kind.eq("replication_candidate")].copy()
    if not args.include_unclassified: cand=cand[cand.study_id.ne("unclassified")]
    cand=cand[cand.size_bytes <= args.max_candidate_mb*1024*1024]
    cand, source_audit = select_preferred_candidates(cand)
    atomic_csv(source_audit,out/"source_selection_audit.csv")
    print(f"Inventory: {len(inv)} data members/files; processing {len(cand)} preferred replication candidates with {args.io_workers} I/O workers")
    ready_by_study={}; schema=[]
    with ThreadPoolExecutor(max_workers=max(1,args.io_workers)) as ex:
        fut={ex.submit(process_one,root,r):r.source for _,r in cand.iterrows()}
        for i,f in enumerate(as_completed(fut),1):
            x,meta=f.result(); schema.append(meta)
            if x is not None:
                ready_by_study.setdefault(meta['study_id'],[]).append(x)
            if i%25==0 or i==len(fut): print(f"Processed {i}/{len(fut)} candidate sources")
    schema_df=pd.DataFrame(schema); atomic_csv(schema_df,out/"schema_registry.csv")
    method_parts=[]; pair_parts=[]; dedupe_log=[]
    for sid,frames in ready_by_study.items():
        print(f"Summarising {sid}: {len(frames)} normalized sources")
        x=pd.concat(frames,ignore_index=True,sort=False)
        n0=len(x)
        # Collapse exact/complementary copies on identity; quarantine only genuinely conflicting non-null values.
        x, nconf = merge_identity_rows(x,sid,out)
        dedupe_log.append({"study_id":sid,"rows_before":n0,"rows_after_identity_merge":len(x),"rows_collapsed_or_complementary":n0-len(x),"conflicting_values":nconf})
        if x.empty: continue
        method_parts.append(summarize_long(x)); pair_parts.append(paired_bblr(x))
    method=pd.concat(method_parts,ignore_index=True) if method_parts else pd.DataFrame()
    paired=pd.concat(pair_parts,ignore_index=True) if pair_parts else pd.DataFrame()
    atomic_csv(pd.DataFrame(dedupe_log),out/"deduplication_audit.csv")
    atomic_csv(method,out/"uncertainty_method_cell_metric.csv")
    atomic_csv(paired,out/"uncertainty_paired_bblr_comparator.csv")
    cov=coverage(expected,schema_df,method,paired); atomic_csv(cov,out/"coverage_by_study.csv")
    rerun=cov[cov.coverage_status.ne("READY")].copy(); atomic_csv(rerun,out/"rerun_or_recovery_queue.csv")
    review=schema_df[schema_df.status.ne("ready")].copy(); atomic_csv(review,out/"manual_review_queue.csv")
    write_markdown(out/"GLOBAL_UNCERTAINTY_BACKFILL_REPORT.md",root,cov,schema_df)
    run={"root":str(root),"io_workers":args.io_workers,"compute_workers":args.compute_workers,"inventory_rows":len(inv),"candidates_processed":len(cand),"ready_sources":int((schema_df.status=='ready').sum()) if not schema_df.empty else 0}
    (out/"RUN_CONFIG.json").write_text(json.dumps(run,indent=2)+"\n")
    # Compact handback: summaries and audits only; raw archives remain in place.
    hand=Path.cwd()/"BBLR_Global_Uncertainty_Backfill_v1_0_1_HANDback.zip"
    with zipfile.ZipFile(hand,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(out.parent))
    digest=sha256(hand); (Path(str(hand)+'.sha256')).write_text(f"{digest}  {hand.name}\n")
    print("PASS_GLOBAL_UNCERTAINTY_BACKFILL")
    print(hand); print(str(hand)+'.sha256')
if __name__=="__main__": main()
