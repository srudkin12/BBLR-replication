from __future__ import annotations
import csv,io,re,zipfile
from pathlib import Path
import pandas as pd

KNOWN=["strong_comparator_v1_strong_comparator_handback_20260907_053631.zip","rcv1000_operating_v1_repeated_cv_handback_20260906_201053.zip","univariate_paper_outputs_20260907_055021.zip"]

def inventory(root:str,outfile:str):
    root=Path(root); rows=[]
    candidates=[]
    for name in KNOWN: candidates.extend(root.rglob(name))
    # Also inspect obvious BBLR handback zips without descending huge unrelated archives.
    candidates.extend([p for p in root.glob("*.zip") if "BBLR" in p.name or "strong" in p.name.lower() or "univariate" in p.name.lower()])
    seen=set()
    for z in candidates:
        if z in seen: continue
        seen.add(z)
        try:
            with zipfile.ZipFile(z) as f:
                for info in f.infolist():
                    if not info.filename.lower().endswith('.csv'): continue
                    cols=""; nrows=""
                    try:
                        with f.open(info) as h:
                            line=h.readline().decode('utf-8','replace').strip(); cols=line
                    except Exception: pass
                    rows.append(dict(archive=str(z),member=info.filename,size_bytes=info.file_size,columns=cols))
        except Exception as e:
            rows.append(dict(archive=str(z),member="<ERROR>",size_bytes=0,columns=str(e)))
    df=pd.DataFrame(rows); Path(outfile).parent.mkdir(parents=True,exist_ok=True); df.to_csv(outfile,index=False); return df
