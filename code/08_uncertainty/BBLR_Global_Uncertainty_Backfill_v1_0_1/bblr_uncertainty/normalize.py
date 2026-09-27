from __future__ import annotations
import re
import numpy as np
import pandas as pd
from .config import CELL_SYNONYMS, REP_SYNONYMS, METHOD_SYNONYMS, METHOD_ALIASES, METRIC_RE, EXCLUDE_METRIC_RE

KNOWN_METHODS = sorted(set(METHOD_ALIASES.values()), key=len, reverse=True)


def _first_col(df, synonyms):
    low = {str(c).lower(): c for c in df.columns}
    return next((low[x] for x in synonyms if x in low), None)


def canonical_method(x):
    s = str(x).strip().upper().replace("_", "-")
    return METHOD_ALIASES.get(s, s)


def canonical_metric(study_id: str, method: str, metric: str) -> str:
    m = str(metric)
    # Matched-H-lambda stores matched-radius LLR metrics as llr_rB_*.
    # Once the method is identified as LLR, strip rB_ so BBLR/LLR are pairable.
    if study_id == "matched_hlambda" and method == "LLR" and m.lower().startswith("rb_"):
        return m[3:]
    return m


def composite_cell(df: pd.DataFrame, study_id: str, cell_col: str) -> pd.Series:
    # Legacy replay reuses numeric cell_id across UV92, UV27 and BIV101 suites.
    # The suite name is part of the design identity and must be retained.
    if study_id == "legacy_coefficient_replay" and "suite" in df.columns:
        return df["suite"].astype(str) + "::" + df[cell_col].astype(str)
    return df[cell_col]


def metric_columns(df):
    out = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        s = str(c)
        if re.search(METRIC_RE, s) and not re.search(EXCLUDE_METRIC_RE, s):
            out.append(c)
    return out


def normalize_long(df: pd.DataFrame, study_id: str, source: str):
    cell = _first_col(df, CELL_SYNONYMS)
    rep = _first_col(df, REP_SYNONYMS)
    method = _first_col(df, METHOD_SYNONYMS)
    metrics = metric_columns(df)
    notes = []
    if rep is None:
        return None, {"status":"reject", "reason":"no_replication_column", "source":source}
    if cell is None:
        # Some single-design audit files may genuinely omit cell id; source member then defines the cell.
        if df[rep].nunique(dropna=True) >= 100:
            df = df.copy(); df["__cell__"] = source.rsplit("/",1)[-1].split(".")[0]; cell="__cell__"
            notes.append("cell_from_source")
        else:
            return None, {"status":"reject", "reason":"no_cell_column", "source":source}
    if method is not None and metrics:
        keep = [cell, rep, method] + metrics
        x = df[keep].copy()
        x.columns = ["cell_id","rep","method"] + [str(c) for c in metrics]
        x["cell_id"] = composite_cell(df, study_id, cell).to_numpy()
        x["method"] = x["method"].map(canonical_method)
        # Study-specific selector labels in the operating map identify the actual estimator.
        if study_id == "operating_map":
            x["method"] = x["method"].replace({"SELECTED-REPEATED-CV":"BBLR", "LLR-SELECTED-REPEATED-CV":"LLR"})
        x["study_id"] = study_id; x["source"] = source
        dup = x.duplicated(["cell_id","rep","method"], keep=False)
        if dup.any():
            return None, {"status":"review", "reason":"duplicate_cell_rep_method", "duplicates":int(dup.sum()), "source":source,
                          "metrics":"|".join(map(str,metrics))}
        return x, {"status":"ready", "reason":"long_schema", "source":source, "rows":len(x), "cells":x.cell_id.nunique(),
                   "reps":x.rep.nunique(), "methods":x.method.nunique(), "metrics":"|".join(map(str,metrics)), "notes":"|".join(notes)}

    # Wide-method schema: columns such as bblr_* / llr_* / spline_*.
    wide = []
    for c in df.columns:
        cs = str(c)
        for m in KNOWN_METHODS:
            pat = re.compile(rf"(?i)^{re.escape(m).replace('\\-','[-_]')}[_-](.+)$")
            mm = pat.match(cs)
            if mm and re.search(METRIC_RE, mm.group(1)) and not re.search(EXCLUDE_METRIC_RE, mm.group(1)):
                wide.append((c,m,mm.group(1)))
                break
    if wide:
        frames=[]
        for m in sorted(set(x[1] for x in wide)):
            cols=[x for x in wide if x[1]==m]
            z=df[[cell,rep]+[x[0] for x in cols]].copy()
            cm = canonical_method(m)
            z.columns=["cell_id","rep"]+[canonical_metric(study_id, cm, x[2]) for x in cols]
            z["cell_id"] = composite_cell(df, study_id, cell).to_numpy()
            z["method"]=cm; z["study_id"]=study_id; z["source"]=source
            frames.append(z)
        x=pd.concat(frames,ignore_index=True,sort=False)
        dup=x.duplicated(["cell_id","rep","method"],keep=False)
        if dup.any():
            return None, {"status":"review","reason":"duplicate_wide_cell_rep_method","source":source,"duplicates":int(dup.sum())}
        return x, {"status":"ready","reason":"wide_method_schema","source":source,"rows":len(x),"cells":x.cell_id.nunique(),"reps":x.rep.nunique(),"methods":x.method.nunique(),"metrics":"|".join(sorted(set(x[2] for x in wide)))}

    return None, {"status":"review", "reason":"no_method_or_wide_method_metrics", "source":source, "numeric_metrics":"|".join(map(str,metrics))}
