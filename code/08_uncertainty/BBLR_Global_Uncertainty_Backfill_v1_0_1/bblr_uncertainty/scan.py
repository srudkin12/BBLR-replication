from __future__ import annotations
import csv, gzip, io, os, re, zipfile, hashlib
from pathlib import Path
from typing import Iterable
import pandas as pd
from .config import CELL_SYNONYMS, REP_SYNONYMS, METHOD_SYNONYMS, METRIC_RE, EXCLUDE_METRIC_RE
from .studies import classify_study

DATA_EXT = (".csv", ".csv.gz", ".parquet", ".feather", ".jsonl")
MANIFEST_HINT = re.compile(r"(?i)(manifest|expected_chunks|inventory|file_list|index)")
RAW_HINT = re.compile(r"(?i)(replication|replications|rep_level|rep-level|risk|selected|metric|result|shard|chunk|pairwise)")
SUMMARY_HINT = re.compile(r"(?i)(summary|aggregate|cell_method|cell_pairwise|block_summary|automated_summary)")
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules"}


def _header_from_bytes(b: bytes, name: str) -> list[str]:
    try:
        if name.lower().endswith(".gz"):
            b = gzip.decompress(b)
        text = b.decode("utf-8-sig", errors="replace")
        return next(csv.reader(io.StringIO(text)))
    except Exception:
        return []


def _header_file(path: Path) -> list[str]:
    try:
        p = str(path).lower()
        if p.endswith(".csv"):
            return pd.read_csv(path, nrows=0).columns.astype(str).tolist()
        if p.endswith(".csv.gz"):
            return pd.read_csv(path, nrows=0, compression="gzip").columns.astype(str).tolist()
        if p.endswith(".parquet"):
            import pyarrow.parquet as pq
            return pq.ParquetFile(path).schema.names
        if p.endswith(".feather"):
            import pyarrow.feather as feather
            return feather.read_table(path, memory_map=True).column_names
        if p.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                line = f.readline()
            import json
            x = json.loads(line) if line else {}
            return list(x) if isinstance(x, dict) else []
    except Exception:
        return []
    return []


def _classify_columns(cols: list[str]) -> dict:
    low = {c.lower(): c for c in cols}
    cell = next((low[x] for x in CELL_SYNONYMS if x in low), None)
    rep = next((low[x] for x in REP_SYNONYMS if x in low), None)
    method = next((low[x] for x in METHOD_SYNONYMS if x in low), None)
    metrics = [c for c in cols if re.search(METRIC_RE, c) and not re.search(EXCLUDE_METRIC_RE, c)]
    return {"cell_col": cell, "rep_col": rep, "method_col": method, "metric_candidates": metrics}


def _candidate_kind(name: str, cols: list[str]) -> str:
    cc = _classify_columns(cols)
    lname = name.lower()
    if MANIFEST_HINT.search(lname):
        return "manifest"
    if cc["rep_col"] and (cc["metric_candidates"] or RAW_HINT.search(lname)):
        return "replication_candidate"
    if SUMMARY_HINT.search(lname) or (not cc["rep_col"] and cc["metric_candidates"]):
        return "summary_only"
    return "other"


def scan_root(root: str | Path, expected: pd.DataFrame, max_zip_header_bytes: int = 262144) -> pd.DataFrame:
    root = Path(root).resolve()
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            path = Path(dirpath) / fn
            rel = os.path.relpath(path, root)
            lower = fn.lower()
            if lower.endswith(".zip"):
                try:
                    with zipfile.ZipFile(path) as z:
                        for info in z.infolist():
                            if info.is_dir():
                                continue
                            n = info.filename
                            nl = n.lower()
                            if not nl.endswith(DATA_EXT):
                                continue
                            cols = []
                            if nl.endswith((".csv", ".csv.gz")):
                                try:
                                    with z.open(info) as f:
                                        b = f.read(min(max_zip_header_bytes, max(4096, info.file_size)))
                                    cols = _header_from_bytes(b, n)
                                except Exception:
                                    cols = []
                            source = f"{rel}::{n}"
                            study, priority = classify_study(source, expected)
                            cc = _classify_columns(cols)
                            rows.append({
                                "source": source, "container": rel, "member": n,
                                "size_bytes": info.file_size, "kind": _candidate_kind(n, cols),
                                "study_id": study, "priority": priority,
                                "cell_col": cc["cell_col"], "rep_col": cc["rep_col"], "method_col": cc["method_col"],
                                "metric_candidates": "|".join(cc["metric_candidates"]), "columns": "|".join(cols),
                            })
                except zipfile.BadZipFile:
                    rows.append({"source": rel, "container": rel, "member": "", "size_bytes": path.stat().st_size,
                                 "kind": "bad_zip", "study_id": "unclassified", "priority": "review",
                                 "cell_col": None, "rep_col": None, "method_col": None, "metric_candidates": "", "columns": ""})
            elif lower.endswith(DATA_EXT):
                cols = _header_file(path)
                study, priority = classify_study(rel, expected)
                cc = _classify_columns(cols)
                rows.append({
                    "source": rel, "container": rel, "member": "", "size_bytes": path.stat().st_size,
                    "kind": _candidate_kind(fn, cols), "study_id": study, "priority": priority,
                    "cell_col": cc["cell_col"], "rep_col": cc["rep_col"], "method_col": cc["method_col"],
                    "metric_candidates": "|".join(cc["metric_candidates"]), "columns": "|".join(cols),
                })
    return pd.DataFrame(rows)


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()
