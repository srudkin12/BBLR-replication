from __future__ import annotations
import gzip, io, os, tempfile, zipfile
from pathlib import Path
import pandas as pd


def split_source(source: str):
    if "::" in source:
        return source.split("::", 1)
    return source, None


def read_source(root: str | Path, source: str) -> pd.DataFrame:
    root = Path(root)
    container, member = split_source(source)
    path = root / container
    if member is None:
        return _read_path(path)
    with zipfile.ZipFile(path) as z:
        ml = member.lower()
        if ml.endswith(".csv"):
            with z.open(member) as f:
                return pd.read_csv(f)
        if ml.endswith(".csv.gz"):
            with z.open(member) as f:
                with gzip.GzipFile(fileobj=f) as gz:
                    return pd.read_csv(gz)
        if ml.endswith((".parquet", ".feather")):
            suffix = ".parquet" if ml.endswith(".parquet") else ".feather"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(z.read(member)); t = tmp.name
            try:
                return _read_path(Path(t))
            finally:
                os.unlink(t)
        if ml.endswith(".jsonl"):
            with z.open(member) as f:
                return pd.read_json(f, lines=True)
    raise ValueError(f"Unsupported member: {member}")


def _read_path(path: Path) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".csv"): return pd.read_csv(path)
    if p.endswith(".csv.gz"): return pd.read_csv(path, compression="gzip")
    if p.endswith(".parquet"): return pd.read_parquet(path)
    if p.endswith(".feather"): return pd.read_feather(path)
    if p.endswith(".jsonl"): return pd.read_json(path, lines=True)
    raise ValueError(path)
