from __future__ import annotations
import re
from pathlib import Path
import pandas as pd


def load_expected(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def classify_study(source: str, expected: pd.DataFrame) -> tuple[str, str]:
    s = source.replace("\\", "/")
    matches = []
    for _, row in expected.iterrows():
        if re.search(str(row.patterns), s, flags=re.I):
            matches.append((str(row.study_id), str(row.priority)))
    if not matches:
        return "unclassified", "review"
    # Specific bivariate subfamilies should beat generic names if multiple match.
    for preferred in ["bivariate_support_aware", "bivariate_fine_locality", "bivariate_broad"]:
        for m in matches:
            if m[0] == preferred:
                return m
    return matches[0]
