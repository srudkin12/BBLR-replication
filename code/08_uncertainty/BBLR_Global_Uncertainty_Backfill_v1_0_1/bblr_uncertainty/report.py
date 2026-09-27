from __future__ import annotations
from pathlib import Path
import pandas as pd


def coverage(expected, schema, method_summary, paired_summary):
    rows=[]
    for _,e in expected.iterrows():
        sid=e.study_id
        sch=schema[schema.study_id.eq(sid)] if not schema.empty and "study_id" in schema else pd.DataFrame()
        ms=method_summary[method_summary.study_id.eq(sid)] if not method_summary.empty else pd.DataFrame()
        ps=paired_summary[paired_summary.study_id.eq(sid)] if not paired_summary.empty else pd.DataFrame()
        rows.append({
            "study_id":sid,"description":e.description,"priority":e.priority,
            "expected_cells":e.expected_cells,"expected_reps_per_cell":e.expected_reps_per_cell,"expected_replications":e.expected_replications,
            "ready_sources":int((sch.status=="ready").sum()) if not sch.empty else 0,
            "review_sources":int((sch.status=="review").sum()) if not sch.empty else 0,
            "observed_cells":int(ms.cell_id.nunique()) if not ms.empty else 0,
            "method_metric_summaries":len(ms),"paired_summaries":len(ps),
            "min_nonmissing_n":float(ms.loc[ms.n>0,"n"].min()) if (not ms.empty and "n" in ms and (ms.n>0).any()) else float("nan"),
            "max_nonmissing_n":float(ms.loc[ms.n>0,"n"].max()) if (not ms.empty and "n" in ms and (ms.n>0).any()) else float("nan"),
        })
    out=pd.DataFrame(rows)
    out["coverage_status"]=out.apply(lambda r: "READY" if r.observed_cells>=r.expected_cells and r.method_metric_summaries>0 else ("PARTIAL" if r.method_metric_summaries>0 else "MISSING"),axis=1)
    return out


def write_markdown(path, root, coverage_df, schema_df):
    lines=["# BBLR global uncertainty backfill report","",f"Project root scanned: `{root}`","",
           "## Interpretation","",
           "`q025`--`q975` are central 95% empirical replication intervals. They describe sampling variation across Monte Carlo replications.",
           "`mean_ci_*` quantify Monte Carlo uncertainty for the mean using mean ± 1.96 MCSE. `median_ci_*` are distribution-free 95% order-statistic confidence intervals for the population median.",
           "Paired BBLR/comparator summaries are created only from rows with unambiguous common study, cell, replication, method and metric identifiers.","",
           "## Coverage",""]
    if coverage_df.empty:
        lines.append("No expected studies were recovered.")
    else:
        lines.append("| Study | Status | Cells recovered / expected | Replication n range | Ready sources | Review sources | Paired summaries |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _,r in coverage_df.iterrows():
            lines.append(f"| {r.study_id} | {r.coverage_status} | {r.observed_cells} / {r.expected_cells} | {r.min_nonmissing_n:g}--{r.max_nonmissing_n:g} | {r.ready_sources} | {r.review_sources} | {r.paired_summaries} |")
    lines += ["","## Files requiring manual schema review",""]
    rev=schema_df[schema_df.status.eq("review")] if not schema_df.empty else pd.DataFrame()
    if rev.empty: lines.append("None.")
    else:
        for _,r in rev.head(100).iterrows(): lines.append(f"- `{r.source}` — {r.reason}")
        if len(rev)>100: lines.append(f"- … {len(rev)-100} additional review items in `schema_registry.csv`.")
    Path(path).write_text("\n".join(lines)+"\n",encoding="utf-8")
