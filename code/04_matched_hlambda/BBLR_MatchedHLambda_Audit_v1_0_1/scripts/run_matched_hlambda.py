#!/usr/bin/env python3
"""Run the frozen 92,000-replication matched-H_lambda mechanism audit.

No cross-validation is rerun.  Each row contains the already selected repeated-CV
BBLR (r, eps) pair from the frozen v1.6 operating map.  Development and test
samples are regenerated from the original deterministic seeds.  The script then
evaluates:

1. production BBLR at the frozen selected pair;
2. strict one-stage matched-W_lambda local-linear derivative estimator;
3. safeguard-matched W_lambda estimator (secondary diagnostic);
4. exact matched-radius production LLR.

This isolates finite-sample behaviour beyond the first-order composite derivative
kernel while preserving the original selection path.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bblr.univariate import DGP_REGISTRY, SUPPORT_DESIGN_REGISTRY, differentiable_mask, generate_sample
from bblr.univariate_fast import FastBBLR1D, FastLocalLinear1D
from bblr.matched_kernel import MatchedHLocalLinear1D, I_lambda, b_lambda, gamma_lambda

SEED_OFFSET = 1618033
N_TEST = 3000
N_GRID = 401
GUARD_WIDTH = 0.10
MIN_STAGE1 = 15
MIN_STAGE2 = 1


def stable_seed(base: int, *parts: int) -> int:
    ss = np.random.SeedSequence([int(base), *map(int, parts)])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def _finite_mse(est, truth, mask=None):
    est = np.asarray(est, float)
    truth = np.asarray(truth, float)
    ok = np.isfinite(est) & np.isfinite(truth)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    n = int(ok.sum())
    if n == 0:
        return np.nan, 0
    d = est[ok] - truth[ok]
    return float(np.mean(d * d)), n


def _pair_mse(a, b, mask=None):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    if not np.any(ok):
        return np.nan
    d = a[ok] - b[ok]
    return float(np.mean(d*d))


def structural_support_mask(x: np.ndarray, design: str) -> np.ndarray:
    """Evaluation-only mask excluding known zero-density hard-gap interiors."""
    x = np.asarray(x, float)
    if design in SUPPORT_DESIGN_REGISTRY:
        spec = SUPPORT_DESIGN_REGISTRY[design]
        if spec.family == "hard_gap" and float(spec.parameter_value) > 0:
            return np.abs(x) >= float(spec.parameter_value) - 1e-14
    return np.ones(x.size, dtype=bool)


def _safe_share(a) -> float:
    a = np.asarray(a, bool)
    return float(np.mean(a)) if a.size else np.nan


def evaluate_row(row: Dict[str, object]) -> Dict[str, object]:
    cell_id = int(row["cell_id"]); rep = int(row["rep"])
    dgp_code = str(row["dgp"]); design = str(row["design"])
    n_dev = int(row["n_development"]); noise_sd = float(row["noise_sd"])
    r = float(row["r"]); eps = float(row["eps"])
    lam = eps / r

    x_dev, y_dev, _ = generate_sample(
        dgp_code, n_dev, noise_sd, stable_seed(SEED_OFFSET, cell_id, rep, 700), design
    )
    x_test, y_test, _ = generate_sample(
        dgp_code, N_TEST, noise_sd, stable_seed(SEED_OFFSET, cell_id, rep, 701), design
    )
    dgp = DGP_REGISTRY[dgp_code]
    x_grid = np.linspace(dgp.support[0], dgp.support[1], N_GRID)
    truth_test = np.asarray(dgp.derivative(x_test), float)
    truth_grid = np.asarray(dgp.derivative(x_grid), float)
    diff_test = differentiable_mask(x_test, dgp, GUARD_WIDTH)
    diff_grid = differentiable_mask(x_grid, dgp, GUARD_WIDTH)
    support_grid = structural_support_mask(x_grid, design)
    regular_grid = diff_grid & support_grid
    center_test = diff_test & (np.abs(x_test) <= 0.40)
    center_grid = regular_grid & (np.abs(x_grid) <= 0.40)
    outer_test = diff_test & (np.abs(x_test) > 0.40)
    outer_grid = regular_grid & (np.abs(x_grid) > 0.40)
    boundary_test = diff_test & (np.abs(x_test) >= 1.60)
    boundary_grid = regular_grid & (np.abs(x_grid) >= 1.60)

    bblr = FastBBLR1D(r, eps, MIN_STAGE1, MIN_STAGE2).fit(x_dev, y_dev)
    _, b_test, bd_test = bblr.predict_with_slope(x_test)
    _, b_grid, bd_grid = bblr.predict_with_slope(x_grid)

    w_strict = MatchedHLocalLinear1D(r, lam, MIN_STAGE1, safe=False).fit(x_dev, y_dev)
    _, ws_test, wsd_test = w_strict.predict_with_slope(x_test)
    _, ws_grid, wsd_grid = w_strict.predict_with_slope(x_grid)

    w_safe = MatchedHLocalLinear1D(r, lam, MIN_STAGE1, safe=True).fit(x_dev, y_dev)
    _, wf_test, wfd_test = w_safe.predict_with_slope(x_test)
    _, wf_grid, wfd_grid = w_safe.predict_with_slope(x_grid)

    llr = FastLocalLinear1D(r, MIN_STAGE1).fit(x_dev, y_dev)
    _, l_test, ld_test = llr.predict_with_slope(x_test)
    _, l_grid, ld_grid = llr.predict_with_slope(x_grid)

    out = dict(row)
    out.update({
        "lambda": float(lam), "I_lambda": I_lambda(lam),
        "b_lambda": b_lambda(lam), "Gamma_lambda": gamma_lambda(lam),
        "lambda_stratum": "eps0" if eps == 0 else ("lt1" if lam < 1-1e-12 else ("eq1" if abs(lam-1)<=1e-12 else "gt1")),
        "gamma_abs_lt_0p01": bool(abs(gamma_lambda(lam)) < 0.01),
        "gamma_abs_lt_0p05": bool(abs(gamma_lambda(lam)) < 0.05),
    })

    # Global and structurally supported risks.
    method_arrays = {
        "bblr": (b_test, b_grid),
        "wstrict": (ws_test, ws_grid),
        "wsafe": (wf_test, wf_grid),
        "llr_rB": (l_test, l_grid),
    }
    for name, (te, gr) in method_arrays.items():
        out[f"{name}_test_mse"], out[f"{name}_test_n"] = _finite_mse(te, truth_test, diff_test)
        out[f"{name}_grid_mse"], out[f"{name}_grid_n"] = _finite_mse(gr, truth_grid, diff_grid)
        out[f"{name}_supported_grid_mse"], out[f"{name}_supported_grid_n"] = _finite_mse(gr, truth_grid, regular_grid)
        out[f"{name}_center_test_mse"], out[f"{name}_center_test_n"] = _finite_mse(te, truth_test, center_test)
        out[f"{name}_center_grid_mse"], out[f"{name}_center_grid_n"] = _finite_mse(gr, truth_grid, center_grid)
        out[f"{name}_outer_test_mse"], out[f"{name}_outer_test_n"] = _finite_mse(te, truth_test, outer_test)
        out[f"{name}_outer_grid_mse"], out[f"{name}_outer_grid_n"] = _finite_mse(gr, truth_grid, outer_grid)
        out[f"{name}_boundary_test_mse"], out[f"{name}_boundary_test_n"] = _finite_mse(te, truth_test, boundary_test)
        out[f"{name}_boundary_grid_mse"], out[f"{name}_boundary_grid_n"] = _finite_mse(gr, truth_grid, boundary_grid)

    # Common-support strict-W risks are the primary paired comparison whenever
    # strict W is undefined at any evaluation location.  This prevents unequal
    # location sets from masquerading as an estimator-risk difference.
    common_test = diff_test & wsd_test["valid"]
    common_grid = diff_grid & wsd_grid["valid"]
    common_supported_grid = regular_grid & wsd_grid["valid"]
    out["bblr_wstrict_common_test_mse"], out["wstrict_common_test_n"] = _finite_mse(b_test, truth_test, common_test)
    out["wstrict_common_test_mse"], _ = _finite_mse(ws_test, truth_test, common_test)
    out["bblr_wstrict_common_grid_mse"], out["wstrict_common_grid_n"] = _finite_mse(b_grid, truth_grid, common_grid)
    out["wstrict_common_grid_mse"], _ = _finite_mse(ws_grid, truth_grid, common_grid)
    out["bblr_wstrict_common_supported_grid_mse"], out["wstrict_common_supported_grid_n"] = _finite_mse(b_grid, truth_grid, common_supported_grid)
    out["wstrict_common_supported_grid_mse"], _ = _finite_mse(ws_grid, truth_grid, common_supported_grid)

    # Estimator disagreement directly measures finite-sample non-equivalence.
    out["bblr_minus_wstrict_test_disagreement_mse"] = _pair_mse(b_test, ws_test, diff_test)
    out["bblr_minus_wstrict_supported_grid_disagreement_mse"] = _pair_mse(b_grid, ws_grid, regular_grid)
    out["bblr_minus_wsafe_test_disagreement_mse"] = _pair_mse(b_test, wf_test, diff_test)
    out["bblr_minus_wsafe_supported_grid_disagreement_mse"] = _pair_mse(b_grid, wf_grid, regular_grid)

    # Diagnostics distinguish strict-kernel availability from safeguard effects.
    out.update({
        "wstrict_test_valid_share": _safe_share(wsd_test["valid"] & diff_test),
        "wstrict_grid_valid_share": _safe_share(wsd_grid["valid"] & diff_grid),
        "wstrict_supported_grid_valid_share": float(np.mean(wsd_grid["valid"][regular_grid])) if np.any(regular_grid) else np.nan,
        "wstrict_test_min_positive_weight_count": int(np.min(wsd_test["positive_weight_count"])),
        "wstrict_grid_min_positive_weight_count": int(np.min(wsd_grid["positive_weight_count"])),
        "wsafe_test_fallback_share": _safe_share(wfd_test["fallback"] & diff_test),
        "wsafe_grid_fallback_share": _safe_share(wfd_grid["fallback"] & diff_grid),
        "wsafe_supported_grid_fallback_share": float(np.mean(wfd_grid["fallback"][regular_grid])) if np.any(regular_grid) else np.nan,
        "llr_test_fallback_share": _safe_share(ld_test["fallback"] & diff_test),
        "llr_grid_fallback_share": _safe_share(ld_grid["fallback"] & diff_grid),
        "bblr_stage1_centre_fallback_share": float(np.mean(bd_grid["stage1_fallback"])),
        "bblr_test_stage2_fallback_share": float(np.mean(bd_test.get("stage2_fallback", np.zeros(len(x_test), bool)))),
        "bblr_grid_stage2_fallback_share": float(np.mean(bd_grid.get("stage2_fallback", np.zeros(len(x_grid), bool)))),
    })

    # Frozen-archive reconstruction audit.  These must agree up to numerical roundoff.
    archived_test_rmse = float(row["test_slope_rmse"])
    archived_grid_rmse = float(row["grid_slope_rmse"])
    recomputed_test_rmse = float(np.sqrt(out["bblr_test_mse"]))
    recomputed_grid_rmse = float(np.sqrt(out["bblr_grid_mse"]))
    out["bblr_recomputed_test_rmse"] = recomputed_test_rmse
    out["bblr_recomputed_grid_rmse"] = recomputed_grid_rmse
    out["archive_test_rmse_abs_diff"] = abs(recomputed_test_rmse - archived_test_rmse)
    out["archive_grid_rmse_abs_diff"] = abs(recomputed_grid_rmse - archived_grid_rmse)

    # eps=0 safe-W must reproduce production LLR/BBLR exactly, including safeguard policy.
    if eps == 0.0:
        out["eps0_max_abs_bblr_wsafe_test"] = float(np.nanmax(np.abs(b_test - wf_test)))
        out["eps0_max_abs_bblr_wsafe_grid"] = float(np.nanmax(np.abs(b_grid - wf_grid)))
    else:
        out["eps0_max_abs_bblr_wsafe_test"] = np.nan
        out["eps0_max_abs_bblr_wsafe_grid"] = np.nan
    return out


@dataclass(frozen=True)
class Chunk:
    chunk_id: int
    start: int
    stop: int


def _chunk_stem(c: Chunk) -> str:
    return f"chunk_{c.chunk_id:05d}_rows{c.start:06d}-{c.stop-1:06d}"


def run_chunk(c: Chunk, rows: List[Dict[str, object]], output_dir: str, resume: bool):
    out = Path(output_dir)
    shard_dir = out / "replication_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"{_chunk_stem(c)}.csv.gz"
    if resume and path.exists() and path.stat().st_size > 100:
        return {"chunk_id": c.chunk_id, "status": "skipped", "path": str(path), "seconds": 0.0}
    t0 = time.perf_counter()
    ans = [evaluate_row(rows[i]) for i in range(c.start, c.stop)]
    df = pd.DataFrame(ans)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    df.to_csv(tmp, index=False, compression="gzip")
    os.replace(tmp, path)
    return {"chunk_id": c.chunk_id, "status": "written", "path": str(path), "rows": len(df), "seconds": time.perf_counter()-t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", type=Path, default=ROOT / "data/FROZEN_SELECTED_CONFIGS.csv.gz")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--chunk-rows", type=int, default=25)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0, help="development only; 0 means all rows")
    args = ap.parse_args()

    df = pd.read_csv(args.configs).sort_values(["cell_id","rep"]).reset_index(drop=True)
    if args.max_rows > 0:
        df = df.iloc[:args.max_rows].copy()
    rows = df.to_dict("records")
    chunks=[]
    for cid,start in enumerate(range(0,len(rows),args.chunk_rows)):
        chunks.append(Chunk(cid,start,min(start+args.chunk_rows,len(rows))))

    out = args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/"FROZEN_SELECTED_CONFIGS_USED.csv.gz",index=False,compression="gzip")
    pd.DataFrame([asdict(c)|{"filename":f"{_chunk_stem(c)}.csv.gz"} for c in chunks]).to_csv(out/"EXPECTED_CHUNKS.csv",index=False)
    cfg={
        "configs":str(args.configs.resolve()),"rows":len(rows),"jobs":args.jobs,
        "chunk_rows":args.chunk_rows,"resume":args.resume,
        "seed_offset":SEED_OFFSET,"n_test":N_TEST,"n_grid":N_GRID,
        "guard_width":GUARD_WIDTH,"min_stage1":MIN_STAGE1,"min_stage2":MIN_STAGE2,
        "primary_comparator":"strict W_lambda local linear at frozen BBLR-selected (r,eps)",
        "secondary_comparator":"W_lambda with BBLR Stage-1 15-NN safeguard when under-supported/singular",
        "thread_env":{v:os.environ.get(v) for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")},
    }
    (out/"RUN_CONFIG.json").write_text(json.dumps(cfg,indent=2)+"\n")

    t0=time.perf_counter()
    if args.jobs == 1:
        results=[run_chunk(c,rows,str(out),args.resume) for c in chunks]
    else:
        with parallel_config(backend="loky", n_jobs=args.jobs, inner_max_num_threads=1):
            results=Parallel(verbose=10,batch_size=1,pre_dispatch="2*n_jobs")(
                delayed(run_chunk)(c,rows,str(out),args.resume) for c in chunks
            )
    pd.DataFrame(results).to_csv(out/"CHUNK_RUN_LOG.csv",index=False)

    missing=[]
    for c in chunks:
        p=out/"replication_shards"/f"{_chunk_stem(c)}.csv.gz"
        if not p.exists() or p.stat().st_size <= 100:
            missing.append(str(p.name))
    audit={
        "expected_rows":len(rows),"expected_chunks":len(chunks),
        "missing_chunks":len(missing),"missing_files":missing[:100],
        "wall_seconds":time.perf_counter()-t0,"jobs":args.jobs,
        "status":"PASS" if not missing else "FAIL",
    }
    (out/"COMPLETION_AUDIT.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit,indent=2))

if __name__ == "__main__":
    main()
