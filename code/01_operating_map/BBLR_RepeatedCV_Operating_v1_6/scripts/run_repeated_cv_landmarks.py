#!/usr/bin/env python3
"""HPC runner for the BBLR repeated-CV landmark experiment.

This runner is self-contained inside the v1.5 production package.  It writes
one compressed tuning-grid shard and one compact selector shard per Monte Carlo
chunk and is resumable.
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

from bblr.univariate import DGP_REGISTRY, SUPPORT_DESIGN_REGISTRY, generate_sample
from bblr.repeated_cv import (
    evaluate_bblr_cv_surface, evaluate_full_bblr_grid, mark_selectors, selector_rows,
)
from scripts.run_univariate_hpc import stable_seed, parse_float_list


@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: int
    cell_id: int
    label: str
    dgp: str
    design: str
    n_train: int
    noise_sd: float
    panel_block: str
    expected_regime: str
    reason: str
    rep_start: int
    rep_stop: int


def chunk_stem(s: ChunkSpec) -> str:
    return f"chunk_{s.chunk_id:06d}_cell{s.cell_id:02d}_{s.dgp}_{s.design}_rep{s.rep_start:05d}-{s.rep_stop-1:05d}"


def make_chunks(cells: pd.DataFrame, reps: int, chunk_reps: int) -> List[ChunkSpec]:
    out = []
    cid = 0
    for cell_id, row in enumerate(cells.itertuples(index=False)):
        for start in range(0, reps, chunk_reps):
            out.append(ChunkSpec(
                cid, cell_id, str(row.label), str(row.dgp), str(row.design),
                int(row.n_train), float(row.noise_sd),
                str(getattr(row, "panel_block", "")),
                str(getattr(row, "expected_regime", "")),
                str(getattr(row, "reason", "")),
                start, min(start + chunk_reps, reps),
            ))
            cid += 1
    return out


def run_chunk(spec: ChunkSpec, cfg: Dict[str, object], out_dir: str) -> Dict[str, object]:
    out = Path(out_dir)
    grid_dir = out / "grid_shards"
    sel_dir = out / "selector_shards"
    grid_dir.mkdir(parents=True, exist_ok=True)
    sel_dir.mkdir(parents=True, exist_ok=True)
    grid_path = grid_dir / f"{chunk_stem(spec)}_grid.csv.gz"
    sel_path = sel_dir / f"{chunk_stem(spec)}_selectors.csv.gz"
    if bool(cfg["resume"]) and grid_path.exists() and sel_path.exists() and grid_path.stat().st_size > 50 and sel_path.stat().st_size > 50:
        return {"chunk_id": spec.chunk_id, "status": "skipped", "grid_path": str(grid_path), "selector_path": str(sel_path), "seconds": 0.0}

    t0 = time.perf_counter()
    grid_frames = []
    sel_frames = []
    r_grid = cfg["r_grid"]
    eps_grid = cfg["eps_grid"]
    for rep in range(spec.rep_start, spec.rep_stop):
        # Total observed development information matches the legacy n+n_validation
        # design: default multiplier 2 means 1000 observations when n_train=500.
        n_dev = int(round(spec.n_train * float(cfg["development_multiplier"])))
        seed_base = stable_seed(int(cfg["seed_offset"]), spec.cell_id, rep, 700)
        x_dev, y_dev, _ = generate_sample(spec.dgp, n_dev, spec.noise_sd, seed_base, spec.design)
        x_test, y_test, _ = generate_sample(
            spec.dgp, int(cfg["n_test"]), spec.noise_sd,
            stable_seed(int(cfg["seed_offset"]), spec.cell_id, rep, 701), spec.design,
        )
        dgp = DGP_REGISTRY[spec.dgp]
        stability_grid = np.linspace(dgp.support[0], dgp.support[1], int(cfg["n_stability_grid"]))
        cv = evaluate_bblr_cv_surface(
            x_dev, y_dev, r_grid=r_grid, eps_grid=eps_grid,
            min_stage1=int(cfg["min_stage1"]), min_stage2=int(cfg["min_stage2"]),
            folds=int(cfg["folds"]), repeats=int(cfg["repeats"]),
            seed=stable_seed(int(cfg["seed_offset"]), spec.cell_id, rep, 702),
            stability_grid=stability_grid,
        )
        truth = evaluate_full_bblr_grid(
            x_dev, y_dev, x_test, y_test, dgp_code=spec.dgp,
            r_grid=r_grid, eps_grid=eps_grid,
            min_stage1=int(cfg["min_stage1"]), min_stage2=int(cfg["min_stage2"]),
            guard_width=float(cfg["guard_width"]),
            x_grid=np.linspace(dgp.support[0], dgp.support[1], int(cfg["n_grid_eval"])),
        )
        g = cv.merge(truth, on=["r", "eps"], how="inner", validate="one_to_one")
        g = mark_selectors(g, tolerances=cfg["tolerances"])
        meta = {
            "cell_id": spec.cell_id, "landmark_label": spec.label,
            "panel_block": spec.panel_block, "expected_regime": spec.expected_regime,
            "reason": spec.reason,
            "dgp": spec.dgp, "design": spec.design, "n_train_legacy": spec.n_train,
            "n_development": n_dev, "noise_sd": spec.noise_sd, "rep": rep,
        }
        for k, v in meta.items():
            g[k] = v
        # LLR is the exact eps=0 subset. Mark its outcome-CV selections for a
        # clean Stage-2 comparison using identical folds and full-fit sample.
        llr = g[g["eps"].eq(0.0)].copy()
        if len(llr):
            for risk_col, flag in [
                ("single_split_rmse", "llr_selected_single_split"),
                ("cv5_rmse", "llr_selected_cv5"),
                ("repeated_cv_rmse", "llr_selected_repeated_cv"),
            ]:
                g[flag] = False
                idx = int(llr.sort_values([risk_col, "r"], kind="mergesort").index[0])
                g.loc[idx, flag] = True
        grid_frames.append(g)
        s = selector_rows(g)
        for k, v in meta.items():
            s[k] = v
        sel_frames.append(s)

    grid = pd.concat(grid_frames, ignore_index=True)
    sel = pd.concat(sel_frames, ignore_index=True)
    tmpg = grid_path.with_suffix(grid_path.suffix + f".tmp.{os.getpid()}")
    tmps = sel_path.with_suffix(sel_path.suffix + f".tmp.{os.getpid()}")
    grid.to_csv(tmpg, index=False, compression="gzip")
    sel.to_csv(tmps, index=False, compression="gzip")
    os.replace(tmpg, grid_path)
    os.replace(tmps, sel_path)
    return {
        "chunk_id": spec.chunk_id, "status": "written",
        "grid_path": str(grid_path), "selector_path": str(sel_path),
        "grid_rows": int(len(grid)), "selector_rows": int(len(sel)),
        "seconds": float(time.perf_counter() - t0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, default=ROOT / "RCV1000_OPERATING_CELLS.csv")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--development-multiplier", type=float, default=2.0)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--n-stability-grid", type=int, default=81)
    ap.add_argument("--n-grid-eval", type=int, default=401)
    ap.add_argument("--r-grid", default="0.08,0.12,0.18,0.26,0.38,0.55,0.80,1.20,2.00,4.00")
    ap.add_argument("--eps-grid", default="0,0.04,0.08,0.12,0.18,0.26,0.38,0.55,0.80,1.20,2.00,4.00")
    ap.add_argument("--tolerances", default="0.005,0.01,0.02")
    ap.add_argument("--min-stage1", type=int, default=15)
    ap.add_argument("--min-stage2", type=int, default=1)
    ap.add_argument("--guard-width", type=float, default=0.10)
    ap.add_argument("--seed-offset", type=int, default=1618033)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--chunk-reps", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()

    cells = pd.read_csv(args.cells)
    required = {"label", "dgp", "design", "n_train", "noise_sd"}
    if not required.issubset(cells.columns):
        raise SystemExit(f"cells file must contain {sorted(required)}")
    for d in cells.dgp:
        if d not in DGP_REGISTRY:
            raise SystemExit(f"unknown DGP {d}")
    allowed_designs = set(SUPPORT_DESIGN_REGISTRY) | {"uniform", "normal", "skewed", "gap"}
    for design in cells.design:
        if design not in allowed_designs:
            raise SystemExit(f"unknown support design {design}")

    cfg = {
        "folds": args.folds, "repeats": args.repeats,
        "development_multiplier": args.development_multiplier,
        "n_test": args.n_test, "n_stability_grid": args.n_stability_grid,
        "n_grid_eval": args.n_grid_eval,
        "r_grid": parse_float_list(args.r_grid), "eps_grid": parse_float_list(args.eps_grid),
        "tolerances": parse_float_list(args.tolerances),
        "min_stage1": args.min_stage1, "min_stage2": args.min_stage2,
        "guard_width": args.guard_width, "seed_offset": args.seed_offset,
        "jobs": args.jobs, "chunk_reps": args.chunk_reps, "resume": args.resume,
        "thread_env": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "cells_file": str(args.cells.resolve()),
    }
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "RUN_CONFIG.json").write_text(json.dumps(cfg, indent=2) + "\n")
    cells.to_csv(out / "LANDMARK_CELLS.csv", index=False)
    chunks = make_chunks(cells, args.reps, args.chunk_reps)
    pd.DataFrame([asdict(c) | {"grid_filename": f"{chunk_stem(c)}_grid.csv.gz", "selector_filename": f"{chunk_stem(c)}_selectors.csv.gz"} for c in chunks]).to_csv(out / "EXPECTED_CHUNKS.csv", index=False)

    t0 = time.perf_counter()
    if args.jobs == 1:
        results = [run_chunk(c, cfg, str(out)) for c in chunks]
    else:
        with parallel_config(backend="loky", n_jobs=args.jobs, inner_max_num_threads=1):
            results = Parallel(verbose=10, batch_size=1, pre_dispatch="2*n_jobs")(
                delayed(run_chunk)(c, cfg, str(out)) for c in chunks
            )
    elapsed = time.perf_counter() - t0
    manifest = pd.DataFrame(results).sort_values("chunk_id")
    manifest.to_csv(out / "CHUNK_MANIFEST.csv", index=False)
    missing = []
    for c in chunks:
        stem = chunk_stem(c)
        if not (out / "grid_shards" / f"{stem}_grid.csv.gz").exists():
            missing.append(f"grid:{stem}")
        if not (out / "selector_shards" / f"{stem}_selectors.csv.gz").exists():
            missing.append(f"selectors:{stem}")
    audit = {
        "expected_chunks": len(chunks), "present_chunks": len(chunks) - len(set(x.split(':',1)[1] for x in missing)),
        "missing_artifacts": len(missing), "missing_files": missing[:100],
        "wall_seconds": elapsed, "jobs": args.jobs, "chunk_reps": args.chunk_reps,
        "folds": args.folds, "repeats": args.repeats,
        "status": "PASS" if not missing else "FAIL",
    }
    (out / "COMPLETION_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))

if __name__ == "__main__":
    main()
