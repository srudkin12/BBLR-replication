# BBLR univariate HPC development v1.1

## Purpose

This package turns the corrected one-dimensional BBLR laboratory into a
production-oriented Monte Carlo runner suitable for a large Linux machine.
The key scientific correction relative to the v1.0 pilot is that tuning is now
based on **held-out observed outcome Y RMSE**, not RMSE against the known
noise-free conditional mean.

For every Monte Carlo replication the runner generates independent:

1. **training data** — estimator fitting;
2. **validation data** — hyperparameter selection using observed `Y` only;
3. **test data** — independent outcome RMSE plus simulation-only mean/slope
   diagnostics;
4. **uniform support grid** — shape, boundary and support-balanced derivative
   diagnostics.

Thus the simulation behaves like an empirical researcher at the tuning stage,
then reveals the known DGP only after selection.

## Core output coordinates

The proposed performance map uses:

- x axis: `test_y_rmse`;
- y axis: `test_slope_rmse` (distribution-weighted) or `grid_slope_rmse`
  (support-balanced).

Every tuning configuration is retained.  Rows also contain:

- `validation_y_rmse` (the feasible tuning criterion);
- `selected_by_validation_y`;
- `test_mean_rmse`;
- `oracle_slope_within_method` (diagnostic only);
- `oracle_mean_within_method` (diagnostic only);
- Pareto flags for all configurations and for the outcome-selected estimators;
- boundary/interior slope RMSE;
- Stage-1/Stage-2 fallback diagnostics;
- numerical DGP-shape descriptors (slope dispersion, RMS curvature, total
  variation of the slope surface).

## Comparator set

Production runner currently supports:

- `OLS` — exact linear sentinel;
- `LLR` — uniform-ball local-linear regression, the essential Stage-1/nested
  BBLR comparator;
- `LP2` — local quadratic regression;
- `KLLR` — tricube-weighted local-linear regression, testing whether BBLR gains
  survive comparison with conventional smooth local weighting;
- `SPLINE` — smoothing spline with validation-selected smoothing parameter;
- `BBLR` — fast mean-aggregation two-stage estimator.

Further comparators can be added after the first mapping pass, but this set is
strong enough to diagnose whether BBLR's second smoothing stage contributes
something beyond a hard-ball local line.

## Parallel architecture

The unit of parallel work is a **chunk of Monte Carlo replications within one
DGP/design/n/noise condition**.  This has four advantages:

1. Stage-1 BBLR fits are reused across every epsilon inside a replication.
2. There is no nested estimator-level parallelism.
3. Process-launch/serialization overhead is amortised across several reps.
4. Every worker writes a deterministic compressed shard, so the full run is
   resumable and the parent process never stores the full result table in RAM.

The runner forces OMP/OpenBLAS/MKL/NumExpr to one thread per worker to prevent
150 processes from each starting their own BLAS thread pool.

Random seeds use NumPy `SeedSequence` with structured indices.  The previous
arithmetic seed construction could collide once replication counts or grids
became large; v1.1 removes that risk.

## Exactness and speed

`bblr/univariate_fast.py` uses sorted one-dimensional data and prefix sums for
hard-ball local-linear fits and mean-aggregation BBLR.  Automated tests compare
it directly with the transparent v1.0 reference implementation.  Current test
suite: **11 passing tests**.

A local microbenchmark at n=1000, 401 queries gave approximately:

- reference LLR: 0.0092 s; fast LLR: 0.00035 s;
- reference BBLR: 0.0243 s; fast BBLR: 0.00048 s.

These timings are only local implementation diagnostics, not predictions for a
large-machine production run.

## Recommended staged use of the large machine

### Preflight

First verify the environment and a small parallel run:

```bash
pytest -q

python scripts/run_univariate_hpc.py \
  --dgps U01,U02,U04,U08 \
  --designs uniform,gap \
  --n-grid 500 \
  --noise-grid 0.25 \
  --reps 10 \
  --jobs 8 \
  --chunk-reps 2 \
  --output-dir results_hpc/preflight

python scripts/summarize_univariate_hpc.py \
  --run-dir results_hpc/preflight
```

The run is green only if `COMPLETION_AUDIT.json` reports `PASS`.

### First high-core mapping pass

The first substantive pass should concentrate on the smooth DGPs U01-U08,
because the immediate objective is to understand how increasing curvature,
oscillation and localised nonlinearity move estimators through outcome-RMSE /
slope-RMSE space.  Kinks/jumps U09-U12 are better treated as a separate stress
class after smooth behaviour is understood.

Example 140-worker run (leaves headroom on a 150-core allocation):

```bash
python scripts/run_univariate_hpc.py \
  --dgps U01,U02,U03,U04,U05,U06,U07,U08 \
  --designs uniform,normal,gap \
  --n-grid 250,500,1000 \
  --noise-grid 0.10,0.25,0.50 \
  --reps 200 \
  --n-test 2000 \
  --r-grid 0.10,0.15,0.22,0.30,0.42,0.60 \
  --eps-grid 0.05,0.10,0.15,0.22,0.30,0.42,0.60,0.85 \
  --spline-lambda-grid 1e-5,1e-4,1e-3,1e-2,1e-1,1,10 \
  --jobs 140 \
  --chunk-reps 20 \
  --resume \
  --output-dir results_hpc/smooth_map_v1

python scripts/summarize_univariate_hpc.py \
  --run-dir results_hpc/smooth_map_v1
```

This design contains 43,200 Monte Carlo replications before method/tuning-grid
expansion.  Every tuning point is retained in compressed shards.

## DGP strategy beyond the first mapping pass

The current twelve DGPs should be treated as prototypes, not the final design.
The next development should create **parametric nonlinearity ladders** so that
BBLR performance can be related to controlled changes in the DGP rather than a
collection of unrelated functions.  Priority families are:

1. quadratic curvature ladder: `m(x)=x+a x^2`, varying `a`;
2. oscillation ladder: `m(x)=x+a sin(omega x)`, varying frequency and amplitude;
3. transition ladder: `tanh(k x)`, varying steepness `k`;
4. local-bump ladder: varying bump width/amplitude;
5. slope-kink ladder: varying the size of the slope discontinuity;
6. support geometry crossed separately with each family.

Useful DGP-level explanatory objects include RMS curvature, slope dispersion,
slope total variation and local support density.  These allow us to ask not
just *which DGP BBLR wins*, but *which properties of the DGP predict movement
along the prediction/slope frontier*.

## Files

- `bblr/univariate_fast.py` — fast exact 1D LLR/BBLR and tricube LLR comparator.
- `scripts/run_univariate_hpc.py` — sharded/resumable parallel production runner.
- `scripts/summarize_univariate_hpc.py` — incremental aggregation of shards.
- `tests/test_univariate_fast.py` — exact equivalence and linear-recovery tests.
