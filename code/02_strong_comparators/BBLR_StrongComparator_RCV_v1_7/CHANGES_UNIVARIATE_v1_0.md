# Changes in BBLR univariate development v1.0

## Legacy estimator repair

- Removed the 1D dummy-coordinate workaround from `bblr/bblr2.py`.
- Made the neighbour index handle scalar covariates natively.
- Added explicit query-dimension validation.
- Added exact `query_knn()` fallback for minimum-neighbour requirements.
- Stage 2 now indexes training coefficient centres directly rather than concatenating train and test coordinates.
- Corrected the `epsilon_multiplied` fallback that previously retained only one returned index.
- Replaced a Python string identity comparison (`is not 'LPR2'`) with value comparison.
- Retained the original pre-fix estimator as `bblr/bblr2_legacy_pre_1d_fix.py`.

## New auditable 1D estimator laboratory

Added `bblr/univariate.py` with:

- exact sorted 1D radius/kNN search;
- `LocalLinear1D`;
- `LocalPolynomial1D` (degree configurable; LP2 used in experiments);
- `BBLR1D` with distinct Stage-1 radius `r` and Stage-2 radius `eps`;
- distinct Stage-1 and Stage-2 minimum-neighbour controls;
- twelve DGPs with analytical derivatives;
- uniform, truncated-normal, skewed and gapped X-support designs;
- derivative guard masks for kinks/jumps;
- separate boundary/interior scoring utilities.

## New simulation workflow

Added `scripts/run_univariate_study.py`:

- derivative recovery is primary, function recovery separate;
- same-r LLR and LP2 comparators;
- grids over `r`, `eps`, sample size, noise and Stage-2 minimum centres;
- Stage-1 fits reused across all Stage-2 parameter values;
- deterministic outer Monte Carlo parallelism via joblib;
- diagnostic oracle outputs for slope-optimal versus function-optimal smoothing;
- local support/fallback diagnostics.

Added `scripts/summarize_univariate_study.py`:

- direct Stage-2 gain/loss versus same-r LLR;
- best derivative configuration by method;
- function-optimal versus slope-optimal parameter mismatch.

## Verification

Eight automated tests pass, covering:

- exact 1D radius search against brute force;
- exact kNN ordering;
- noiseless local-linear recovery;
- noiseless BBLR linear recovery;
- kink guard bands;
- analytical derivative formulas against finite differences;
- repaired legacy 1D neighbour search;
- repaired legacy BBLR linear recovery.

## Included diagnostic run

`results_univariate/pilot_30/` contains the 30-replication development run described in `PILOT_30_FINDINGS.md`. It is not intended as manuscript evidence.
