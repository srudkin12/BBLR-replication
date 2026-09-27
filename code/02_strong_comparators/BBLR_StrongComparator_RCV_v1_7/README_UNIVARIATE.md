# BBLR one-dimensional development pass v1.0

## Purpose

This pass makes the one-dimensional case an explicit methodological laboratory rather than one small part of the old simulation grid.

Two changes are deliberately separated:

1. **Legacy estimator repair.** `bblr/bblr2.py` now handles one-dimensional data natively. The old dummy-coordinate workaround has been removed, radius queries validate dimensionality, and the minimum-points fallback uses exact nearest neighbours rather than an arbitrary enlarged radius.
2. **Clean 1D research implementation.** `bblr/univariate.py` provides a small, auditable implementation of BBLR in one dimension, a single-stage local-linear comparator, exact 1D neighbour searches, analytical DGP derivatives, and support-design generators.

The pre-fix estimator is retained as `bblr/bblr2_legacy_pre_1d_fix.py`.

## Why the clean 1D implementation matters

For a query `x0`, BBLR is written explicitly as two stages:

- Stage 1: estimate a local line around every training centre using radius `r`;
- Stage 2: average those local slopes over centres within radius `eps` of `x0`.

`r` and `eps` are therefore not forced to be equal. Their separation is central to testing whether BBLR's second smoothing stage adds value to ordinary local-linear regression.

The main comparison is therefore **BBLR vs single-stage LLR using the same Stage-1 radius**. This isolates the contribution of coefficient smoothing more cleanly than a generic prediction comparison.

## DGP set

Smooth DGPs:

- U01 linear
- U02 quadratic
- U03 cubic
- U04 sinusoid
- U05 linear + sine
- U06 mild exponential
- U07 smooth tanh transition
- U08 local Gaussian bump

Non-smooth stress tests:

- U09 absolute-value kink
- U10 slope kink
- U11 level jump
- U12 step function

For U09-U12, derivative RMSE is scored only away from a configurable guard band around kinks/jumps. This avoids assigning a conventional derivative where it does not exist.

## X-support designs

Functional form and design geometry are varied separately:

- `uniform`: even support coverage;
- `normal`: dense centre / sparse boundaries;
- `skewed`: asymmetric support;
- `gap`: two clusters with sparse central support.

The `gap` design is particularly useful for studying the minimum-points fallback and whether Stage-2 smoothing traverses regions with little data.

## Metrics

The runner reports, separately:

- function RMSE against the noise-free conditional mean `m(x)`;
- derivative/slope RMSE against `m'(x)`;
- derivative bias;
- interior derivative RMSE;
- boundary derivative RMSE;
- Stage-1 fallback frequency and neighbourhood size;
- Stage-2 fallback frequency and number of coefficient centres.

It also writes **oracle diagnostic choices** for the `(r, eps)` pair that minimises function RMSE versus slope RMSE. These are not proposed as practical tuning rules. They diagnose whether prediction-optimal smoothing is misaligned with BBLR's coefficient/gradient estimand.

## First run

From the project root:

```bash
python -m pytest -q
python scripts/run_univariate_study.py \
  --reps 10 \
  --n 400 \
  --dgps U01,U02,U04,U08,U10,U11 \
  --designs uniform,normal,gap \
  --r-grid 0.15,0.25,0.40,0.60 \
  --eps-grid 0.10,0.20,0.35,0.55 \
  --noise-sd 0.25
```

For production, increase `--reps` (for example 500 or 1000) only after the pilot surfaces look sensible.

## Outputs

`results_univariate/` receives:

- `univariate_metrics.csv`: replicate-level results;
- `univariate_summary.csv`: Monte Carlo means by DGP/design/method/parameter pair;
- `univariate_oracle_choices.csv`: slope-optimal vs function-optimal BBLR parameter choices;
- `univariate_rep0_curves.csv`: ready-to-plot mean and slope curves for diagnostic figures.

## Recommended empirical sequence

1. U01 with zero noise: strict correctness sentinel; slope should be recovered to numerical precision.
2. U02/U04/U08: smooth curvature of increasing local difficulty.
3. U10: kink without a level discontinuity.
4. U11: jump with constant derivative away from the jump.
5. Repeat under `normal` and `gap` support.
6. Only after those are understood, expand sample-size and noise grids.

The most important question is not whether BBLR always wins. It is **where the second smoothing stage reduces variance in local slopes enough to compensate for its extra bias, and how that depends on support geometry, curvature, noise, `r`, and `eps`.**

## Additional comparator and production controls

The development runner also includes `LP2`, a centred local-quadratic regression. Its coefficient on `(x-x0)` estimates the derivative at the query point. Comparing BBLR with both `LLR` and `LP2` is important: BBLR should not be credited with a generic curvature correction that an ordinary local polynomial already provides.

Large experiments can vary sample size, noise and the Stage-2 minimum-centre rule directly:

```bash
python scripts/run_univariate_study.py \
  --reps 500 \
  --n-grid 250,500,1000 \
  --noise-grid 0.10,0.25,0.50 \
  --min-stage2-grid 1,5,15 \
  --jobs 80 \
  --backend multiprocessing
```

The default Stage-2 minimum is deliberately `1`, not the Stage-1 minimum. A local line needs multiple observations to be estimable, whereas Stage 2 can mathematically return a single already-estimated local slope. Forcing the same minimum at both stages introduces an additional k-nearest-neighbour smoothing rule whenever support is sparse. The new code exposes this as a separate design parameter so its effect can be studied rather than hidden.

`run_univariate_study.py` reuses each Stage-1 fit across the full `eps` grid for a given `r`; production work therefore does not recompute the expensive local coefficient surface for every Stage-2 radius.

After a run, create the main diagnostic comparison tables with:

```bash
python scripts/summarize_univariate_study.py results_univariate/univariate_metrics.csv
```

Important derived outputs include:

- `second_stage_gain_vs_same_r_llr.csv`: direct BBLR gain/loss relative to a single-stage local line with the identical Stage-1 radius;
- `best_slope_method_summary.csv`: best derivative-recovery configuration within each method;
- `function_vs_slope_optimal_summary.csv`: how often function-optimal and derivative-optimal BBLR smoothing parameters coincide, and the derivative penalty when they do not.
