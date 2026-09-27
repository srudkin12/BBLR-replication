# BBLR Univariate Power v1.4

This stage is an intensive diagnostic follow-up to the support-regime experiment.
It is designed to identify **why** BBLR differs from conventional smoothers and
how much useful Stage-2 coefficient smoothing can be selected using outcomes
alone.

## Main methodological changes

1. **Exact nesting:** `epsilon=0` disables Stage 2, so `BBLR(r,0)=LLR(r)` exactly.
   This gives BBLR a genuine no-second-stage option rather than approximating it
   using a very small positive epsilon.
2. **Feasible alternative selectors:** in addition to minimum validation-Y RMSE,
   each method records a one-SE/low-smoothing selector and 0.5%/1% prediction-
   tolerance low-smoothing selectors. None uses the true simulated slope.
3. **Support-aware comparators:** `CSPLINE`, `CAKLLR` and `CAKLP2` infer support
   components from training X only. Candidate spacing thresholds are tuned via
   validation outcomes and include `inf`, which means no support split.
4. **Existing-shard intensive analysis:** the complete v1.3 tuning clouds can be
   analysed without rerunning Monte Carlo.
5. **Landmark design:** only 12 pre-specified cells are rerun initially, spanning
   global-smooth, support-boundary, soft-sparse and hard-gap regimes.

## Recommended order

Read `POWER_ANALYSIS_PLAN.md`, then follow `POWER_RUN.md`.

Stage A should be run first because it uses the already-completed v1.3 shards and
requires no new Monte Carlo. Stage B is the 140-core landmark experiment.

## Validation

The package passes 27 automated tests, including exact BBLR/LLR nesting and
support-aware/global comparator nesting when component splitting is disabled.
