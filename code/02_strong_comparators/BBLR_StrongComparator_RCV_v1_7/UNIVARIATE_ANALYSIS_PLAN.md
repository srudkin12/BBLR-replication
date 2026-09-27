# BBLR univariate analysis plan

## Central question

The revised one-dimensional analysis should establish **when smoothing local coefficient estimates is beneficial**, not merely whether BBLR predicts `Y` well.

For a conditional mean `m(x)=E[Y|X=x]`, the principal target is the derivative `m'(x)`. BBLR should therefore be evaluated as an estimator of a slope surface and compared first with estimators that target the same local object.

## Phase A — correctness sentinels

1. **U01 linear, zero noise, uniform X.** Every sensible radius should recover the constant slope to numerical precision except degenerate neighbourhood choices.
2. Repeat U01 under normal, skewed and gapped support. Any substantial error in the zero-noise case should be interpreted as an algorithm/neighbourhood problem rather than a statistical finding.
3. Verify radius-search membership against brute-force distances. This is automated in `tests/`.

No manuscript inference should proceed unless Phase A is green.

## Phase B — smooth functions

Use U02-U08 to span different sources of local curvature:

- U02 quadratic: derivative changes linearly;
- U03 cubic: asymmetric derivative magnitude;
- U04 sinusoid: rapidly changing derivative and repeated sign changes;
- U05 linear+sine: smooth nonlinearity around a positive baseline slope;
- U06 exponential: asymmetric curvature;
- U07 tanh transition: concentrated derivative change around the centre;
- U08 Gaussian bump: strongly localised curvature.

Primary comparison: **BBLR versus LLR at the same Stage-1 radius `r`**. This directly measures what Stage-2 smoothing contributes.

Secondary comparison: **BBLR versus local quadratic (LP2)**. This tests whether a BBLR gain survives comparison with a conventional higher-order local polynomial.

## Phase C — separate the two bandwidths

Evaluate a two-dimensional `(r, eps)` surface rather than imposing `r=eps`.

- `r` controls the bias/variance of each local linear slope.
- `eps` controls smoothing of the estimated slope surface.

For each DGP, report the slope-RMSE surface and function-RMSE surface separately. A key empirical question is whether their minima occur at different `(r, eps)` combinations.

A useful parameterisation for later production is to include `eps/r` ratios approximately `{0.25, 0.5, 1, 1.5, 2}` while retaining several absolute `r` values.

## Phase D — support geometry

Repeat the smooth DGPs under:

- uniform support;
- centre-dense normal support;
- skewed support;
- gapped/two-cluster support.

Report interior and boundary slope RMSE separately. Also report Stage-1 and Stage-2 fallback frequencies. Sparse-support results should never be interpreted without those diagnostics.

The `gap` design is intentionally hostile. Failure there is informative if it identifies when Stage-2 averaging crosses regions with insufficient support.

## Phase E — non-smooth stress tests

U09-U12 assess kinks and discontinuities. They are **stress tests, not conventional differentiable cases**.

Derivative RMSE must exclude a guard band around each kink/jump. Run guard-width sensitivity (for example 0.05, 0.10 and 0.20 on support [-2,2]). Also report the fitted slope surface near the break because oversmoothing across a discontinuity is substantively important even when excluded from derivative RMSE.

## Phase F — sample size and noise

After the parameter surfaces are understood, vary:

- `n = 250, 500, 1000` initially;
- noise SD `= 0.10, 0.25, 0.50` initially.

The important pattern is whether Stage-2 smoothing is most helpful in the expected high-variance regime: smaller samples and/or larger noise, provided curvature is not so rapid that the extra smoothing dominates through bias.

## Phase G — the Stage-2 minimum-centre rule

Do not automatically set the Stage-2 minimum equal to the Stage-1 minimum.

Stage 1 requires enough observations to fit a local line. Stage 2 is averaging already-estimated slopes and mathematically needs only one centre to return an estimate. Requiring 10/15/etc. centres creates an additional adaptive nearest-neighbour smoother in sparse areas.

Compare `min_stage2 = 1, 5, 15` explicitly. Record fallback shares. This is a BBLR design parameter, not a harmless implementation detail.

## Core manuscript outputs eventually required

1. A small table confirming exact/noiseless linear recovery.
2. A figure showing representative true and estimated derivative curves for BBLR, LLR and LP2.
3. A heatmap or contour representation of slope RMSE over `(r,eps)` for selected smooth DGPs.
4. A summary of BBLR's Stage-2 gain/loss versus same-`r` LLR across DGPs, support designs, sample sizes and noise levels.
5. A comparison of derivative-optimal and function-optimal tuning choices.
6. Boundary versus interior performance.
7. Separate stress-test treatment of kinks/jumps with guard-band sensitivity.

## Interpretation discipline

A result in which BBLR loses on prediction but wins on derivative recovery is not contradictory if the paper's estimand is the conditional mean gradient. Conversely, a derivative advantage that appears only because competitors were tuned for prediction is not sufficient. Comparator tuning and scoring must ultimately be aligned with the estimand being evaluated.

The revised univariate analysis should therefore distinguish **estimator capacity** (oracle parameter surfaces) from **feasible tuning** (later validation/CV rules). The current oracle outputs are diagnostic, not a proposed empirical tuning procedure.
