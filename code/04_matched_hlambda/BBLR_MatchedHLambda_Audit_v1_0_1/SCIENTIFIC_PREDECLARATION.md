# Scientific pre-declaration: matched-H-lambda audit v1.0.1

## Question
How much finite-sample derivative-risk behaviour remains different between BBLR and a faithful
one-stage local-linear estimator after matching BBLR's first-order composite derivative kernel
at the **already selected** BBLR scales?

## Frozen selection
No tuning is rerun. For each of the 92,000 validated operating-map replications, use the existing
10x5 repeated-outcome-CV BBLR selection `(r_hat_B, eps_hat_B)` and set
`lambda_hat = eps_hat_B / r_hat_B`. The one-stage matched estimator is not retuned.

## Scope
Use all 92 original operating-map cells and all 1,000 replications per cell. This choice removes
cell-selection discretion and preserves favourable, adverse, null, nonsmooth, non-uniform-density,
soft-depletion, hard-gap, noise-scaling, and sample-size-scaling regimes.

## Primary estimator comparison
For positive-Stage-2 replications, compare BBLR at `(r_hat_B, eps_hat_B)` with the **strict**
one-stage local-linear derivative estimator using the exact `W_lambda` from Proposition 3.7.
For `lambda>1`, the mathematically implied central zero region of `W_lambda` is retained literally.

When strict W is undefined at an evaluation location because its weighted local-linear design is
singular or empty, primary paired risks are computed on the common valid evaluation locations and
the strict-validity share is reported. This avoids comparing risks over different location sets.

## Secondary safeguard comparison
Also evaluate `W_lambda` with the same 15-nearest-observation unweighted local-linear fallback
used by production BBLR Stage 1, invoked only when the strict weighted fit has fewer than 15
positive-weight observations or is singular. This is a finite-sample support/safeguard diagnostic,
not the primary matched-kernel estimator.

## Risk targets
1. Independent test-support derivative MSE.
2. Equal-location fixed-grid derivative MSE.
3. Structurally-supported fixed-grid derivative MSE, excluding known zero-density interiors of
   hard-gap designs.
4. Central-window (`|x|<=0.4`), outer-window, and boundary diagnostics retained at replication level.

Hard-gap full-grid results are extrapolation diagnostics and are not interpreted as regular derivative
recovery.

## Theory-facing strata
Results will be stratified by:
- `eps=0` (identity control);
- `0<lambda<1`, `lambda=1`, `lambda>1`;
- `|Gamma_lambda|<0.01`, `0.01<=|Gamma_lambda|<0.05`, and `|Gamma_lambda|>=0.05`;
- operating-map panel and support design.

## Summary statistics
Report paired mean-MSE differences, bootstrap 95% intervals, MSE ratios, replication win shares,
median RMSE, and 90th/95th/99th MSE quantiles. Do not substitute median rankings for expected-risk
comparisons.

## Integrity conditions
- Regenerated BBLR test/grid RMSE must reproduce frozen archived values within `1e-9`.
- For `eps=0`, safeguard-matched `W_0`, matched-radius production LLR, and BBLR must agree within
  `1e-9`.
- Completion must contain exactly 92,000 rows, 92 cells, and 1,000 replications per cell.

## Interpretation rule
A BBLR advantage over strict matched W in locally flat-density or `lambda=1` regimes cannot be
attributed to the first-order density-gradient term. It points instead to finite-sample empirical-centre
normalisation, generated-coefficient dependence, support-boundary/higher-order behaviour, selection
interaction, or safeguard geometry. Near-equivalence is also informative and would support a more
modest architectural/composite-regulariser positioning.
