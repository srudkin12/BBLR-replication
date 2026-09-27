# Scientific predeclaration — targeted local-sparse comparator audit

## Purpose

This audit addresses one specific Journal of Econometrics referee risk: whether BBLR's active-versus-nuisance coefficient-field frontier survives comparison with a **local estimator that explicitly imposes sparsity**, rather than only with global OLS/ridge/Lasso or unpenalised local smoothers.

It is not a new operating-map programme. The audit freezes nine representative cells from the already completed `BBLR_AlphaIrrelevant_DimensionProduction_v1_0` design and reuses their DGPs, sample sizes, CV folds/repeats, and deterministic base seeds.

## Local sparse benchmark

The benchmark is `LL_ALASSO`: a hard-ball **local-linear adaptive Lasso**.

At query `q`, it fits

`y_i ~= a(q) + b(q)'(X_i-q)`

inside the same radius-based local neighbourhood used by LLR. The intercept is not penalised. Local slope coordinates are centred and standardised before penalisation. Adaptive weights are constructed from the unpenalised local-linear fit in the standardised coordinates. The penalised objective is

`0.5/n_q ||y_c - Z_s gamma||^2 + lambda sum_j w_j |gamma_j|`.

The reported slope is transformed back to the original covariate scale. Penalty strength is parameterised as a fraction of the query-specific weighted `lambda_max`. `lambda_frac=0` exactly recovers unpenalised LLR under the same neighbourhood safeguard.

This comparator is inspired by the local-polynomial + adaptive-Lasso varying-coefficient literature, but it is implemented here in the BBLR simulation geometry rather than claimed as an exact reproduction of any one published estimator.

## Tuning

All feasible methods are tuned by prediction of held-out outcomes only.

- BBLR: same radius grid and `epsilon/r in {0, 0.5, 1, 1.5, 2}` as the completed moderate-dimensional production study.
- LLR: same radius grid.
- LL_ALASSO: same radius grid jointly with `lambda_frac in {0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.70}`.
- Global Lasso: same global penalty grid as the completed production study.
- OLS: untuned descriptive benchmark.

The local sparse method is not given the active set, the number of active variables, derivative truth, or support truth.

## Frozen cells

Nine cells are predeclared before the audit results are observed:

1. linear independent-nuisance negative control (`d=5`);
2. pure-null sparse negative control (`d=8`);
3. smooth independent nuisance (`d=5`);
4. sharp transition independent nuisance (`d=5`);
5. high-frequency negative control (`d=8`);
6. smooth correlated nuisance (`d=8`);
7. sharp transition correlated nuisance (`d=8`);
8. sharp transition correlated nuisance (`d=12`);
9. sharp transition with aligned thin support (`d=8`).

The exact rows and original production cell IDs are in `AUDIT_CELL_MAP.md` and `data/AUDIT_CELLS.csv`.

## Replications

Production target: **500 replications per cell = 4,500 cell-replications**.

The audit is intentionally much smaller than the 82,000-replication production experiment because it addresses one comparator question rather than remapping the estimator's operating region.

## Primary comparison

Primary comparison: **BBLR vs LL_ALASSO** at independently outcome-CV-selected tuning values.

Primary metrics:

- active-gradient vector RMSE;
- nuisance-gradient vector RMSE divided by `sqrt(d_nuisance)`;
- tangent-intercept RMSE;
- conditional-mean RMSE.

Secondary metrics:

- reference-state alpha absolute error;
- raw nuisance-vector RMSE;
- exact nuisance-zero share for the sparse comparator;
- active-coordinate zero share;
- scale-specific joint-coordinate RMSE.

The joint-coordinate norm remains explicitly scale-specific and is not treated as an invariant affine-field loss.

## Interpretation rule

No universal-dominance criterion is predeclared.

A scientifically useful BBLR result would be a frontier: LL_ALASSO may produce substantially sparser nuisance slopes, while BBLR may retain lower active-gradient and tangent-intercept risk in nonlinear settings. Linear and null controls are expected to favour sparse/global methods on zero coefficients. High-frequency cases remain adverse controls. If LL_ALASSO dominates BBLR on both active and nuisance targets in the nonlinear cells, that result must be reported directly.

## What this audit cannot establish

This audit does not establish a sparsity oracle property for BBLR, does not make BBLR a variable-selection estimator, and does not exhaust the varying-coefficient selection literature. It asks a narrow robustness question using a local sparse benchmark constructed under the same feasible outcome-CV principle.
