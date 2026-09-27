# Scientific pre-declaration: stable-arithmetic matched-H-lambda audit v1.1

## Question
The v1.0.1 matched-kernel audit found substantial finite-sample differences between BBLR and the
one-stage matched `W_lambda` estimator, including very large one-stage risks in some `lambda>1`
replications.  Are those differences genuine properties of the finite-sample matched estimator, or are
they artifacts of solving weighted local-linear normal equations by a cancellation-prone formula?

## Frozen design
Use all 92 original operating-map cells and all 1,000 replications per cell.  Do not rerun CV.  Use the
already selected BBLR `(r_hat_B,eps_hat_B)` and `lambda_hat=eps_hat_B/r_hat_B`.

## Two algebraically equivalent matched estimators
At every query location evaluate:

1. `W_normal`: the v1.0.1 safeguard matched estimator using the original normal-equation slope formula;
2. `W_centered`: exactly the same `W_lambda` weights, support, 15-positive-weight rule, singularity rule,
   and 15-NN fallback, but compute an accepted weighted slope as

   `sum w(z-zbar)(y-ybar) / sum w(z-zbar)^2`.

No bandwidth, lambda, kernel, safeguard threshold, or tuning rule changes.  The estimators are
mathematically identical; only floating-point arithmetic differs.

## Interpretation
- If the large v1.0.1 risks disappear under `W_centered`, the mechanism result was contaminated by
  numerical cancellation and must be revised.
- If `W_centered` reproduces the same risk pattern, including the large `lambda>1` failures and the
  `lambda=1` / small-`|Gamma|` BBLR advantage, those are genuine finite-sample properties of the
  matched one-stage estimator rather than normal-equation roundoff.

## Integrity
- 92,000 rows, 92 cells, 1,000 replications per cell;
- BBLR archive reproduction within 1e-9;
- `W_normal` positive-Stage-2 test-support cell classifications reproduce v1.0.1:
  73 BBLR-better, 14 W-better, 5 unclear;
- fallback masks for `W_normal` and `W_centered` must agree except for impossible-denominator emergency
  fallback in the centred implementation, which is separately counted.
