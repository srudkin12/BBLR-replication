# BBLR Bivariate Four-Day Suite v1.0

Standalone bivariate Monte Carlo suite designed to transfer the univariate BBLR lessons into two dimensions.

## Main estimand

For `m(x1,x2)=E[Y|X=x]`, the primary simulation target is the gradient

`∇m(x) = (∂m/∂x1, ∂m/∂x2)`.

Every method is tuned using repeated-CV outcome RMSE only. True gradients are revealed only after tuning.

## Methods

- BBLR: two-stage ball-based local coefficient smoothing.
- LLR: fixed-radius local linear regression; exactly nested by BBLR at epsilon=0.
- LP2: fixed-radius local quadratic regression.
- AKLLR / AKLP2: adaptive k-nearest-neighbour local polynomial estimators.
- TSPLINE: tensor-product cubic spline with ridge smoothing.
- CTSPLINE: empirically inferred support-component tensor spline.
- OLS: global linear control.
- GQ2: global quadratic control.

## Design

58 cells cover:

- linear/polynomial controls;
- axis-aligned and diagonal interactions;
- low/high-frequency slope variation;
- broad/sharp marginal-effect transitions;
- local Gaussian features;
- multiple regimes and mixed anisotropy;
- correlated, skewed, clustered and near-manifold support;
- soft sparse bands, hard gaps, diagonal gaps and central holes;
- support/function alignment and mismatch;
- noise scaling and sample-size scaling.

The suite is resumable and writes independent compressed risk and selected-model shards.
