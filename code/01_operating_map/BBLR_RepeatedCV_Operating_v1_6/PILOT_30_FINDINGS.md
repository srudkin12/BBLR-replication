# BBLR 1D diagnostic pilot — 30 replications

## Status

This is **development evidence only**, not manuscript evidence. The run used `n=300`, noise SD `0.25`, six representative DGPs (`U01`, `U02`, `U04`, `U08`, `U10`, `U11`), uniform and deliberately gapped support, four Stage-1 radii and four Stage-2 radii, with 30 replications. Hyperparameter comparisons below are oracle diagnostics based on known simulation truth.

All eight automated correctness tests pass before this run, including exact 1D radius membership and noiseless linear slope recovery.

## Main provisional pattern

The second BBLR smoothing stage is **not uniformly beneficial**, which is methodologically useful.

Under uniform support, the best BBLR slope RMSE was lower than the best single-stage local-linear (LLR) slope RMSE in five of the six representative DGPs and essentially tied on the sinusoid. Examples:

| DGP | LLR | BBLR |
|---|---:|---:|
| U01 linear | 0.129 | 0.090 |
| U02 quadratic | 0.188 | 0.163 |
| U04 sinusoid | 0.502 | 0.497 |
| U08 local bump | 0.294 | 0.280 |
| U10 slope kink | 0.248 | 0.229 |
| U11 level jump* | 0.686 | 0.564 |

`*` Gradient scoring excludes the guard band around the discontinuity.

Under gapped support, the pattern becomes sharply conditional. BBLR improved strongly for the linear case, slope kink and level jump, but lost to LLR for the quadratic, sinusoid and local-bump cases. The strongest failures occurred where the derivative changes rapidly in a region with little support. That is consistent with an extra smoothing stage trading variance reduction for bias.

## Same-r comparison

The cleanest diagnostic compares BBLR with LLR using the **same Stage-1 radius** and lets only the Stage-2 radius vary.

Across the current grid, the best Stage-2 radius improved slope RMSE for almost every same-r comparison in the uniform linear, quadratic, kink and jump cases. Improvement was much less reliable for the sinusoid and local bump. Under gapped support, BBLR improved almost universally for the kink and jump but consistently worsened the sinusoid.

This is a better statement of what BBLR appears to do than a generic prediction comparison: coefficient smoothing can reduce variance, but its value depends on the spatial frequency of the true slope surface and the local support geometry.

## Function-optimal versus slope-optimal smoothing

Prediction/function and derivative objectives frequently selected different `(r, eps)` pairs. Under uniform support, the same pair was selected in only about 0–10% of replications for most DGPs in this small grid. Choosing the function-optimal pair imposed a positive derivative-RMSE penalty in every representative DGP on average.

This supports a central redesign of the paper: if BBLR is presented as an estimator of local coefficient/gradient surfaces, tuning solely for prediction MSE is generally misaligned with the stated estimand.

## Boundary behaviour

For uniform support, BBLR's oracle-selected slope surface generally improved both interior and boundary RMSE relative to LLR, although the gains were modest for rapidly oscillating functions. Under gapped support, boundary improvement can coexist with worse interior recovery for high-curvature DGPs. Consequently, aggregate RMSE alone is insufficient; support geometry and local neighbourhood counts need to accompany performance claims.

## Local quadratic comparator

A local-quadratic (`LP2`) comparator has now been added. In this pilot LP2 is not generally superior for derivative recovery, especially under gapped support where higher-order local fits can become unstable. However, LP2 sometimes beats both BBLR and LLR for the gapped sinusoid. BBLR therefore cannot be evaluated only against LLR; conventional higher-order local polynomials remain necessary comparators.

## Immediate implications

1. The repaired 1D code no longer supports the previous conclusion that BBLR simply performs badly in one dimension.
2. The central empirical object should be `m'(x)`, with `m(x)` prediction reported separately.
3. Stage-1 radius `r` and Stage-2 radius `eps` should remain separate design parameters.
4. Stage-2 minimum centre count should be treated as its own design parameter rather than inherited automatically from Stage 1.
5. Smooth high-frequency functions and sparse-support regions are likely to define BBLR's failure boundary.
6. Kinks/jumps need separate stress-test treatment with explicit exclusion bands for derivative scoring.
7. A larger Monte Carlo should be designed only after visual inspection of the parameter surfaces and representative slope curves.
