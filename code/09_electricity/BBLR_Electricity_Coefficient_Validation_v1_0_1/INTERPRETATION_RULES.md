# Frozen interpretation rules

## Primary accuracy-facing diagnostics

1. Simpson integrated-slope secant MSE, design-weighted.
2. Temperature separation bands 0.5--1.0 C and 1.0--2.0 C.
3. Calendar-week-grouped cross-fitting is the stronger dependence-aware check; random cross-fitting is reported alongside it.

## Secondary robustness diagnostics

- midpoint derivative/secant MSE;
- 2.0--4.0 C separation band;
- equal-location weighting over ten midpoint-temperature bins;
- outcome-scale local-contrast MSE.

## Stability diagnostics

Week-block bootstrap derivative dispersion, sign probabilities, heating-to-cooling crossing, number of sign changes, and total variation are reported separately from held-out coefficient accuracy. A smoother or more stable curve is not declared more accurate solely because it is smooth or stable.

## BBLR architecture diagnostic

Whenever the frozen BBLR configuration has positive Stage 2, the audit also evaluates `BBLR_MatchedZero`, which keeps the selected Stage-1 radius fixed and switches only `epsilon` to zero. This is the empirical coefficient-validation analogue of the matched-radius Stage-2 ablation used in the simulation analysis.

## Language threshold

A manuscript statement that an empirical coefficient field is **better supported** should require consistent local-contrast evidence across the two primary separation bands and should not depend solely on ordinary random folds when the week-grouped result points in the opposite direction.

The audit never licenses the phrase "true derivative" for an empirical curve. If BBLR, LP2, and the spline remain statistically/diagnostically difficult to distinguish, the manuscript should say that the data support a common broad coefficient shape while not identifying a unique best derivative estimator.
