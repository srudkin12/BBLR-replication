# BBLR Univariate Power v1.4 changes

- Defines exact `epsilon=0`: Stage 2 is disabled and BBLR equals LLR for the same Stage-1 radius.
- Adds validation-MSE standard errors and feasible one-SE/low-smoothing, 0.5%-tolerance and 1%-tolerance selection flags.
- Adds training-X-only support-component detection.
- Adds componentwise smoothing spline (`CSPLINE`).
- Adds componentwise adaptive local-linear (`CAKLLR`) and adaptive local-quadratic (`CAKLP2`) comparators.
- Gap-factor candidates (16, 20, 30, inf) are outcome-tuned and include `inf` (no split), so support-aware methods nest their global counterparts.
- Adds streaming intensive analysis of the existing v1.3 raw tuning clouds.
- Adds a non-Cartesian landmark runner and 12 frozen landmark cells.
- Adds support-aware and feasible-selector power summaries.
- Adds five new unit tests covering exact nesting and support-aware comparator nesting.
