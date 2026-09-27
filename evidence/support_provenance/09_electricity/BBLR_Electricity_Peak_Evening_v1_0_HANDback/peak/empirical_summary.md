# Daily peak demand empirical extension

Workday sample: n=251; temperature 9.80 to 43.20 °C.

## Random repeated CV
- BBLR: parameter (2.5, 2.5); RMSE 0.379589.
- LLR: parameter 2.5; RMSE 0.381275.
- LP2: parameter 6.0; RMSE 0.378971.
- PSpline: parameter (5, 0.1); RMSE 0.37684.
- Hinge: parameter (20.0, 26.0); RMSE 0.386066.

BBLR positive-Stage-2 repeat-selection share: **0.859**.
Fixed-library best-positive minus best-zero MSE: -0.00128264 (SE 7.0099e-05); positive win share 0.760.

## Calendar-week-grouped repeated CV
- BBLR: parameter (3.0, 0.0); RMSE 0.404377.
- LLR: parameter 3.0; RMSE 0.404377.
- LP2: parameter 8.0; RMSE 0.679197.
- PSpline: parameter (5, 0.1); RMSE 0.389214.
- Hinge: parameter (20.0, 28.0); RMSE 0.393194.

BBLR positive-Stage-2 repeat-selection share: **0.514**.
Fixed-library best-positive minus best-zero MSE: 0.00190774 (SE 0.000140992); positive win share 0.316.

## Chronological five-fold audit
- BBLR: mean fold RMSE 0.483484.
- Hinge: mean fold RMSE 0.470934.
- LLR: mean fold RMSE 0.475361.
- LP2: mean fold RMSE 0.601624.
- PSpline: mean fold RMSE 0.473854.

## Derivative diagnostics
- BBLR derivative total variation: 0.5022; LLR: 2.441.
- Lower-support-quartile derivative total variation: BBLR 0.169; LLR 1.147.
- Median week-block-bootstrap derivative-SD ratio BBLR/LLR: 0.634; lower-support quartile 0.612.
- Main 5--95% grid Stage-1 safeguard share: 0.000.

## Interpretation rule
No causal interpretation is attached. Positive Stage 2 is reported only if selected by the observed-outcome validation exercise; the package does not search over alternate peak definitions, clock times, or temperature summaries.
