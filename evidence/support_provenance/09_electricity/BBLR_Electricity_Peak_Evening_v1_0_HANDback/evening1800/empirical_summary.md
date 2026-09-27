# 18:00 demand empirical extension

Workday sample: n=251; temperature 9.80 to 43.20 °C.

## Random repeated CV
- BBLR: parameter (2.5, 2.0); RMSE 0.452058.
- LLR: parameter 2.5; RMSE 0.454135.
- LP2: parameter 6.0; RMSE 0.452064.
- PSpline: parameter (5, 0.1); RMSE 0.445868.
- Hinge: parameter (20.0, 28.0); RMSE 0.448511.

BBLR positive-Stage-2 repeat-selection share: **0.921**.
Fixed-library best-positive minus best-zero MSE: -0.00188215 (SE 8.74052e-05); positive win share 0.814.

## Calendar-week-grouped repeated CV
- BBLR: parameter (5.0, 4.0); RMSE 0.463823.
- LLR: parameter 3.0; RMSE 0.463935.
- LP2: parameter 8.0; RMSE 0.860383.
- PSpline: parameter (4, 0.01); RMSE 0.45374.
- Hinge: parameter (20.0, 28.0); RMSE 0.454768.

BBLR positive-Stage-2 repeat-selection share: **0.703**.
Fixed-library best-positive minus best-zero MSE: -0.000104048 (SE 0.000152605); positive win share 0.596.

## Chronological five-fold audit
- BBLR: mean fold RMSE 0.572125.
- Hinge: mean fold RMSE 0.54184.
- LLR: mean fold RMSE 0.565295.
- LP2: mean fold RMSE 0.68757.
- PSpline: mean fold RMSE 0.566867.

## Derivative diagnostics
- BBLR derivative total variation: 0.6384; LLR: 2.699.
- Lower-support-quartile derivative total variation: BBLR 0.2803; LLR 1.209.
- Median week-block-bootstrap derivative-SD ratio BBLR/LLR: 0.674; lower-support quartile 0.683.
- Main 5--95% grid Stage-1 safeguard share: 0.000.

## Interpretation rule
No causal interpretation is attached. Positive Stage 2 is reported only if selected by the observed-outcome validation exercise; the package does not search over alternate peak definitions, clock times, or temperature summaries.
