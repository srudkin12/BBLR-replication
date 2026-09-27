# Electricity empirical production audit v1.1

Workday sample: n=251; temperature 9.80 to 43.20 °C.

## Random repeated CV
- BBLR: parameter (2.5, 0.0); RMSE 10.7932.
- LLR: parameter 2.5; RMSE 10.7932.
- LP2: parameter 6.0; RMSE 10.8611.
- PSpline: parameter (4, 0.001); RMSE 10.8509.
- Hinge: parameter (20.0, 28.0); RMSE 11.2051.

BBLR positive-Stage-2 selection share across repeat-level 5-fold selections: **0.630**.

## Calendar-week-grouped repeated CV
- BBLR: parameter (3.0, 0.0); RMSE 11.4656.
- LLR: parameter 3.0; RMSE 11.4656.
- LP2: parameter 6.0; RMSE 19.2087.
- PSpline: parameter (4, 0.001); RMSE 11.4434.
- Hinge: parameter (20.0, 30.0); RMSE 11.5609.

## Chronological five-fold audit
- BBLR: mean fold RMSE 14.2399.
- Hinge: mean fold RMSE 13.023.
- LLR: mean fold RMSE 14.2399.
- LP2: mean fold RMSE 15.284.
- PSpline: mean fold RMSE 13.8142.

## Derivative regularisation diagnostics
- BBLR derivative total variation: 63.17; LLR: 63.17; ratio 1.000.
- Lower-support-quartile derivative total variation: BBLR 28.35; LLR 28.35; ratio 1.000.
- Median week-block-bootstrap derivative-SD ratio BBLR/LLR across main grid: 1.000; lower-support quartile: 1.000.
- Main 5--95% grid Stage-1 safeguard share: 0.000.

## Interpretation
No causal interpretation is attached. Prediction dominance is not required. The empirical example is useful only if the enlarged resampling exercise shows that positive Stage 2 and derivative stabilisation are not artifacts of one set of random folds, while economically important changes in slope remain visible.
