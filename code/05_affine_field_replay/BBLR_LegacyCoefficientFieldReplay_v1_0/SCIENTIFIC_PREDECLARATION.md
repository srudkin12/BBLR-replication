# Scientific predeclaration: legacy coefficient-field replay

## Primary purpose

Measure intercept/coefficient-field performance in the already completed univariate and bivariate experiments without changing their outcome-based tuning decisions.

## Frozen design

No DGP, support design, sample size, noise level, seed, comparator set or selected hyperparameter from the historical experiments is changed.

The replay uses archived selected configurations from:

1. 92-cell repeated-CV univariate operating map;
2. 27-cell univariate strong-comparator confirmation;
3. 58-cell broad bivariate suite;
4. 19-cell support-aware bivariate suite;
5. 16-cell bivariate fine-locality suite.

## New estimands

At observed test locations:

- tangent intercept: `alpha(x)=m(x)-x'g(x)`;
- tangent-intercept RMSE and bias;
- supported-grid tangent-intercept RMSE.

At the reference state zero:

- `alpha0=m(0)`;
- absolute error, squared error and signed error;
- origin support classification and nearest-development-point distance.

## Matched-radius Stage-2 comparison

For every BBLR replication, additionally fit

`LLR(r_hat_B) = BBLR(r_hat_B,0)`

without retuning. This is a new diagnostic evaluated at the historical selected BBLR Stage-1 radius. It isolates the conditional Stage-2 increment for tangent-alpha/reference-alpha risk.

## Interpretation constraints

- Do not call tangent alpha a structural finance alpha without an identification/model argument.
- Do not interpret `alpha0=m(0)` as supported when zero lies in a hard gap/hole.
- Do not retune on alpha or derivative truth.
- Do not replace historical outcome-based selections.
- Preserve adverse cells and cases where Stage 2 worsens intercept risk.
- Use exact replay of archived risks as a necessary integrity check before interpreting the new alpha metrics.
