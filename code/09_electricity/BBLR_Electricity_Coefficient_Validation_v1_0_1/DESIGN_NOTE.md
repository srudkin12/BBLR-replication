# BBLR electricity coefficient-validation audit — frozen design

## Why this audit exists

The Monte Carlo analysis observes the true derivative `g(x)` by construction. After all methods are tuned using observed-outcome cross-validation, derivative performance can therefore be evaluated directly with test-support and fixed-grid derivative MSE. A surrogate derivative diagnostic would be strictly less informative in those experiments and is not needed.

The electricity application is different: the conditional-mean derivative is the scientific object of interest but is not observed. Outcome RMSE validates the conditional mean, not the local coefficient field. The coefficient audit therefore uses an observable implication of a differentiable conditional mean rather than pretending that derivative truth is available.

For two held-out observations with temperatures `x_i < x_j`,

`E[Y_j - Y_i | x_i, x_j] = m(x_j) - m(x_i) = integral_{x_i}^{x_j} g(t) dt`

under the maintained conditional-mean model. The observed secant `(Y_j-Y_i)/(x_j-x_i)` is noisy, but its noise is common to all methods evaluated on the same held-out pair. A derivative field that is closer to the latent conditional-mean derivative should, other things equal, predict these held-out local contrasts more accurately.

This is therefore an **empirical observability audit**, not a replacement risk criterion and not a new tuning rule. Its purpose is precisely to fill the gap that does not exist in Monte Carlo work, where derivative truth is known.

## Separation from estimator selection

All estimator configurations are frozen at the aggregate outcome-CV choices from the completed 1,000-repetition peak/fixed-evening production run. The audit does not retune BBLR or any comparator using coefficient-validation loss.

Fresh random and calendar-week-grouped resampling streams are used. Each method is refitted on the training observations and its coefficient field is evaluated only on held-out temperature contrasts. The resulting resampling variation is a stability diagnostic, not formal post-selection sampling inference.

The source handback used to freeze the configurations had SHA-256:

`3736384d2750dc7d7a4ac681ebe23dc8d21f82d129469b04eb8d32cf11ab4fd9`

## Outcomes and methods

The audit uses the same two pre-specified outcomes:

1. daily peak half-hourly demand;
2. demand at exactly 18:00.

Both use daily maximum temperature and the workday sample.

Methods are frozen BBLR, separately tuned LLR, LP2, penalised cubic spline, and the heating/cooling hinge model. For BBLR, a same-Stage-1-radius `epsilon=0` ablation is also evaluated whenever the selected BBLR configuration has positive Stage 2. This isolates the coefficient-field regularisation increment from Stage-1 radius choice.

## Held-out coefficient targets

Three temperature-separation bands are fixed before analysis:

- 0.5--1.0 degrees C: most local coefficient check;
- 1.0--2.0 degrees C: intermediate local contrast;
- 2.0--4.0 degrees C: broader shape/coherence check.

For every eligible held-out pair, the audit reports two derivative-facing losses:

1. **midpoint secant loss**: squared error between the observed held-out secant and the fitted derivative at the pair midpoint;
2. **Simpson integrated-slope loss**: squared error between the observed held-out secant and the Simpson-rule average of the fitted derivative at the two endpoints and midpoint.

The first is closest to a local coefficient check. The second allows derivative curvature over the wider bands while still evaluating the coefficient field rather than the fitted level directly.

Both are reported in coefficient units `(GW per degree C)^2`. Outcome-scale contrast loss is also retained as a secondary diagnostic.

## Weighting

Two summaries parallel the two simulation risk perspectives:

- **design-weighted**: all eligible held-out pairs receive equal weight;
- **equal-location**: pair losses are first averaged within ten fixed midpoint-temperature bins over the common 5--95% support and then averaged across non-empty bins.

No claim is made that these are identical to the simulation MSE targets. The parallel is conceptual: one weights the empirical design, the other gives locations more nearly equal representation.

## Dependence and uncertainty

The primary audit is repeated under:

- fresh repeated random five-fold splits;
- fresh repeated five-fold splits that hold out complete calendar weeks.

Paired method comparisons are summarised at the **repeat** level. Pair observations are never treated as independent units for standard-error calculations.

A separate calendar-week block bootstrap refits all five methods at the frozen random-CV configurations and reports:

- derivatives at 15, 18, 20, 22, 24, 28 and 30 degrees C;
- probability that the derivative is negative at 18 and 20 degrees C;
- probability that the derivative is positive at 24, 28 and 30 degrees C;
- first negative-to-positive crossing between 18 and 30 degrees C;
- number of derivative sign changes on the main support;
- derivative total variation.

This separates **stability** from **accuracy**. Smoothness or narrow bootstrap dispersion alone is not treated as evidence that a derivative curve is true; the held-out contrast loss supplies the accuracy-facing diagnostic.

## Interpretation rule

The audit may support statements such as:

- a derivative field predicts held-out local changes more accurately than another field;
- several structurally different estimators recover a common broad derivative shape;
- a method has lower derivative uncertainty or fewer unstable sign reversals.

It does **not** identify the true derivative, prove causal effects, or turn the empirical contrast loss into a new tuning objective. If rankings vary materially across separation bands or validation schemes, that ambiguity remains part of the empirical result.

## v1.0.1 implementation repair

No estimand, tuning value, validation band, seed rule, comparator, or interpretation rule changed. The revision is computational only: GUI plotting is isolated from worker processes, and completed outcome outputs may be reused after an interrupted run.
