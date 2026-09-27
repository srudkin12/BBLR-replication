# BBLR electricity peak / fixed-evening extension — frozen design

## Purpose

This is a pre-specified extension of the completed 2014 Victorian daily-demand application. It changes the response variable, not the estimator architecture or comparator set.

Two outcomes are analysed symmetrically from the canonical half-hourly `fpp2::elecdemand` series:

1. **Daily peak demand**: the maximum of the 48 half-hourly demand observations within each calendar day.
2. **Fixed-evening demand**: the observed half-hourly demand at **18:00 local clock time** on each calendar day.

For both outcomes, the covariate is **daily maximum Melbourne temperature**, computed as the maximum of the 48 half-hourly temperature observations within the same calendar day. The sample is restricted to `WorkDay == 1`, exactly as in the completed daily-demand exercise.

No other outcome definition, clock time, moving window, temperature transformation, or sample restriction is searched over in this package.

## Validation and comparators

The package uses the same method family and tuning grids as the v1.1 production audit:

- BBLR with hard-radius Stage 1 and exact `epsilon = 0` LLR branch;
- separately labelled LLR zero branch;
- LP2/local quadratic;
- penalised cubic B-spline;
- heating/cooling hinge regression.

For each outcome, defaults are:

- 1,000 repeated 5-fold random-CV partitions;
- 1,000 repeated 5-fold calendar-week-grouped partitions;
- chronological five-fold audit;
- 1,000 calendar-week block-bootstrap replications for BBLR/LLR derivative uncertainty;
- 5--95% support as the main estimation grid;
- 2--98% support as a support/safeguard audit.

The exact same resampling partitions are used for the peak and 18:00 outcomes.

## Data-integrity checks

The half-hourly source must contain exactly 17,520 observations (365 x 48), beginning at 2014-01-01 00:00 and advancing in 30-minute increments. The run stops unless:

- every calendar day has exactly 48 half-hourly observations;
- `WorkDay` is constant within each day;
- each day has exactly one 18:00 observation;
- derived daily maximum temperature matches the bundled canonical `fpp2::elecdaily` maximum-temperature series to numerical tolerance;
- derived daily `WorkDay` matches the canonical daily series exactly.

## Interpretation rule

The objective is not to find a specification in which BBLR wins. The extension asks whether changing from daily aggregate demand to economically sharper peak/fixed-evening demand moves the observed problem into a positive-Stage-2 operating region. If both outcomes select `epsilon = 0`, the electricity application is treated as an empirical negative control and no further electricity-response search is warranted for Paper 1.
