# Changes in BBLR Univariate Support–Regime v1.3

- Adds 25 controlled smooth transition DGPs crossing transition width and location.
- Adds six hard-gap support designs and five soft central-depletion designs.
- Adds support-design metadata to every result row.
- Adds AKLLR: adaptive kNN tricube local-linear regression.
- Adds AKLP2: adaptive kNN tricube local-quadratic regression.
- Adds k-grid outcome-based tuning for adaptive comparators.
- Adds transition-local, away-from-transition, left-regime and right-regime slope RMSE.
- Replaces the phase-map analysis path that dropped NaN metadata rows.
- Adds robust selected-method summaries (mean, median, 10th/90th percentiles).
- Adds replication-level BBLR/comparator win and dominance probabilities.
- Adds explicit support/transition alignment variables.
- Adds method- and BBLR-specific tuning-boundary audits.
- Adds a smaller routine analysis handback that excludes raw shards, full
  configuration summaries and per-replication oracle files.
- Adds seven support/adaptive-estimator tests; full suite now passes 22/22.
