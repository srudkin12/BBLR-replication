# Changes in BBLR univariate HPC v1.1

- Corrected the production tuning target: hyperparameters are selected using
  held-out observed `Y` RMSE rather than oracle RMSE against the known mean.
- Added independent train/validation/test samples per Monte Carlo replication.
- Retained simulation-only true-mean and true-slope metrics after tuning.
- Added complete outcome-RMSE / slope-RMSE tuning clouds and Pareto flags.
- Added DGP nonlinearity descriptors (slope dispersion, RMS curvature, slope
  total variation).
- Added exact prefix-sum `FastLocalLinear1D` and `FastBBLR1D`; equivalence to
  reference implementations is tested.
- Added tricube local-linear and smoothing-spline comparators.
- Added deterministic `SeedSequence` seeds with separate train/validation/test
  streams.
- Added chunked outer Monte Carlo parallelism, single-threaded BLAS policy,
  compressed per-chunk shards, resume support, expected-chunk manifest and
  completion audit.
- Added incremental HPC summary script.
- Test suite expanded to 11 passing tests.
