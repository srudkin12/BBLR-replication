# BBLR Univariate Phase Map v1.2

## Purpose

This package extends the corrected 1D/HPC BBLR laboratory into a controlled
DGP phase-map experiment.

The central question is not simply whether BBLR performs well on a named
nonlinear function.  The experiment varies one feature of the conditional mean
geometry at a time and asks where prediction-selected second-stage coefficient
smoothing improves or harms recovery of the true marginal-effect surface.

## Controlled DGP families

The default phase map contains **37 DGPs**:

- `U01`: linear anchor.
- `Q01-Q06`: quadratic curvature amplitude ladder.
- `F01-F07`: slope-frequency ladder with slope amplitude held fixed.
- `A01-A06`: slope-amplitude ladder with frequency held fixed.
- `T01-T06`: smooth transition-width ladder with limiting slopes held fixed.
- `B01-B06`: local Gaussian feature width ladder with maximum slope
  perturbation held fixed.
- `L01-L05`: the same local feature translated through the support.

The DGP rows carry family and parameter metadata, and the runner also records
slope/curvature/roughness diagnostics that are never used for tuning.

## Tuning principle

All empirical tuning remains outcome-based.  Training data estimate the model;
an independent validation sample chooses each method's tuning parameters using
observed `Y`; an independent test sample reports outcome RMSE.  True mean and
true slope are revealed only afterwards for simulation evaluation.

## Comparator policy for this phase

The comparator set is deliberately held fixed relative to `smooth_map_v1`:

`OLS, LLR, LP2, KLLR, SPLINE, BBLR`

This makes changes in results attributable to DGP geometry rather than to a
simultaneous change in the benchmark set.  Adaptive-bandwidth local-polynomial
comparators should be added in a later targeted pass once interesting regions
of the phase map have been identified.

## Broadened tuning grids

The radius/epsilon grids now include near-global smoothing values up to the full
support width, preventing weakly nonlinear/linear cases from being artificially
truncated at the old upper bound.  The smoothing-spline lambda grid is likewise
expanded.

## Recommended exploratory production design

- n = 500
- support designs = uniform, normal, skewed, gap
- noise SD = 0.10, 0.25, 0.50
- 150 Monte Carlo replications
- 140 workers
- 25 replications per shard

This isolates DGP shape first.  Sample-size scaling should be studied in a
second pass on representative locations of the phase map rather than multiplying
the exploratory grid immediately.
