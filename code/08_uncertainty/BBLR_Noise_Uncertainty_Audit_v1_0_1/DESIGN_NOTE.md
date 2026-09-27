# Design note

This is a bounded evidence upgrade, not a reopening of the 92-cell operating map.

## Why nine noise points

The existing figure connects only three simulated values (`0.10`, `0.25`, `0.50`). The new frozen grid is `0.10:0.05:0.50`, which is dense enough to reveal curvature or a transition in the variance-bias trade-off without creating a large new DGP programme.

## Why rerun the three old anchors

Raw replication distributions are needed for the revised uncertainty graphic. Running all nine levels with one code path avoids mixing new replication-level results with old summary-only values and permits continuity checks at the three historical levels.

## Why five methods rather than all componentwise variants

Uniform support on `[-2,2]` contains no structural gap. BBLR/LLR, AKLLR, AKLP2, and spline span the mechanisms needed for the noise trade-off. Componentwise methods remain important for the hard-gap analysis but do not answer the present question.

## Evidential distinction

The main paired quantity is `BBLR derivative RMSE / comparator derivative RMSE` within the same replication. Its 2.5--97.5 percentiles define the central 95% empirical replication interval, with 25--75 and 10--90 percentiles retained as inner distribution summaries. These intervals describe replication heterogeneity. Separate Monte Carlo confidence intervals quantify precision of the estimated center: mean ± 1.96 MCSE for the mean and a bootstrap percentile 95% interval for the median. None of the replication intervals is interpreted as an inferential confidence band for an empirical population parameter.
