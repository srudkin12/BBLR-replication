# v1.0.1 change log

Scientific design is unchanged from v1.0. The reporting layer is expanded before production begins.

Changes:

- adds q2.5 and q97.5 for every replication-level method metric and every paired BBLR/comparator ratio;
- labels q2.5--q97.5 as the central 95% empirical replication interval, not a confidence interval for a population parameter;
- retains q10/q90 and q25/q75 summaries;
- adds mean Monte Carlo 95% confidence limits as mean +/- 1.96 MCSE;
- retains the bootstrap percentile 95% confidence interval for the median;
- revises the main paired-ratio uncertainty figure to show q2.5--q97.5 as the outer band and q25--q75 as the inner band;
- updates handback filenames to v1_0_1.

No DGP, tuning grid, estimator, safeguard, seed rule, replication count, comparator set, or CV rule changed.
