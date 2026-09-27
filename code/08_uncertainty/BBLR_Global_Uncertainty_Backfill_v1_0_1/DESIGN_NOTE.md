# BBLR Global Uncertainty Backfill v1.0.1 — Design Note

## Purpose

This package inventories the existing BBLR2 simulation archive and reconstructs referee-ready uncertainty summaries from replication-level rows wherever those rows can be recovered. It does **not** launch or alter any Monte Carlo design.

The evidence base is already large and scientifically frozen: 92,000 operating-map replications, 27,000 strong-comparator replications, 101,000 bivariate replications, 92,000 matched-H-lambda replications, the 220,000-replication legacy coefficient-field replay, 82,000 higher-dimensional replications, and the 4,500-replication local-sparse audit. The first task is therefore recovery and standardisation, not new simulation.

## Uncertainty schema

For every unambiguous method/cell/metric distribution the backfill reports:

- n, mean, SD, MCSE;
- mean ± 1.96 MCSE (95% Monte Carlo CI for the mean);
- q2.5, q10, q25, median, q75, q90, q97.5;
- central q2.5--q97.5 empirical replication interval;
- distribution-free 95% order-statistic CI for the population median.

For every unambiguous paired BBLR/comparator distribution it reports the same summaries for:

- BBLR minus comparator difference;
- BBLR/comparator ratio when the metric is finite and non-negative and comparator > 0;
- BBLR win share;
- classification from the paired mean-difference CI and median-ratio CI.

The replication interval and the Monte Carlo confidence intervals answer different questions and must not be conflated.

## Conservative schema rule

No paired result is created unless the source contains an unambiguous common study/cell/replication/method/metric identity. Conflicting duplicates are written to `conflicts_<study>.csv`; ambiguous schemas enter `manual_review_queue.csv`.

## Expected archive families

The registry includes the operating map, strong comparator suite, three bivariate suites, matched-H-lambda and stable-arithmetic audits, legacy coefficient-field replay, higher-dimensional production, local-sparse audit, and the new nine-level noise audit. Exact large handback names documented in the BBLR4 handover include:

- `BBLR_LegacyCoefficientFieldReplay_v1_0_HANDback.zip`
- `BBLR_MatchedHLambda_Audit_v1_0_1_HANDback.zip`
- `BBLR_MatchedHLambda_StableArithmetic_v1_1_HANDback.zip`

The operating-map working-paper archive is also documented as containing the replication-level audit files used for the safeguard, local-risk, selected-radius, tail, CV-materiality and matched-radius analyses.

## Output philosophy

Raw simulation rows remain on the workstation. The handback contains only inventories, schemas, uncertainty summaries, coverage audits, conflict/review queues and the deterministic recovery/rerun queue.
