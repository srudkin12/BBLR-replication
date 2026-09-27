# BBLR Univariate Support–Regime v1.3

This package is the next-stage BBLR univariate investigation.

Start with:

- `SUPPORT_REGIME_PLAN.md` for the statistical design;
- `SUPPORT_REGIME_RUN.md` for the Linux/HPC workflow;
- `CHANGES_SUPPORT_REGIME_v1_3.md` for the implementation delta.

The package preserves the corrected BBLR 1D implementation and prior
comparators, adds adaptive local-polynomial benchmarks, and focuses the DGP
experiment on the interaction between marginal-effect transitions and weak or
disconnected covariate support.

Routine handbacks should be created with:

`python scripts/make_analysis_handoff.py --run-dir <run-dir>`

Raw shards remain on the BBLR2 Linux machine.
