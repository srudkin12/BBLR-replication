# Electricity-data redistribution audit

## Status

**AUTHOR REVIEW REQUIRED before making the repository public.**

The constructor does not infer data-redistribution rights from the fact that a dataset is distributed by an R package or available from an upstream public source.

Paper 1 identifies the illustration as the 2014 Victorian electricity-demand series distributed with `fpp2`, with underlying Australian Energy Market Operator demand data and Australian Bureau of Meteorology Melbourne temperature data.

Likely data files currently present in the release candidate: **92**.

See `ELECTRICITY_FILE_INVENTORY.csv` for exact paths and hashes, and `ELECTRICITY_PROVENANCE_HITS.csv` for source/licensing language found in retained code and documentation.

## Release rule

If redistribution permission is confirmed, retain the required empirical data and document source/licence terms explicitly.

If redistribution permission is not confirmed, remove raw/source data from the public repository while retaining code, frozen derived selections that are legally redistributable, and precise instructions for obtaining/reconstructing the source series.

No automatic deletion has been performed.