# Electricity data availability and local reconstruction

## Public-repository policy

Electricity observations used in the empirical illustration are **not distributed with this replication repository**.

The repository contains the analysis code, frozen tuning/selection records, compact derived results and provenance needed to identify the empirical exercise. Users who wish to rerun the electricity analysis should obtain the source series independently and reconstruct the local input files on their own machine.

Do not commit the reconstructed `source_data/*.csv` files back to this repository.

## Source series used in Paper 1

Paper 1 uses the 2014 Victorian electricity-demand series distributed with the R package `fpp2`.

The manuscript records the underlying provenance as:

- electricity demand: Australian Energy Market Operator (AEMO);
- Melbourne temperature: Australian Bureau of Meteorology;
- temperature site: Bureau of Meteorology site 086071;
- packaged teaching/research series: `fpp2::elecdemand`.

The retained coefficient-validation code contains the canonical source endpoint used during the production audit:

`https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/elecdemand.csv`

Users should verify the source and any applicable terms at the time they obtain the data. The repository does not redistribute a copy.

## Empirical sample used in the paper

Paper 1 restricts the analysis to workdays and uses daily maximum Melbourne temperature as the covariate.

Three outcomes are considered:

1. supplied daily aggregate electricity demand;
2. daily peak demand, defined as the maximum of the 48 half-hourly demand observations within a day;
3. electricity demand at exactly 18:00.

The workday sample reported in the manuscript contains 251 days. The observed daily maximum-temperature range is 9.8–43.2 degrees Celsius.

No alternative peak definition, clock time, moving window, temperature transformation or additional sample restriction is part of the frozen Paper 1 comparison.

## Recommended local preparation sequence

### A. Coefficient-validation package

Change to:

`code/09_electricity/BBLR_Electricity_Coefficient_Validation_v1_0_1/`

Create/activate the Python environment specified by that package and run:

```bash
python scripts/preflight.py
```

The retained preflight is designed to obtain the canonical `fpp2::elecdemand` half-hourly series if it is absent and validate the expected source schema before the empirical audit proceeds.

Any files created under that package's `source_data/` directory are local working inputs and are intentionally ignored by Git.

### B. Grouped-primary package

The grouped-primary analysis expects locally reconstructed workday files under:

`code/09_electricity/BBLR_Electricity_GroupedPrimary_v1_0/source_data/`

with the historical filenames:

- `peak_workdays.csv`
- `evening1800_workdays.csv`

Construct these from the same half-hourly `fpp2::elecdemand` series using the frozen definitions above:

- retain workdays only;
- use daily maximum Melbourne temperature as `Temperature`;
- define peak outcome as the maximum of the 48 half-hourly demands for that date;
- define the evening outcome using the observation at exactly 18:00;
- retain the calendar-date information required to construct the calendar-week grouping used by grouped CV.

Before running the grouped-primary analysis, read:

- `DESIGN_NOTE.md`
- `README.md`
- `frozen_grouped_selection.csv`
- `scripts/preflight.py`
- `scripts/run_grouped_primary.py`

Those files define the expected column names and frozen analysis configuration. The preflight must pass before production is run.

## Frozen analysis choices

Electricity inputs must not be used to retune or redefine the empirical exercise.

In particular:

- BBLR/LLR/comparator grids remain as frozen in the code;
- grouped calendar-week CV remains the primary dependence-aware validation;
- the peak outcome remains the daily maximum of the 48 half-hourly observations;
- the fixed-evening outcome remains exactly 18:00;
- frozen selections supplied in the repository are not re-estimated for the manuscript reproduction audit unless a full replication run is explicitly intended.

## What remains in the public repository

The repository intentionally retains:

- source code;
- package READMEs/design notes;
- frozen selection/configuration CSVs;
- compact result summaries;
- coefficient-validation summaries;
- manuscript-facing figures;
- provenance manifests and SHA-256 records.

These are computational results or configuration/provenance artefacts, not redistributed copies of the underlying electricity observations.

## Why the data are absent

Excluding the observations keeps the GitHub repository compact and avoids making the repository itself a distributor of third-party empirical data. Reproducibility is provided through precise source identification, retained preparation/analysis code, frozen definitions and provenance rather than by bundling the data.
