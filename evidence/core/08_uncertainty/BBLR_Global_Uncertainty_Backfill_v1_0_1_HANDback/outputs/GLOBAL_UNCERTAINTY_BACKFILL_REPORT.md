# BBLR global uncertainty backfill report

Project root scanned: `/media/simon/E490B01790AFEE6C/BBLR2`

## Interpretation

`q025`--`q975` are central 95% empirical replication intervals. They describe sampling variation across Monte Carlo replications.
`mean_ci_*` quantify Monte Carlo uncertainty for the mean using mean ± 1.96 MCSE. `median_ci_*` are distribution-free 95% order-statistic confidence intervals for the population median.
Paired BBLR/comparator summaries are created only from rows with unambiguous common study, cell, replication, method and metric identifiers.

## Coverage

| Study | Status | Cells recovered / expected | Replication n range | Ready sources | Review sources | Paired summaries |
|---|---:|---:|---:|---:|---:|---:|
| operating_map | READY | 92 / 92 | 1000--1000 | 3682 | 3681 | 10764 |
| strong_comparator | READY | 27 / 27 | 1000--1000 | 5401 | 5400 | 945 |
| bivariate_broad | READY | 58 / 58 | 1000--1000 | 11601 | 11600 | 2842 |
| bivariate_support_aware | READY | 19 / 19 | 1000--1000 | 9501 | 9501 | 665 |
| bivariate_fine_locality | READY | 16 / 16 | 1500--1500 | 12001 | 12000 | 336 |
| matched_hlambda | READY | 92 / 92 | 908--1000 | 1 | 0 | 2367 |
| matched_hlambda_stable | READY | 92 / 92 | 1000--1000 | 1 | 0 | 368 |
| legacy_coefficient_replay | READY | 212 / 212 | 1000--1500 | 2 | 0 | 19802 |
| higher_dimension | READY | 82 / 82 | 1000--1000 | 41000 | 0 | 15663 |
| local_sparse | READY | 9 / 9 | 500--500 | 4504 | 0 | 383 |
| noise_uncertainty | READY | 9 / 9 | 1000--1000 | 1802 | 0 | 108 |

## Files requiring manual schema review

- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000001_cell00_B01_uniform_rep00005-00009_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000000_cell00_B01_uniform_rep00000-00004_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000009_cell00_B01_uniform_rep00045-00049_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000002_cell00_B01_uniform_rep00010-00014_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000004_cell00_B01_uniform_rep00020-00024_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000003_cell00_B01_uniform_rep00015-00019_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000005_cell00_B01_uniform_rep00025-00029_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000010_cell00_B01_uniform_rep00050-00054_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000007_cell00_B01_uniform_rep00035-00039_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000006_cell00_B01_uniform_rep00030-00034_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000008_cell00_B01_uniform_rep00040-00044_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000065_cell00_B01_uniform_rep00325-00329_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000066_cell00_B01_uniform_rep00330-00334_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000067_cell00_B01_uniform_rep00335-00339_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000068_cell00_B01_uniform_rep00340-00344_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000069_cell00_B01_uniform_rep00345-00349_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000070_cell00_B01_uniform_rep00350-00354_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000071_cell00_B01_uniform_rep00355-00359_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000072_cell00_B01_uniform_rep00360-00364_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000073_cell00_B01_uniform_rep00365-00369_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000074_cell00_B01_uniform_rep00370-00374_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000023_cell00_B01_uniform_rep00115-00119_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000075_cell00_B01_uniform_rep00375-00379_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000076_cell00_B01_uniform_rep00380-00384_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000024_cell00_B01_uniform_rep00120-00124_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000025_cell00_B01_uniform_rep00125-00129_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000029_cell00_B01_uniform_rep00145-00149_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000027_cell00_B01_uniform_rep00135-00139_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000026_cell00_B01_uniform_rep00130-00134_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000030_cell00_B01_uniform_rep00150-00154_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000028_cell00_B01_uniform_rep00140-00144_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000031_cell00_B01_uniform_rep00155-00159_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000032_cell00_B01_uniform_rep00160-00164_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000033_cell00_B01_uniform_rep00165-00169_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000011_cell00_B01_uniform_rep00055-00059_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000022_cell00_B01_uniform_rep00110-00114_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000050_cell00_B01_uniform_rep00250-00254_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000077_cell00_B01_uniform_rep00385-00389_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000064_cell00_B01_uniform_rep00320-00324_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000086_cell00_B01_uniform_rep00430-00434_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000122_cell00_B01_uniform_rep00610-00614_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000104_cell00_B01_uniform_rep00520-00524_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000105_cell00_B01_uniform_rep00525-00529_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000106_cell00_B01_uniform_rep00530-00534_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000108_cell00_B01_uniform_rep00540-00544_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000107_cell00_B01_uniform_rep00535-00539_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000110_cell00_B01_uniform_rep00550-00554_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000109_cell00_B01_uniform_rep00545-00549_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000113_cell00_B01_uniform_rep00565-00569_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000112_cell00_B01_uniform_rep00560-00564_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000111_cell00_B01_uniform_rep00555-00559_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000116_cell00_B01_uniform_rep00580-00584_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000114_cell00_B01_uniform_rep00570-00574_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000118_cell00_B01_uniform_rep00590-00594_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000119_cell00_B01_uniform_rep00595-00599_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000117_cell00_B01_uniform_rep00585-00589_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000115_cell00_B01_uniform_rep00575-00579_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000120_cell00_B01_uniform_rep00600-00604_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000051_cell00_B01_uniform_rep00255-00259_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000052_cell00_B01_uniform_rep00260-00264_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000121_cell00_B01_uniform_rep00605-00609_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000054_cell00_B01_uniform_rep00270-00274_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000058_cell00_B01_uniform_rep00290-00294_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000053_cell00_B01_uniform_rep00265-00269_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000056_cell00_B01_uniform_rep00280-00284_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000055_cell00_B01_uniform_rep00275-00279_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000057_cell00_B01_uniform_rep00285-00289_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000060_cell00_B01_uniform_rep00300-00304_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000059_cell00_B01_uniform_rep00295-00299_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000062_cell00_B01_uniform_rep00310-00314_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000063_cell00_B01_uniform_rep00315-00319_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000061_cell00_B01_uniform_rep00305-00309_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000087_cell00_B01_uniform_rep00435-00439_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000089_cell00_B01_uniform_rep00445-00449_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000088_cell00_B01_uniform_rep00440-00444_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000090_cell00_B01_uniform_rep00450-00454_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000092_cell00_B01_uniform_rep00460-00464_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000095_cell00_B01_uniform_rep00475-00479_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000093_cell00_B01_uniform_rep00465-00469_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000098_cell00_B01_uniform_rep00490-00494_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000096_cell00_B01_uniform_rep00480-00484_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000094_cell00_B01_uniform_rep00470-00474_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000097_cell00_B01_uniform_rep00485-00489_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000101_cell00_B01_uniform_rep00505-00509_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000099_cell00_B01_uniform_rep00495-00499_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000100_cell00_B01_uniform_rep00500-00504_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000091_cell00_B01_uniform_rep00455-00459_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000123_cell00_B01_uniform_rep00615-00619_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000102_cell00_B01_uniform_rep00510-00514_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000128_cell00_B01_uniform_rep00640-00644_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000129_cell00_B01_uniform_rep00645-00649_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000103_cell00_B01_uniform_rep00515-00519_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000124_cell00_B01_uniform_rep00620-00624_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000130_cell00_B01_uniform_rep00650-00654_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000125_cell00_B01_uniform_rep00625-00629_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000126_cell00_B01_uniform_rep00630-00634_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000131_cell00_B01_uniform_rep00655-00659_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000013_cell00_B01_uniform_rep00065-00069_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000127_cell00_B01_uniform_rep00635-00639_risk.csv.gz` — duplicate_cell_rep_method
- `BBLR_Bivariate_FourDay_v1_0/results_hpc/bivariate_broad_v1/risk_shards/chunk_000012_cell00_B01_uniform_rep00060-00064_risk.csv.gz` — duplicate_cell_rep_method
- … 42082 additional review items in `schema_registry.csv`.
