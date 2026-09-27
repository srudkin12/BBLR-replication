# BBLR Noise-Resolution and Replication-Uncertainty Audit v1.0.1

Targeted extension for BBLR Paper 1. It addresses two presentation/evidence issues only:

1. the high-frequency noise mechanism is currently represented by three noise values; this package uses nine values, `0.10` to `0.50` by `0.05`;
2. the paper often plots one cell summary despite 1,000 independent Monte Carlo replications; this package retains and reports the replication distribution.

## Frozen scientific design

- high-frequency smooth DGP on `[-2,2]`: `m(x)=x+sin(2*pi*x)/(2*pi)`, `g(x)=1+cos(2*pi*x)`;
- Gaussian additive noise with standard deviation sigma;
- `Ndev=1000`, `Ntest=3000`, fixed grid of 401 support locations;
- 1,000 independent replications per noise level;
- five repetitions of five-fold outcome CV;
- production 15-neighbour safeguard for BBLR/LLR;
- BBLR `r` and `epsilon` grids, adaptive-k grids, and spline-lambda range copied from the manuscript-frozen strong-comparator specification;
- methods used in the revised main noise figure: BBLR, separately tuned LLR, AKLLR, AKLP2, smoothing spline.

The componentwise comparators are intentionally not rerun here: support is connected uniform `[-2,2]`, and the purpose of this audit is the high-frequency noise mechanism rather than gap segmentation. Existing componentwise evidence remains elsewhere in the paper.

All nine noise levels are rerun. The old `0.10`, `0.25`, and `0.50` levels are continuity sentinels, not new claims. `tables/anchor_continuity_check.csv` compares selected published legacy summaries to the new medians. Do not replace the manuscript figure if those anchors indicate a material specification mismatch; inspect first.

## Reporting principle

`noise_replication_level.csv` keeps one row per method per replication. Summary tables distinguish:

- mean MSE/RMSE-style quantities;
- median replication RMSE;
- q2.5/q10/q25/q75/q90/q97.5 replication variation, including a central 95% empirical replication interval;
- Monte Carlo SE and normal-approximation 95% confidence interval for the mean;
- bootstrap confidence interval for the median;
- paired BBLR/comparator RMSE ratios and BBLR win share.

The q2.5--q97.5 interval is a **central 95% empirical replication interval**: it describes between-dataset variation, not uncertainty about the mean or median. Separate Monte Carlo confidence intervals are reported for the mean (mean ± 1.96 MCSE) and the median (bootstrap percentile CI). The main uncertainty figure uses the 25--75% band inside the 2.5--97.5% replication band; q10 and q90 remain in the summary tables for robustness and alternative plotting.

## Linux run

From `/media/simon/E490B01790AFEE6C/BBLR2` after checksum/unzip:

```bash
cd BBLR_Noise_Uncertainty_Audit_v1_0_1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/preflight.py
python scripts/smoke_test.py
python scripts/run_noise_uncertainty.py --workers 80 --reps 1000 --batch-size 5 --legacy-root ..
```

The production run is restartable. Completed batch CSVs are detected and skipped.

## Handback

Upload:

- `BBLR_Noise_Uncertainty_Audit_v1_0_1_HANDback.zip`
- `BBLR_Noise_Uncertainty_Audit_v1_0_1_HANDback.zip.sha256`
