# Production run

Project root expected on Linux:

`/media/simon/E490B01790AFEE6C/BBLR2/BBLR_Bivariate_FourDay_v1_0`

The unattended runner is:

```bash
bash scripts/run_four_day_suite.sh
```

It runs three stages sequentially and creates a compact handback after every completed stage.

## Stage 1: broad operating map

58 cells x 1,000 replications. General comparator set:

`BBLR, LLR, LP2, AKLLR, AKLP2, TSPLINE, OLS, GQ2`

## Stage 2: support-aware adversarial pass

19 targeted support cells x 1,000 replications. Adds `CTSPLINE`, an empirically inferred componentwise tensor spline.

## Stage 3: fine locality map

16 difficult cells x 1,500 replications with a much finer BBLR radius/epsilon grid. This starts only after Stages 1 and 2 are complete.

All stages are resumable. Raw shards stay on the Linux machine. Handbacks are written under `results_hpc/handbacks/`.
