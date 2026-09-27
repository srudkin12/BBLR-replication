# BBLR Electricity Coefficient Validation v1.0.1

Standalone post-selection coefficient-field audit for the pre-specified Victorian peak-demand and 18:00-demand application.

## Linux workflow

Place this ZIP and SHA-256 file in:

`/media/simon/E490B01790AFEE6C/BBLR2`

Then:

```bash
cd /media/simon/E490B01790AFEE6C/BBLR2
sha256sum -c BBLR_Electricity_Coefficient_Validation_v1_0_1.zip.sha256
unzip -o BBLR_Electricity_Coefficient_Validation_v1_0_1.zip
cd BBLR_Electricity_Coefficient_Validation_v1_0_1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/preflight.py
python scripts/run_coefficient_validation.py --workers 32 --random-repeats 1000 --group-repeats 1000 --bootstrap 1000
```

The preflight downloads the canonical `fpp2::elecdemand` half-hourly data if it is not already present and reconstructs the same daily peak/18:00 workday samples as the production exercise.

## Expected handback

`BBLR_Electricity_Coefficient_Validation_v1_0_1_HANDback.zip`

and

`BBLR_Electricity_Coefficient_Validation_v1_0_1_HANDback.zip.sha256`

The handback contains compact summaries plus repeat-level coefficient-validation losses and all-method coefficient bootstrap diagnostics. It intentionally does not retune any method.

Before production, an optional implementation smoke test is:

```bash
python scripts/smoke_test.py
```

It should print `PASS_COEFFICIENT_VALIDATION_SMOKE`.

## v1.0.1 multiprocessing fix

This revision prevents Tcl/Tk crashes under ProcessPoolExecutor by forcing a non-interactive Matplotlib backend and delaying pyplot import until main-process figure creation. It also supports `--outcomes` and `--reuse-peak-dir` for recovery after an interrupted run.

## Recovery from the v1.0 Tcl/Tk interruption

If v1.0 completed `peak` and then aborted during `evening1800/random`, the completed peak outputs can be reused without recomputation:

```bash
python scripts/run_coefficient_validation.py \
  --workers 32 \
  --random-repeats 1000 \
  --group-repeats 1000 \
  --bootstrap 1000 \
  --outcomes evening1800 \
  --reuse-peak-dir ../BBLR_Electricity_Coefficient_Validation_v1_0/outputs/peak
```

The repair validates that the reused peak directory contains the complete summary/bootstrap/agreement output set before copying it. The interrupted evening folds were not checkpointed and are recomputed from the start.
