# BBLR v1.3 production run

Project root:

`/media/simon/E490B01790AFEE6C/BBLR2`

## Verify, unpack and test

After manually placing the ZIP and SHA in the project root:

```bash
cd /media/simon/E490B01790AFEE6C/BBLR2

sha256sum -c BBLR_Univariate_SupportRegime_v1_3.zip.sha256
unzip -q BBLR_Univariate_SupportRegime_v1_3.zip

cd BBLR_Univariate_SupportRegime_v1_3

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_univariate.txt

python -m pytest -q
```

Expected: `22 passed`.

## Production geometry map

```bash
cd /media/simon/E490B01790AFEE6C/BBLR2/BBLR_Univariate_SupportRegime_v1_3

source .venv/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs

python scripts/run_univariate_hpc.py \
  --n-grid 500 \
  --noise-grid 0.25 \
  --reps 200 \
  --n-test 2000 \
  --jobs 140 \
  --chunk-reps 25 \
  --resume \
  --output-dir results_hpc/support_regime_v1 \
  2>&1 | tee logs/support_regime_v1.log
```

Expected:

- 25 DGPs;
- 11 support designs;
- 275 cells;
- 55,000 replications;
- 2,200 shards.

## Summarise and analyse

```bash
python scripts/summarize_univariate_hpc.py \
  --run-dir results_hpc/support_regime_v1

python scripts/analyze_support_regime.py \
  --run-dir results_hpc/support_regime_v1

cat results_hpc/support_regime_v1/COMPLETION_AUDIT.json
```

Expected completion status: `PASS`.

## Compact analysis handback

Do not return the raw shard directory unless specifically requested.

```bash
python scripts/make_analysis_handoff.py \
  --run-dir results_hpc/support_regime_v1
```

Return only the resulting analysis-handback ZIP and matching `.sha256` file from:

`results_hpc/handbacks/`

The raw shards, configuration summary and per-replication oracle files should
remain on the Linux source machine.
