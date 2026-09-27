# Linux run instructions — full production v1.0

The user manually places the ZIP and `.sha256` file in:

`/media/simon/E490B01790AFEE6C/BBLR2`

## 1. Verify and unpack

```bash
cd /media/simon/E490B01790AFEE6C/BBLR2
sha256sum -c BBLR_AlphaIrrelevant_DimensionProduction_v1_0.zip.sha256
unzip -o BBLR_AlphaIrrelevant_DimensionProduction_v1_0.zip
cd BBLR_AlphaIrrelevant_DimensionProduction_v1_0
```

## 2. Environment and tests

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Expected: `9 passed`.

## 3. Reduced-size scientific preflight

The preflight exercises all important code branches, including d=12, null, correlated nuisance, all-active and shifted-support cells, but uses reduced n and 2x1 CV so that it is a code/integrity check rather than a miniature production run.

```bash
rm -rf output/preflight
python scripts/run_production.py \
  --cells data/PREFLIGHT_CELLS.csv \
  --output-dir output/preflight \
  --reps 1 \
  --jobs 8 \
  --chunk-reps 1 \
  --resume

python scripts/summarize_production.py \
  --run-dir output/preflight \
  --expected-reps 1

cat output/preflight/COMPLETION_AUDIT.json
cat output/preflight/MATCHED_RADIUS_IDENTITY_AUDIT.json
cat output/preflight/AUTOMATED_SUMMARY.md
```

`COMPLETION_AUDIT.json` should report `PASS`.

## 4. Full production: 82 cells x 1,000 replications, 80 workers

```bash
mkdir -p logs

nohup python scripts/run_production.py \
  --cells data/PRODUCTION_CELLS.csv \
  --output-dir output/production1000 \
  --reps 1000 \
  --jobs 80 \
  --chunk-reps 2 \
  --resume \
  > logs/production1000.log 2>&1 &

echo $! | tee logs/production1000.pid
```

## 5. Monitor

```bash
python scripts/monitor_production.py \
  --run-dir output/production1000 \
  --cells data/PRODUCTION_CELLS.csv \
  --reps 1000

tail -n 50 logs/production1000.log
```

The run is resumable. Re-running the same command with `--resume` skips completed shard files.

## 6. Summarise after completion

```bash
python scripts/summarize_production.py \
  --run-dir output/production1000 \
  --expected-reps 1000

cat output/production1000/COMPLETION_AUDIT.json
cat output/production1000/MATCHED_RADIUS_IDENTITY_AUDIT.json
cat output/production1000/AUTOMATED_SUMMARY.md
```

## 7. Compact handback

```bash
python scripts/make_handback.py \
  --run-dir output/production1000 \
  --out-dir .
```

Return:
- `BBLR_AlphaIrrelevant_DimensionProduction_v1_0_HANDback.zip`
- `BBLR_AlphaIrrelevant_DimensionProduction_v1_0_HANDback.zip.sha256`

Raw shards remain on the Linux workstation.
