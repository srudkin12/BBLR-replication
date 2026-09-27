# Run instructions

Assume these files have been manually placed in:

`/media/simon/E490B01790AFEE6C/BBLR2`

## 1. Verify and unpack

```bash
cd /media/simon/E490B01790AFEE6C/BBLR2
sha256sum -c BBLR_LegacyCoefficientFieldReplay_v1_0.zip.sha256
unzip -o BBLR_LegacyCoefficientFieldReplay_v1_0.zip
cd BBLR_LegacyCoefficientFieldReplay_v1_0
```

## 2. Environment and unit tests

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

## 3. Cross-track preflight

```bash
rm -rf output/preflight
mkdir -p output/preflight

python scripts/run_replay.py --track uv92 --configs data/PREFLIGHT_UV92.csv.gz --output-dir output/preflight --jobs 4 --chunk-reps 1 --resume
python scripts/run_replay.py --track uv27 --configs data/PREFLIGHT_UV27.csv.gz --output-dir output/preflight --jobs 4 --chunk-reps 1 --resume
python scripts/run_replay.py --track biv101 --configs data/PREFLIGHT_BIV101.csv.gz --output-dir output/preflight --jobs 4 --chunk-reps 1 --resume
python scripts/summarize_replay.py --run-dir output/preflight
cat output/preflight/ANALYSIS_AUDIT.json
```

Expected status: `PASS` and archived-risk replay discrepancy below `1e-8`.

## 4. Full replay with 80 workers

```bash
mkdir -p logs
nohup env JOBS=80 OUT=output/production bash scripts/run_full_replay.sh \
  > logs/legacy_coefficient_field_replay.log 2>&1 &
echo $! | tee logs/legacy_coefficient_field_replay.pid
```

Monitor:

```bash
python scripts/monitor_replay.py --run-dir output/production
tail -n 50 logs/legacy_coefficient_field_replay.log
```

The script runs the three frozen replay tracks sequentially, each using 80 workers, then creates the analysis summaries.

## 5. Inspect analysis

```bash
cat output/production/ANALYSIS_AUDIT.json
cat output/production/AUTOMATED_SUMMARY.md
```

Key outputs:

- `LEGACY_ALPHA_METHOD_CELL_SUMMARY.csv`
- `MATCHED_RADIUS_ALPHA_SUMMARY.csv`
- `BBLR_ALPHA_PAIRWISE_CELL.csv`
- `LEGACY_TRACK_ALPHA_SUMMARY.csv`
- `REFERENCE_ALPHA_SUPPORT_SUMMARY.csv`
- `REPLAY_INTEGRITY_METRICS.csv`
- `LEGACY_COEFFICIENT_FIELD_REPLICATIONS.csv.gz`

## 6. Compact handback

```bash
python scripts/make_handback.py --run-dir output/production --out-dir .
```

Return:

- `BBLR_LegacyCoefficientFieldReplay_v1_0_HANDback.zip`
- `BBLR_LegacyCoefficientFieldReplay_v1_0_HANDback.zip.sha256`
