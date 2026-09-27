# Run

```bash
python -m pytest -q
rm -rf output/preflight
python scripts/run_stable_arithmetic.py --configs data/PREFLIGHT_SELECTED_CONFIGS.csv --output-dir output/preflight --jobs 8 --chunk-rows 4 --resume
python scripts/analyze_stable_arithmetic.py --run-dir output/preflight --bootstrap-reps 200
cat output/preflight/ANALYSIS_AUDIT.json
```

Production:

```bash
mkdir -p logs
nohup python scripts/run_stable_arithmetic.py --configs data/FROZEN_SELECTED_CONFIGS.csv.gz --output-dir output/production --jobs 140 --chunk-rows 25 --resume > logs/stable_arithmetic_production.log 2>&1 &
echo $! | tee logs/stable_arithmetic_production.pid
```

Monitor:

```bash
python scripts/monitor_stable_arithmetic.py --run-dir output/production
tail -n 40 logs/stable_arithmetic_production.log
```

After completion:

```bash
python scripts/analyze_stable_arithmetic.py --run-dir output/production --bootstrap-reps 2000
cat output/production/ANALYSIS_AUDIT.json
cat output/production/AUTOMATED_SUMMARY.md
python scripts/make_handback.py --run-dir output/production --out-dir .
```
