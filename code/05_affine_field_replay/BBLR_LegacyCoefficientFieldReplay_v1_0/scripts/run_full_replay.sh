#!/usr/bin/env bash
set -euo pipefail
JOBS="${JOBS:-80}"
OUT="${OUT:-output/production}"
mkdir -p "$OUT" logs
python scripts/run_replay.py --track uv92 --output-dir "$OUT" --jobs "$JOBS" --chunk-reps 10 --resume
python scripts/run_replay.py --track uv27 --output-dir "$OUT" --jobs "$JOBS" --chunk-reps 10 --resume
python scripts/run_replay.py --track biv101 --output-dir "$OUT" --jobs "$JOBS" --chunk-reps 5 --resume
python scripts/summarize_replay.py --run-dir "$OUT"
