# Repeated-CV run

Project root:

`/media/simon/E490B01790AFEE6C/BBLR2/BBLR_RepeatedCV_HPC_v1_5`

## Preflight

```bash
source .venv/bin/activate
python -m pytest -q

python scripts/run_repeated_cv_landmarks.py \
  --cells POWER_LANDMARK_CELLS.csv \
  --reps 2 \
  --folds 3 \
  --repeats 2 \
  --jobs 4 \
  --chunk-reps 1 \
  --r-grid 0.18,0.38 \
  --eps-grid 0,0.12,0.38 \
  --output-dir results_hpc/rcv_preflight

python scripts/summarize_repeated_cv.py --run-dir results_hpc/rcv_preflight
python scripts/make_repeated_cv_handoff.py --run-dir results_hpc/rcv_preflight
```

## Production

Start with 100 replications per landmark because 10x5 CV is materially more
expensive than the earlier single-split simulation. Benchmark this run before
raising to 300 replications.

```bash
python scripts/run_repeated_cv_landmarks.py \
  --cells POWER_LANDMARK_CELLS.csv \
  --reps 100 \
  --folds 5 \
  --repeats 10 \
  --jobs 140 \
  --chunk-reps 1 \
  --resume \
  --output-dir results_hpc/repeated_cv_v1

python scripts/summarize_repeated_cv.py --run-dir results_hpc/repeated_cv_v1
python scripts/make_repeated_cv_handoff.py --run-dir results_hpc/repeated_cv_v1
```

Return only the generated repeated-CV handback ZIP and SHA. Keep raw grid and
selector shards on the Linux machine.
