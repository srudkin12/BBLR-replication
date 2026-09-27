# Phase-map production command

Project root on the user's Linux machine:

`/media/simon/E490B01790AFEE6C/BBLR2`

After unpacking this package and activating `.venv`, run:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/run_univariate_hpc.py \
  --designs uniform,normal,skewed,gap \
  --n-grid 500 \
  --noise-grid 0.10,0.25,0.50 \
  --reps 150 \
  --n-test 2000 \
  --jobs 140 \
  --chunk-reps 25 \
  --resume \
  --output-dir results_hpc/phase_map_v1 \
  2>&1 | tee logs/phase_map_v1.log
```

Expected design cells:

`37 DGPs x 4 X-designs x 3 noise levels = 444 cells`

Expected replications:

`444 x 150 = 66,600 replications`

Expected shards at 25 reps/shard:

`444 x 6 = 2,664 shards`

After completion:

```bash
python scripts/summarize_univariate_hpc.py \
  --run-dir results_hpc/phase_map_v1

python scripts/analyze_phase_map.py \
  --run-dir results_hpc/phase_map_v1

cat results_hpc/phase_map_v1/COMPLETION_AUDIT.json

python scripts/make_compact_handoff.py \
  --run-dir results_hpc/phase_map_v1
```

Return only the compact handback ZIP + SHA routinely.  Keep all raw shards on
the Linux machine.
