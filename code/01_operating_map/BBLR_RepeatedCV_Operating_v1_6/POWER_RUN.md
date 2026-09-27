# BBLR Power v1.4 run instructions

Project root: `/media/simon/E490B01790AFEE6C/BBLR2`

## A. Intensive analysis of the existing v1.3 shards

Run this from the unpacked v1.4 project directory.  It reads the authoritative
v1.3 shards in place and performs no new Monte Carlo simulation.

```bash
source .venv/bin/activate

python scripts/analyze_bblr_power_existing.py \
  --run-dir /media/simon/E490B01790AFEE6C/BBLR2/BBLR_Univariate_SupportRegime_v1_3/results_hpc/support_regime_v1 \
  --jobs 24

python scripts/make_existing_power_handoff.py \
  --analysis-dir /media/simon/E490B01790AFEE6C/BBLR2/BBLR_Univariate_SupportRegime_v1_3/results_hpc/support_regime_v1/power_analysis
```

Return the generated `existing_bblr_power_analysis_*.zip` and matching SHA.

## B. Landmark support-aware production

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs

python scripts/run_power_landmarks.py \
  --cells POWER_LANDMARK_CELLS.csv \
  --reps 300 \
  --jobs 140 \
  --chunk-reps 10 \
  --resume \
  --output-dir results_hpc/power_landmarks_v1 \
  2>&1 | tee logs/power_landmarks_v1.log

python scripts/summarize_univariate_hpc.py \
  --run-dir results_hpc/power_landmarks_v1

python scripts/analyze_power_landmarks.py \
  --run-dir results_hpc/power_landmarks_v1

python scripts/make_power_handoff.py \
  --run-dir results_hpc/power_landmarks_v1
```

There are 12 cells × 300 replications = 3,600 Monte Carlo replications and
360 expected shards at 10 replications per shard.  Raw shards remain on the
Linux machine.  Return only the generated power-handback ZIP and SHA.
