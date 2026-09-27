#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"; source .venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs results_hpc

run_and_pack () {
  local OUT="$1"
  python scripts/summarize_bivariate.py --run-dir "$OUT"
  python scripts/make_bivariate_handoff.py --run-dir "$OUT"
}

# STAGE 1: broad operating map. 58 cells, 1000 reps, strong general comparators.
OUT1="$ROOT/results_hpc/bivariate_broad_v1"
echo '=== STAGE 1: BROAD BIVARIATE MAP ==='
python scripts/run_bivariate_hpc.py \
  --cells BIVARIATE_CELLS.csv --reps 1000 --folds 5 --repeats 5 \
  --methods BBLR,LLR,LP2,AKLLR,AKLP2,TSPLINE,OLS,GQ2 \
  --jobs 140 --chunk-reps 5 --resume --output-dir "$OUT1" \
  2>&1 | tee -a logs/bivariate_broad_v1.log
run_and_pack "$OUT1"

# STAGE 2: support-aware adversarial pass. Runs only after Stage 1 is frozen.
OUT2="$ROOT/results_hpc/bivariate_support_aware_v1"
echo '=== STAGE 2: SUPPORT-AWARE BIVARIATE PASS ==='
python scripts/run_bivariate_hpc.py \
  --cells BIVARIATE_SUPPORT_AWARE_CELLS.csv --reps 1000 --folds 5 --repeats 5 \
  --methods BBLR,LLR,AKLLR,AKLP2,TSPLINE,CTSPLINE \
  --jobs 140 --chunk-reps 2 --resume --output-dir "$OUT2" \
  2>&1 | tee -a logs/bivariate_support_aware_v1.log
run_and_pack "$OUT2"

# STAGE 3: finer BBLR radius/epsilon map in the scientifically difficult cells.
# If this stage is still running when you return, Stages 1-2 already have complete handbacks.
OUT3="$ROOT/results_hpc/bivariate_fine_locality_v1"
echo '=== STAGE 3: FINE LOCALITY MAP ==='
python scripts/run_bivariate_hpc.py \
  --cells BIVARIATE_FINE_CELLS.csv --reps 1500 --folds 5 --repeats 5 \
  --methods BBLR,LLR,AKLP2,TSPLINE \
  --r-grid 0.20,0.25,0.30,0.35,0.42,0.50,0.60,0.72,0.85,1.00,1.20,1.45,1.75,2.10,2.60,3.20,4.00,6.00 \
  --eps-grid 0,0.08,0.12,0.16,0.20,0.25,0.30,0.36,0.43,0.52,0.62,0.75,0.90,1.10,1.35,1.65,2.00,2.50,3.20,4.00,6.00 \
  --jobs 140 --chunk-reps 2 --resume --output-dir "$OUT3" \
  2>&1 | tee -a logs/bivariate_fine_locality_v1.log
run_and_pack "$OUT3"

echo '=== ALL BIVARIATE STAGES COMPLETE ==='
