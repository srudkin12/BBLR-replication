# Preflight

After installing the environment, run:

```bash
python -m pytest -q

python scripts/run_bivariate_hpc.py \
  --cells PREFLIGHT_CELLS.csv \
  --reps 2 --folds 2 --repeats 1 \
  --jobs 8 --chunk-reps 1 \
  --r-grid 0.40,0.85 \
  --eps-grid 0,0.25,0.60 \
  --k-grid 40,100 \
  --knot-grid 4,6 \
  --alpha-grid 0.01,1 \
  --gap-grid 6,inf \
  --n-test 300 --grid-n 11 \
  --methods BBLR,LLR,LP2,AKLLR,AKLP2,TSPLINE,CTSPLINE,OLS,GQ2 \
  --resume \
  --output-dir results_hpc/bivariate_preflight

python scripts/summarize_bivariate.py --run-dir results_hpc/bivariate_preflight
cat results_hpc/bivariate_preflight/COMPLETION_AUDIT.json
```

Expected: 6 tests pass and 4/4 preflight chunks, status PASS.
