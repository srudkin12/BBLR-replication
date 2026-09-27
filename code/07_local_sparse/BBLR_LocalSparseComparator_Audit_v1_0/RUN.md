# Run instructions

Assume both downloaded files have been manually placed in:

`/media/simon/E490B01790AFEE6C/BBLR2`

## 1. Verify and unpack

```bash
cd /media/simon/E490B01790AFEE6C/BBLR2
sha256sum -c BBLR_LocalSparseComparator_Audit_v1_0.zip.sha256
unzip -o BBLR_LocalSparseComparator_Audit_v1_0.zip
cd BBLR_LocalSparseComparator_Audit_v1_0
```

## 2. Environment and tests

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Expected: `5 passed`.

## 3. Scientific preflight

```bash
rm -rf output/preflight
python scripts/run_audit.py \
  --cells data/PREFLIGHT_CELLS.csv \
  --output-dir output/preflight \
  --reps 1 \
  --jobs 4 \
  --chunk-reps 1 \
  --resume

python scripts/summarize_audit.py \
  --run-dir output/preflight \
  --expected-reps 1

cat output/preflight/COMPLETION_AUDIT.json
cat output/preflight/AUTOMATED_SUMMARY.md
```

The completion audit should report `PASS`. Preflight numerical results are code-validation only.

## 4. Full targeted audit: 80 workers

```bash
mkdir -p logs

nohup python scripts/run_audit.py \
  --cells data/AUDIT_CELLS.csv \
  --output-dir output/audit500 \
  --reps 500 \
  --jobs 80 \
  --chunk-reps 1 \
  --resume \
  > logs/local_sparse_audit500.log 2>&1 &

echo $! | tee logs/local_sparse_audit500.pid
```

The runner forces BLAS/OpenMP libraries to one thread per worker.

## 5. Monitor

```bash
python scripts/monitor_audit.py \
  --run-dir output/audit500 \
  --cells data/AUDIT_CELLS.csv \
  --reps 500

tail -n 50 logs/local_sparse_audit500.log
```

Rerunning the production command with `--resume` skips completed shards.

## 6. Summarise

```bash
python scripts/summarize_audit.py \
  --run-dir output/audit500 \
  --expected-reps 500

cat output/audit500/COMPLETION_AUDIT.json
cat output/audit500/AUTOMATED_SUMMARY.md
```

Key outputs include:

- `METHOD_CELL_SUMMARY.csv`
- `BBLR_VS_LOCAL_SPARSE.csv`
- `LOCAL_SPARSE_FRONTIER.csv`
- `LOCAL_SPARSE_TUNING_SUMMARY.csv`
- `BBLR_PAIRWISE.csv`

## 7. Create compact handback

```bash
python scripts/make_handback.py \
  --run-dir output/audit500 \
  --out-dir .
```

Return:

- `BBLR_LocalSparseComparator_Audit_v1_0_HANDback.zip`
- `BBLR_LocalSparseComparator_Audit_v1_0_HANDback.zip.sha256`

Raw shards remain on the workstation.
