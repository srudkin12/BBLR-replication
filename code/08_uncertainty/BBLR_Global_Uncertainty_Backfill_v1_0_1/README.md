# BBLR Global Uncertainty Backfill v1.0.1

## Recommended workflow

The package is designed for `/media/simon/E490B01790AFEE6C/BBLR2` and does not modify any existing BBLR result.

### 1. Preflight and smoke test

```bash
cd /media/simon/E490B01790AFEE6C/BBLR2/BBLR_Global_Uncertainty_Backfill_v1_0
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/preflight.py
python scripts/smoke_test.py
```

Expected endings:

```text
PASS_FOR_GLOBAL_UNCERTAINTY_BACKFILL
PASS_GLOBAL_UNCERTAINTY_SMOKE
```

### 2. Optional inventory-only pass

```bash
python scripts/scan_archive.py \
  --root /media/simon/E490B01790AFEE6C/BBLR2 \
  --out outputs/archive_inventory.csv
```

This is useful for inspecting what the project contains before running any summaries.

### 3. Full backfill

The task is I/O-heavy. Use modest concurrent readers and record the 80-core compute budget without launching 80 simultaneous ZIP readers:

```bash
python scripts/run_global_backfill.py \
  --root /media/simon/E490B01790AFEE6C/BBLR2 \
  --io-workers 12 \
  --compute-workers 80
```

Outputs include:

- `archive_inventory.csv`
- `schema_registry.csv`
- `uncertainty_method_cell_metric.csv`
- `uncertainty_paired_bblr_comparator.csv`
- `coverage_by_study.csv`
- `deduplication_audit.csv`
- `manual_review_queue.csv`
- `rerun_or_recovery_queue.csv`
- `GLOBAL_UNCERTAINTY_BACKFILL_REPORT.md`
- per-study conflict files where copied archives disagree on the same cell/replication/method identity.

The script then creates:

- `BBLR_Global_Uncertainty_Backfill_v1_0_HANDback.zip`
- `BBLR_Global_Uncertainty_Backfill_v1_0_HANDback.zip.sha256`

## Important limitation

This first backfill is intentionally conservative. Files with tuning-grid rows, ambiguous method encoding or multiple non-identical rows per `(cell, replication, method)` are not silently reduced. They are listed for manual adapter development. This is preferable to manufacturing apparently precise uncertainty summaries from the wrong observational unit.
