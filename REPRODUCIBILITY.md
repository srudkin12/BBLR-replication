# Reproducibility guide

## Source-of-truth hierarchy

1. canonical code package and frozen configuration;
2. validated production HANDback and archive SHA-256;
3. compact machine-readable evidence extracted from that HANDback;
4. manuscript tables/figures generated from the accepted evidence;
5. manuscript prose.

## Important

Do not rerun all experiments merely to verify the repository. Historical production runs are preserved by exact archive hashes and compact returned evidence.

For a fresh rerun of an individual block:

1. read that block's README and DESIGN_NOTE;
2. use its frozen configuration and deterministic seeds;
3. keep numerical-library threading at one thread per worker;
4. run tests/preflight/smoke before production;
5. preserve negative/adverse cells;
6. do not retune after observing results.

Some production jobs were designed for high-core Linux machines. Runtime is not evidence of correctness.

## Full archived HANDbacks

The research machine retains the full HANDbacks. They are not all copied into the public repository because replication-level rows/checkpoints can be large and are unnecessary for ordinary source inspection. Exact hashes are in `provenance/EXTERNAL_ARCHIVE_MANIFEST.csv`.

## Electricity data

Electricity observations are not distributed in the public repository. Obtain and reconstruct them locally using `DATA_AVAILABILITY.md`; keep reconstructed `source_data` files outside version control.

## Manuscript reproduction layer

`manuscript_outputs/` is intentionally reserved. A final small mapping from manuscript figures/tables to accepted compact evidence should be generated after this release candidate passes audit.

## Licence

Repository code and documentation are released under the MIT License. Third-party dependencies retain their own licences. Electricity observations are not distributed with the repository.

