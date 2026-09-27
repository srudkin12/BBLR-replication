# v1.0.1 corrections

This revision changes archive interpretation only; it does not rerun any Monte Carlo simulation.

- Corrects fine-locality design metadata to 16 cells x 1,500 replications = 24,000.
- Corrects the legacy replay to 212 distinct design cells and 220,000 total replications; suite is now part of cell identity.
- Uses authoritative consolidated production files for legacy replay and matched-H-lambda audits.
- Uses only `production1000` shards for the higher-dimensional study, excluding preflight rep-0 contamination.
- Recognises WSAFE, WSTRICT, WNORMAL and WCENTER as matched-H-lambda comparator methods.
- Maps operating-map `SELECTED-REPEATED-CV` to BBLR and `LLR-SELECTED-REPEATED-CV` to LLR so paired uncertainty is produced.
- Adds `source_selection_audit.csv` and replication-count diagnostics to coverage reporting.
