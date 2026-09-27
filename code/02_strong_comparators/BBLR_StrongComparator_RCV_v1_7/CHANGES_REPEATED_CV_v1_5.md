# Changes — BBLR Repeated CV HPC v1.5

- Full standalone package built from validated Power v1.4 production tree.
- No overlay or manual copy workflow.
- Adds single split, 5-fold and repeated K-fold prediction-risk surfaces.
- Adds coefficient-surface stability across CV fold fits.
- Adds prediction-admissible stability selectors at 0.5%, 1%, and 2%.
- Maintains exact `BBLR(r,0) = LLR(r)` nesting.
- Equalises final refit sample size across tuning selectors.
- Adds full-grid post-selection slope oracle diagnostics without leaking truth
  into empirical selection.
- Adds resumable HPC grid/selector shards, summaries and compact handback.
- Adds repeated-CV unit tests.
