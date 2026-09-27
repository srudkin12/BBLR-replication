# BBLR targeted local-sparse comparator audit v1.0

Standalone Linux package for the final targeted robustness exercise requested by the v0.12/v0.13 Journal of Econometrics adversarial review.

The package compares BBLR with `LL_ALASSO`, an adaptive local-linear Lasso, on nine frozen cells drawn from the completed moderate-dimensional coefficient-field production design. It also retains LLR, global Lasso, and OLS for context.

Scientific design and interpretation rules are frozen in `SCIENTIFIC_PREDECLARATION.md` before the production results are inspected.

Production size: **9 cells x 500 replications = 4,500 cell-replications**.

Default high-core run: **80 workers**, one BLAS/OpenMP thread per worker.

Expected handback:

- `BBLR_LocalSparseComparator_Audit_v1_0_HANDback.zip`
- `BBLR_LocalSparseComparator_Audit_v1_0_HANDback.zip.sha256`
