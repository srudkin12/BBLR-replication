# BBLR Alpha / Irrelevant Covariates / Dimension Production v1.0

Standalone full-production package for BBLR4, frozen after the two alpha/nuisance/dimension pilots.

The production design keeps all 82 predeclared Phase-2 DGP cells and raises the Monte Carlo count to **1,000 replications per cell (82,000 cell-replications)**. It adds two scientifically important diagnostics before production:

1. **Conditional matched-radius Stage-2 comparison**
   \[
   BBLR(\hat r_B,\hat\epsilon_B)\quad\text{versus}\quad LLR(\hat r_B)=BBLR(\hat r_B,0),
   \]
   with no retuning of the matched LLR.
2. **Dimension-normalised nuisance diagnostics**, including nuisance-vector norms divided by \(\sqrt{d-d_{active}}\), while retaining the raw vector norms and componentwise RMSE.

The package also corrects a pilot-only safeguard inconsistency: BBLR Stage 1, tuned LLR, the BBLR epsilon=0 branch, and matched-radius LLR now all use the same minimum-neighbour rule `max(d+5,15)`. This enforces the paper's exact finite-sample identity `BBLR(r,0)=LLR(r)` under the production safeguard.

The user manually places this ZIP and SHA-256 file in:

`/media/simon/E490B01790AFEE6C/BBLR2`

No download, overlay, or copy-merge workflow is required.

See `SCIENTIFIC_PREDECLARATION.md`, `PRODUCTION_CELL_MAP.md`, and `RUN.md`.
