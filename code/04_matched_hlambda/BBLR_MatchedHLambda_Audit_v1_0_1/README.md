# BBLR Matched-H-lambda Mechanism Audit v1.0.1

Purpose: resolve the main empirical novelty question left by BBLR Paper 1 v0.9.
The audit does **not** rerun or retune repeated CV. It carries forward all 92,000 frozen
repeated-CV-selected BBLR configurations from the validated v1.6 operating map and
regenerates the original development/test samples from deterministic seeds.

For every replication it compares:

1. BBLR at the frozen selected `(r_hat_B, eps_hat_B)`;
2. a strict one-stage local-linear derivative estimator using the exact matched kernel
   `W_lambda`, with `lambda=eps_hat_B/r_hat_B`;
3. a secondary safeguard-matched W estimator that invokes the production 15-NN local-linear
   fallback only when the strict weighted fit is under-supported or singular;
4. production LLR at the same BBLR-selected radius.

The strict W estimator is the primary mechanism comparator. The safeguarded version separates
finite-sample support/singularity behaviour from the kernel comparison.

## Scientific scope

All 92 operating-map cells are retained. This includes smooth and polynomial controls,
high-frequency failures, normal/skewed density designs, soft depletion, hard gaps, noise ladders,
and sample-size ladders. Results are reported on test-support risk and fixed-grid risk, with a
separate structurally-supported grid measure for hard-gap designs. Central-window, outer-window,
and boundary diagnostics are retained at replication level.

The audit also stratifies by `lambda<1`, `lambda=1`, `lambda>1`, and by `|Gamma_lambda|`
because v0.9 shows that the first-order density-gradient distinction is often inactive.

## Workflow

Use `RUN.md`. The package is standalone. Do not overlay it onto another BBLR directory.

## v1.0.1 compatibility hotfix

v1.0.1 changes only the unit-test integration helper from the removed `numpy.trapz` alias to `numpy.trapezoid`, and updates package/handback version labels. Estimator code, frozen selected configurations, DGP definitions, scientific pre-declaration, and production logic are unchanged from v1.0.
