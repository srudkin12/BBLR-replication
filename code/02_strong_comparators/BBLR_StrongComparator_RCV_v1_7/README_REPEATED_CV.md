# BBLR Repeated CV HPC v1.5

This is a **standalone production package**. It incorporates the complete
validated `BBLR_Univariate_Power_v1_4` tree plus the repeated-CV experiment.
No overlay, copy, merge, or external source-tree step is required.

## Question

Does more stable estimation of observable prediction risk allow BBLR to recover
more of the marginal-effect accuracy that Stage A showed is already available
inside the prediction-near-optimal tuning region?

## Fair information design

For a landmark with legacy `n_train=500`, v1.5 generates a development sample
of 1000 observations by default. This equals the 500 training + 500 validation
observations available to the earlier single-split procedure.

- `single_split`: 500 fit / 500 validation for tuning;
- `5-fold CV`: uses all 1000 development observations across folds;
- `10x5 repeated CV`: uses all 1000 development observations across repeats;
- after selection, **all selectors refit on the same 1000 development observations**;
- an independent test sample is then used for outcome and slope evaluation.

Thus differences in final performance are attributable to tuning rather than to
final estimation sample size.

## Outcome-only selectors

- minimum single-split outcome RMSE;
- minimum 5-fold CV outcome RMSE;
- minimum 10x5 repeated-CV outcome RMSE;
- within 0.5%, 1%, or 2% of minimum repeated-CV RMSE, choose the configuration
  with the most stable coefficient surface across fold fits.

True derivatives are never used by those selectors. Simulation truth is revealed
only afterwards for slope regret and oracle diagnostics.

## Stage-2 baseline

`eps=0` remains an exact nested LLR configuration. The same repeated-CV surface
therefore contains a matched LLR benchmark without a separate fitting system.

# Standalone workflow note

This v1.5 package supersedes the earlier scaffold/overlay packages. Do not copy files from them.
