# Scientific predeclaration — alpha, nuisance and dimension full production

## Status entering production
Two pilots established a coherent coefficient-field pattern worth confirming at paper-level Monte Carlo precision:

- BBLR strongly suppresses nuisance local slopes relative to other local smoothers, while global OLS/ridge/lasso can shrink nuisance slopes still more strongly in correctly specified or nearly linear settings.
- In nonlinear transition designs, BBLR often preserves active local-gradient variation and tangent-alpha variation substantially better than global shrinkage estimators.
- Tangent alpha, \(\alpha(x)=m(x)-x^\top\nabla m(x)\), is the cleaner BBLR estimand than reference-state alpha \(\alpha_0=m(0)\); the latter is deliberately difficult or extrapolative when zero lies outside support.
- Ambient dimension through d=12 creates a real cost when only two coordinates are active, but the loss is partly recovered by larger sample sizes and does not amount to immediate dimensional collapse in the studied designs.

These pilot results are directional only. The production run freezes the full 82-cell design and uses 1,000 replications per cell.

## Production scale
**82 cells x 1,000 replications = 82,000 cell-replications.** Outcome tuning remains 3 folds x 2 repeats, as frozen in the production cell file. BLAS/OpenMP threads are forced to 1; the launch configuration uses 80 workers.

## Primary questions
1. How accurately does BBLR recover the **local tangent intercept** \(\alpha(x)=m(x)-x^\top\nabla m(x)\)?
2. How accurately does it recover the **reference-state alpha** \(\alpha_0=m(0)\) when zero is supported, and how rapidly does this deteriorate under extrapolation?
3. How strongly does BBLR suppress truly irrelevant local slopes relative to local smoothers and global shrinkage estimators?
4. Does BBLR occupy a distinct **coefficient-field frontier**: lower nuisance variation than local smoothers while preserving more genuine active variation than global shrinkage methods?
5. How do these risks change with ambient dimension, effective dimension, support geometry and sample size through d=12?
6. Conditional on BBLR's selected Stage-1 radius, what does **activating Stage 2** do to active gradients, tangent alpha, reference alpha, coefficient-field risk and nuisance slopes?

## Exact nesting and matched-radius design
Production enforces a common Stage-1/LLR safeguard:

`min_points = max(d+5, 15)`.

The same rule is used for:
- BBLR Stage 1;
- the epsilon=0 BBLR candidate in CV;
- separately tuned LLR;
- the matched-radius LLR.

Therefore the implementation satisfies the intended identity
\[
BBLR(r,0)=LLR(r)
\]
under the production safeguard.

For every replication, after outcome CV chooses \((\hat r_B,\hat\epsilon_B)\), production evaluates
\[
LLR(\hat r_B)
\]
**without retuning**. This is the conditional matched-radius Stage-2 audit. It must not be conflated with the separately tuned comparison \(BBLR(\hat r_B,\hat\epsilon_B)\) versus \(LLR(\hat r_L)\).

## Primary risk metrics
- conditional-mean RMSE;
- total vector-gradient RMSE;
- active-coordinate vector-gradient RMSE;
- tangent-alpha RMSE;
- reference-alpha0 absolute and squared error;
- joint local coefficient-field vector RMSE;
- nuisance-vector RMSE and nuisance norm quantiles;
- dimension-normalised nuisance-vector RMSE and norm quantiles, dividing vector norms by \(\sqrt{d-d_{active}}\);
- raw and dimension-normalised nuisance threshold exceedance rates;
- BBLR Stage-2 selection and selected epsilon/r.

Component-averaged gradient RMSE is retained only for continuity. Vector risk is primary for dimension comparisons.

## DGP blocks
All 82 Phase-2 cells are retained:
- two-active-coordinate ambient-dimension sequence, d in {2,3,5,8,12};
- pure flat/null DGPs in which every slope is irrelevant;
- independent and correlated nuisance coordinates;
- aligned versus orthogonal soft support depletion;
- all-active linear and smooth designs;
- shifted factor support [0,2]^d, making alpha0=m(0) extrapolative;
- n-scaling cells at d=8 and 12 with n in {600,1200,2400} when combined with baseline cells.

## Methods
Feasible methods: BBLR, separately tuned LLR, matched-radius LLR at BBLR's selected r, adaptive-k LLR, adaptive-k local quadratic, ridge, lasso, OLS and global quadratic.

Diagnostic/oracle methods: oracle-active BBLR and oracle-active OLS where active_dim>0; mean-only in pure-null cells. Oracle methods are not feasible empirical competitors.

All tunable feasible methods use repeated outcome CV. The BBLR epsilon/r grid remains {0,0.5,1,1.5,2}.

## Interpretation constraints
- Tangent alpha and reference-state alpha are different estimands and must not be conflated.
- Reference alpha at zero is substantively meaningful only when the zero-factor state is meaningful and supported; shifted-support cells are explicit extrapolation controls.
- BBLR is not claimed to perform variable selection. Lasso/OLS/ridge are expected to suppress nuisance slopes more aggressively in exactly linear/null settings.
- The substantive coefficient-field claim requires a **joint frontier**: nuisance regularisation plus preservation of active nonlinear coefficients.
- Raw nuisance-vector norms mechanically grow with nuisance dimension; dimension-normalised diagnostics are therefore primary for cross-d comparisons.
- Oracle-active methods quantify the cost of nuisance geometry and are not admissible empirical procedures.
- d>2 results are an empirical dimension-scaling exercise, not a multivariate asymptotic theorem.
- Hard/unsupported reference states remain extrapolation problems, not regular local recovery.
- The matched-radius comparison isolates the conditional increment from Stage 2 at BBLR's selected Stage-1 radius; it does not isolate every consequence of joint scale selection.
