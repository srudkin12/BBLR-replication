# BBLR Legacy Coefficient-Field Replay v1.0

Purpose: revisit the frozen univariate and bivariate BBLR experiments after the coefficient-field/alpha interpretation emerged, **without retuning any historical experiment**.

The package replays only the already selected estimators from the archived handbacks and adds:

- local tangent intercept / tangent alpha
  `alpha(x) = m(x) - x' grad m(x)`;
- reference-state alpha / intercept `alpha0 = m(0)`;
- support classification of the reference state;
- supported-grid tangent-alpha risk;
- matched-radius LLR at BBLR's selected Stage-1 radius.

Historical coverage:

- 92,000 univariate repeated-CV operating-map replications;
- 27,000 univariate strong-comparator replications;
- 58,000 broad bivariate replications;
- 19,000 support-aware bivariate replications;
- 24,000 fine-locality bivariate replications.

Total historical replications revisited: **220,000**.

No CV surface is rerun. The deterministic seeds and archived selected tuning values are frozen in `data/*_FROZEN_CONFIGS.csv.gz`.

## Scientific distinction

For centred local linear coefficients at query x,

`m(z) ~= a(x) + g(x)'(z-x)`, so `a(x)=m(x)`.

The same tangent hyperplane in global coordinates is

`m(z) ~= alpha(x) + g(x)'z`, with

`alpha(x)=m(x)-x'g(x)`.

For a globally linear DGP, alpha(x) is constant and equals the conventional regression intercept. For nonlinear DGPs it is a local tangent-intercept field.

Reference alpha `alpha0=m(0)` is reported separately. In hard gaps/holes around zero it is explicitly labelled extrapolative/unsupported.

## Integrity

Every historical method row with an archived risk measure is replayed on the identical deterministic development/test samples. `REPLAY_INTEGRITY_METRICS.csv` compares the reproduced old risks with their archived values. The analysis status is PASS only if there are no failed fits and maximum replay discrepancy is <= 1e-8.

## Workflow

See `RUN.md`. The package is standalone. Place ZIP/SHA manually in `/media/simon/E490B01790AFEE6C/BBLR2`, verify, unzip, test, preflight, then launch the frozen replay.
