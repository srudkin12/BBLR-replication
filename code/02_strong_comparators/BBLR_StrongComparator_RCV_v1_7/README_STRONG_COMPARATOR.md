# BBLR Strong-Comparator Repeated-CV v1.7

## Purpose

This pass is the adversarial confirmation stage following the 92,000-replication
BBLR operating-characteristics study.

It asks whether BBLR's favourable operating regions survive against strong
one-dimensional comparators when **every method is tuned using the same
outcome-only repeated-CV rule**.

## Methods

- BBLR
- LLR (the exact `epsilon=0` nested case)
- adaptive kNN local linear (AKLLR)
- adaptive kNN local quadratic (AKLP2)
- smoothing spline (SPLINE)
- componentwise adaptive local linear (CAKLLR)
- componentwise adaptive local quadratic (CAKLP2)
- componentwise smoothing spline (CSPLINE)

Support-aware methods infer components from training X only.  No true support
label or derivative information is used for tuning.

## CV design

The confirmation uses 5-fold CV repeated 5 times for **all** methods.  The
previous BBLR-only experiment established that repeated CV is preferable to a
single split.  A 5x5 rule preserves repeated-risk estimation while keeping the
full strong-comparator experiment practical over several days.

After tuning, each selected method is refitted on the same full development
sample and evaluated on the same independent test sample.

## Frozen panel

`STRONG_COMPARATOR_CELLS.csv` contains 27 representative cells selected from the
92-cell operating map.  The panel deliberately includes:

- linear and smooth controls;
- polynomial and transition shapes;
- high-frequency neutral/failure cases;
- local bump and slope-kink cases;
- normal, skewed and gapped density geometry;
- soft support depletion from moderate to extreme;
- hard support gaps;
- transition-width variation;
- transition location outside a gap;
- low/high noise high-frequency cases;
- sample-size scaling in the sparse-centre regime.

This is an adversarial panel, not a BBLR-favourable subset.

## Primary evidence

For each cell and comparator the handback reports paired:

- test outcome RMSE;
- test slope RMSE on observed support;
- full-grid slope RMSE;
- BBLR/comparator slope-RMSE ratio;
- BBLR slope-win probability;
- BBLR outcome-win probability;
- joint Pareto dominance;
- method Pareto share.
