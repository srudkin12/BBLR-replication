# BBLR Univariate Support–Regime Study v1.3

## Purpose

The preceding smooth-DGP phase map established three useful facts:

1. BBLR's second stage materially improves slope recovery relative to its hard-ball
   local-linear Stage 1.
2. Globally smooth one-dimensional methods, especially smoothing splines, retain
   a strong advantage when the conditional mean is globally smooth and well
   supported.
3. BBLR becomes comparatively more interesting when sharp marginal-effect
   changes interact with weak or disconnected support.

The v1.3 experiment therefore stops adding arbitrary smooth functions and
targets the interaction between **marginal-effect geometry** and **covariate
support geometry**.

## Controlled conditional-mean family

The experiment uses

\[
m'_{\delta,c}(x)=1+\tanh\left(\frac{x-c}{\delta}\right),
\]

with integrated conditional mean

\[
m_{\delta,c}(x)
=
x+\delta\log\cosh\left(\frac{x-c}{\delta}\right).
\]

The limiting slopes are always 0 and 2.  Two aspects vary independently:

- transition width `delta`: 0.80, 0.40, 0.20, 0.10, 0.05;
- transition centre `c`: -0.80, -0.40, 0.00, 0.40, 0.80.

This gives 25 smooth DGPs and makes "how sharp?" distinct from "where does the
slope change occur?".

## Support designs

Two support families are crossed with the DGPs.

### Hard central gaps

- G00: no gap (uniform support);
- G10: half-width 0.10;
- G25: half-width 0.25;
- G40: half-width 0.40;
- G60: half-width 0.60;
- G80: half-width 0.80.

For `Gg`, observations are uniform on the two retained intervals

\[
[-2,-g]\cup[g,2].
\]

This is a deliberately clean test of disconnected support.

### Soft central depletion

A central window `[-0.4,0.4]` remains populated but is progressively thinned:

- D75: central retention 0.75;
- D50: 0.50;
- D25: 0.25;
- D10: 0.10;
- D02: 0.02.

These designs distinguish a literal support gap from a very low-density region.

## Comparators

The existing comparator set is retained and two density-adaptive local
polynomial methods are added:

- OLS;
- fixed-radius hard-ball LLR;
- fixed-radius local quadratic (LP2);
- fixed-bandwidth tricube local linear (KLLR);
- **AKLLR**: tricube local linear using k nearest neighbours;
- **AKLP2**: tricube local quadratic using k nearest neighbours;
- smoothing spline;
- BBLR.

AKLLR and AKLP2 have query-specific effective bandwidths.  They contract in
dense regions and expand in sparse regions.  Any BBLR advantage over these
methods therefore cannot be attributed merely to a fixed local-polynomial
bandwidth.

## Tuning

All methods are tuned using independent validation outcome RMSE only.

True mean and true marginal-effect surfaces are used only after fitting for
simulation evaluation.

For every selected estimator the analysis retains:

- validation outcome RMSE;
- independent test outcome RMSE;
- test conditional-mean RMSE;
- test slope RMSE;
- slope RMSE near and away from the transition;
- left- and right-regime slope RMSE;
- Pareto-frontier membership.

## First production pass

The first pass fixes:

- n = 500;
- noise SD = 0.25;
- 200 replications per cell.

This gives:

- 25 DGPs;
- 11 support designs;
- 275 design cells;
- 55,000 Monte Carlo replications;
- 2,200 shards at 25 replications per shard.

Fixing sample size and noise is deliberate.  The immediate objective is to map
the geometry interaction cleanly.  Noise and sample-size scaling should be
added only after the regions of methodological difference are identified.

## Main diagnostic question

The central object is the pair

\[
\left(RMSE_Y,\ RMSE_{\beta}\right),
\]

evaluated for outcome-selected estimators.

The phase map should identify where BBLR:

- dominates ordinary/fixed local regression;
- survives density-adaptive local polynomial competition;
- approaches or beats a smoothing spline;
- fails because its second stage oversmooths genuine local slope changes.

The signed quantity

\[
|c|-g
\]

is particularly useful for hard-gap designs:

- negative: the transition centre lies inside the gap;
- zero: the transition lies at the support edge;
- positive: the transition lies in observed support.

This creates a direct phase variable linking the geometry of the marginal-effect
change to the geometry of the data.
