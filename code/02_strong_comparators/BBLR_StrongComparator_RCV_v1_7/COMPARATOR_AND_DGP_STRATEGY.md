# Comparator and DGP strategy for the BBLR univariate study

## Central estimand distinction

Empirical tuning uses observable held-out outcome loss.  Simulation evaluation
then asks how the resulting estimator performs on the latent economic object,
`m'(x)`.  The design must therefore preserve both quantities for every method
and every tuning configuration.

## Comparator hierarchy

### Tier 0: sanity benchmark

**OLS.** Required for linear and near-linear DGPs.  BBLR should not be presented
as a universal replacement for a correctly specified parametric model.

### Tier 1: estimator-mechanism benchmark

**Uniform-ball local linear regression (LLR).** Essential.  It uses the same
Stage-1 local model and radius definition as BBLR.  The BBLR-versus-LLR contrast
therefore isolates the incremental effect of Stage 2.

### Tier 2: conventional local-polynomial benchmarks

**Local quadratic (LP2).** Tests whether BBLR's slope improvements can be
obtained by increasing local polynomial order instead of smoothing estimated
local slopes.

**Tricube local linear (KLLR).** Tests whether BBLR merely compensates for the
roughness of a uniform/hard-ball kernel.  This is a particularly important
comparator because conventional local regression normally uses smooth weights.

### Tier 3: global smooth-function benchmark

**Smoothing spline.** Provides a strong alternative in one dimension with a
well-defined derivative of the fitted smooth.  The smoothing parameter is
selected from observed validation Y RMSE exactly as BBLR's radii are.

### Later/robustness comparators

Potential additions after the initial mapping pass include penalised B-splines
/GAMs, Gaussian-kernel local polynomial regression and specialised derivative
estimators.  They should be added deliberately, rather than creating a very
large comparator list before BBLR's basic mechanism has been mapped.

## DGP design principle

A useful Monte Carlo should vary interpretable features rather than only names
of functions.  Three dimensions deserve separate control:

1. **global nonlinearity / curvature**;
2. **spatial frequency or localisation of slope changes**;
3. **support geometry / local information density**.

Noise and sample size form additional experimental axes.

### Smooth nonlinearity ladder

Start with a nested family such as

`m_a(x) = x + a x^2`

with `a=0` giving the linear sentinel and progressively larger values increasing
curvature.  This allows direct plotting of BBLR's incremental Stage-2 gain as a
function of RMS curvature.

### Frequency ladder

Use

`m_{a,w}(x) = x + a sin(w x)`.

Holding amplitude fixed while increasing `w` distinguishes *amount* of slope
heterogeneity from *speed* of slope variation.  BBLR's second stage is expected
to become more vulnerable to bias as the local slope field changes faster than
epsilon can resolve.

### Localisation ladder

Gaussian bumps with varying widths distinguish broad smooth curvature from a
local feature.  This directly tests whether prediction-selected epsilon can
wash out economically important local marginal-effect changes.

### Non-smooth stress class

Kinks, level jumps and step functions should be retained, but analysed as
stress tests with explicit guard bands around non-differentiable points.  They
should not be pooled with smooth DGPs when interpreting derivative RMSE.

## Performance objects

At minimum preserve:

- validation observed-Y RMSE (selection criterion);
- independent test observed-Y RMSE;
- true conditional-mean RMSE;
- true slope RMSE under the empirical X distribution;
- support-balanced slope RMSE on a uniform grid;
- boundary/interior slope RMSE;
- slope bias;
- Stage-1 and Stage-2 fallback shares;
- DGP curvature/roughness descriptors.

The main performance plane is independent-test outcome RMSE versus true slope
RMSE.  No restriction should be imposed that comparators first have 'similar'
prediction RMSE; the complete cloud and Pareto frontier should be shown.
