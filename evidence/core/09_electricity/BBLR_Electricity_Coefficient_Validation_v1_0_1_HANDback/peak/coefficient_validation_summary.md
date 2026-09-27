# Coefficient-validation audit: Daily peak demand

All configurations are frozen from the 1,000-repetition outcome-CV production audit. No method is retuned on coefficient-validation loss.

## Random held-out contrasts

### Temperature separation 0.5-1 °C
- LP2: Simpson secant MSE 0.535376; midpoint secant MSE 0.535352; equal-location Simpson MSE 0.650179.
- PSpline: Simpson secant MSE 0.53557; midpoint secant MSE 0.535564; equal-location Simpson MSE 0.651658.
- BBLR: Simpson secant MSE 0.535683; midpoint secant MSE 0.535727; equal-location Simpson MSE 0.649687.
- Hinge: Simpson secant MSE 0.537012; midpoint secant MSE 0.537148; equal-location Simpson MSE 0.658709.
- LLR: Simpson secant MSE 0.537645; midpoint secant MSE 0.537492; equal-location Simpson MSE 0.653444.
- BBLR_MatchedZero: Simpson secant MSE 0.537645; midpoint secant MSE 0.537492; equal-location Simpson MSE 0.653444.

### Temperature separation 1-2 °C
- PSpline: Simpson secant MSE 0.130287; midpoint secant MSE 0.13027; equal-location Simpson MSE 0.17138.
- BBLR: Simpson secant MSE 0.131051; midpoint secant MSE 0.131024; equal-location Simpson MSE 0.170776.
- LP2: Simpson secant MSE 0.13113; midpoint secant MSE 0.131057; equal-location Simpson MSE 0.172318.
- Hinge: Simpson secant MSE 0.131573; midpoint secant MSE 0.131716; equal-location Simpson MSE 0.176381.
- LLR: Simpson secant MSE 0.132682; midpoint secant MSE 0.133603; equal-location Simpson MSE 0.175273.
- BBLR_MatchedZero: Simpson secant MSE 0.132682; midpoint secant MSE 0.133603; equal-location Simpson MSE 0.175273.

### Temperature separation 2-4 °C
- PSpline: Simpson secant MSE 0.0344823; midpoint secant MSE 0.0345371; equal-location Simpson MSE 0.0455509.
- BBLR: Simpson secant MSE 0.0352136; midpoint secant MSE 0.0352459; equal-location Simpson MSE 0.0455635.
- LP2: Simpson secant MSE 0.0352576; midpoint secant MSE 0.0351562; equal-location Simpson MSE 0.0471045.
- BBLR_MatchedZero: Simpson secant MSE 0.0354836; midpoint secant MSE 0.0359947; equal-location Simpson MSE 0.0454366.
- LLR: Simpson secant MSE 0.0354836; midpoint secant MSE 0.0359947; equal-location Simpson MSE 0.0454366.
- Hinge: Simpson secant MSE 0.0362763; midpoint secant MSE 0.0378077; equal-location Simpson MSE 0.0492166.

## Grouped Week held-out contrasts

### Temperature separation 0.5-1 °C
- PSpline: Simpson secant MSE 0.505377; midpoint secant MSE 0.505372; equal-location Simpson MSE 0.588853.
- Hinge: Simpson secant MSE 0.505462; midpoint secant MSE 0.505468; equal-location Simpson MSE 0.587498.
- LP2: Simpson secant MSE 0.505824; midpoint secant MSE 0.505808; equal-location Simpson MSE 0.58807.
- BBLR: Simpson secant MSE 0.507322; midpoint secant MSE 0.507899; equal-location Simpson MSE 0.590572.
- LLR: Simpson secant MSE 0.507322; midpoint secant MSE 0.507899; equal-location Simpson MSE 0.590572.

### Temperature separation 1-2 °C
- PSpline: Simpson secant MSE 0.125089; midpoint secant MSE 0.125092; equal-location Simpson MSE 0.148722.
- LP2: Simpson secant MSE 0.125889; midpoint secant MSE 0.125924; equal-location Simpson MSE 0.149043.
- Hinge: Simpson secant MSE 0.126376; midpoint secant MSE 0.126014; equal-location Simpson MSE 0.152039.
- BBLR: Simpson secant MSE 0.126493; midpoint secant MSE 0.126584; equal-location Simpson MSE 0.148604.
- LLR: Simpson secant MSE 0.126493; midpoint secant MSE 0.126584; equal-location Simpson MSE 0.148604.

### Temperature separation 2-4 °C
- PSpline: Simpson secant MSE 0.0343664; midpoint secant MSE 0.0344305; equal-location Simpson MSE 0.0439577.
- LP2: Simpson secant MSE 0.0353545; midpoint secant MSE 0.0353099; equal-location Simpson MSE 0.0450104.
- LLR: Simpson secant MSE 0.0359875; midpoint secant MSE 0.0364716; equal-location Simpson MSE 0.0451594.
- BBLR: Simpson secant MSE 0.0359875; midpoint secant MSE 0.0364716; equal-location Simpson MSE 0.0451594.
- Hinge: Simpson secant MSE 0.0367907; midpoint secant MSE 0.0376554; equal-location Simpson MSE 0.0493473.

## Week-block coefficient stability

- BBLR: median heating-to-cooling crossing 22.5 °C (2.5--97.5% 21.5--25.2; available in 1.000 of resamples); P[g(18)<0]=1.000; P[g(28)>0]=0.999; median total variation 0.57.
- LLR: median heating-to-cooling crossing 22 °C (2.5--97.5% 20.6--24.1; available in 1.000 of resamples); P[g(18)<0]=1.000; P[g(28)>0]=0.989; median total variation 2.559.
- LP2: median heating-to-cooling crossing 22.5 °C (2.5--97.5% 21.3--23.6; available in 1.000 of resamples); P[g(18)<0]=1.000; P[g(28)>0]=1.000; median total variation 0.7222.
- PSpline: median heating-to-cooling crossing 22 °C (2.5--97.5% 21.4--23.1; available in 1.000 of resamples); P[g(18)<0]=1.000; P[g(28)>0]=1.000; median total variation 0.4554.
- Hinge: median heating-to-cooling crossing 26 °C (2.5--97.5% 26--26; available in 1.000 of resamples); P[g(18)<0]=1.000; P[g(28)>0]=1.000; median total variation 0.3898.

## Interpretation

Held-out secant losses are noisy empirical implications of the coefficient field, not observations of derivative truth. The Monte Carlo sections use direct derivative MSE because truth is known there; this audit exists only because real-data derivative truth is latent. Stability, smoothness, and agreement are reported separately from held-out contrast accuracy.
