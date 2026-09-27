# Coefficient-validation audit: 18:00 demand

All configurations are frozen from the 1,000-repetition outcome-CV production audit. No method is retuned on coefficient-validation loss.

## Random held-out contrasts

### Temperature separation 0.5-1 °C
- Hinge: Simpson secant MSE 0.790311; midpoint secant MSE 0.790543; equal-location Simpson MSE 0.802215.
- LP2: Simpson secant MSE 0.790603; midpoint secant MSE 0.790545; equal-location Simpson MSE 0.802159.
- PSpline: Simpson secant MSE 0.790869; midpoint secant MSE 0.79086; equal-location Simpson MSE 0.804515.
- BBLR: Simpson secant MSE 0.790959; midpoint secant MSE 0.790963; equal-location Simpson MSE 0.799481.
- LLR: Simpson secant MSE 0.793034; midpoint secant MSE 0.793231; equal-location Simpson MSE 0.801577.
- BBLR_MatchedZero: Simpson secant MSE 0.793034; midpoint secant MSE 0.793231; equal-location Simpson MSE 0.801577.

### Temperature separation 1-2 °C
- PSpline: Simpson secant MSE 0.191229; midpoint secant MSE 0.191214; equal-location Simpson MSE 0.212842.
- Hinge: Simpson secant MSE 0.191466; midpoint secant MSE 0.191044; equal-location Simpson MSE 0.213814.
- LP2: Simpson secant MSE 0.191953; midpoint secant MSE 0.191907; equal-location Simpson MSE 0.213293.
- BBLR: Simpson secant MSE 0.192068; midpoint secant MSE 0.192101; equal-location Simpson MSE 0.210426.
- LLR: Simpson secant MSE 0.193337; midpoint secant MSE 0.194019; equal-location Simpson MSE 0.211478.
- BBLR_MatchedZero: Simpson secant MSE 0.193337; midpoint secant MSE 0.194019; equal-location Simpson MSE 0.211478.

### Temperature separation 2-4 °C
- PSpline: Simpson secant MSE 0.0506687; midpoint secant MSE 0.0507442; equal-location Simpson MSE 0.0595247.
- LP2: Simpson secant MSE 0.0512746; midpoint secant MSE 0.05121; equal-location Simpson MSE 0.0605636.
- BBLR: Simpson secant MSE 0.0514322; midpoint secant MSE 0.0515586; equal-location Simpson MSE 0.058506.
- BBLR_MatchedZero: Simpson secant MSE 0.0518448; midpoint secant MSE 0.0525588; equal-location Simpson MSE 0.0568694.
- LLR: Simpson secant MSE 0.0518448; midpoint secant MSE 0.0525588; equal-location Simpson MSE 0.0568694.
- Hinge: Simpson secant MSE 0.0521925; midpoint secant MSE 0.0535354; equal-location Simpson MSE 0.0640895.

## Grouped Week held-out contrasts

### Temperature separation 0.5-1 °C
- Hinge: Simpson secant MSE 0.719475; midpoint secant MSE 0.719648; equal-location Simpson MSE 0.725534.
- LP2: Simpson secant MSE 0.721032; midpoint secant MSE 0.720969; equal-location Simpson MSE 0.726934.
- BBLR_MatchedZero: Simpson secant MSE 0.72151; midpoint secant MSE 0.721551; equal-location Simpson MSE 0.726933.
- BBLR: Simpson secant MSE 0.721527; midpoint secant MSE 0.721543; equal-location Simpson MSE 0.72525.
- PSpline: Simpson secant MSE 0.72215; midpoint secant MSE 0.722164; equal-location Simpson MSE 0.729773.
- LLR: Simpson secant MSE 0.722865; midpoint secant MSE 0.723495; equal-location Simpson MSE 0.728683.

### Temperature separation 1-2 °C
- Hinge: Simpson secant MSE 0.177256; midpoint secant MSE 0.176927; equal-location Simpson MSE 0.191804.
- LP2: Simpson secant MSE 0.17792; midpoint secant MSE 0.177932; equal-location Simpson MSE 0.190973.
- BBLR_MatchedZero: Simpson secant MSE 0.178151; midpoint secant MSE 0.178255; equal-location Simpson MSE 0.190842.
- PSpline: Simpson secant MSE 0.178218; midpoint secant MSE 0.178254; equal-location Simpson MSE 0.192398.
- BBLR: Simpson secant MSE 0.178581; midpoint secant MSE 0.178607; equal-location Simpson MSE 0.190334.
- LLR: Simpson secant MSE 0.17891; midpoint secant MSE 0.179114; equal-location Simpson MSE 0.189295.

### Temperature separation 2-4 °C
- LP2: Simpson secant MSE 0.0499574; midpoint secant MSE 0.049903; equal-location Simpson MSE 0.0600009.
- PSpline: Simpson secant MSE 0.0499948; midpoint secant MSE 0.0500945; equal-location Simpson MSE 0.0606832.
- BBLR_MatchedZero: Simpson secant MSE 0.0501972; midpoint secant MSE 0.050272; equal-location Simpson MSE 0.0596046.
- Hinge: Simpson secant MSE 0.0506204; midpoint secant MSE 0.0519251; equal-location Simpson MSE 0.0629539.
- BBLR: Simpson secant MSE 0.0508334; midpoint secant MSE 0.0508307; equal-location Simpson MSE 0.0598467.
- LLR: Simpson secant MSE 0.0510016; midpoint secant MSE 0.0515242; equal-location Simpson MSE 0.0581881.

## Week-block coefficient stability

- BBLR: median heating-to-cooling crossing 23.4 °C (2.5--97.5% 22.1--26.2; available in 1.000 of resamples); P[g(18)<0]=1.000; P[g(28)>0]=0.996; median total variation 0.7239.
- LLR: median heating-to-cooling crossing 22.6 °C (2.5--97.5% 20.6--25.3; available in 1.000 of resamples); P[g(18)<0]=1.000; P[g(28)>0]=0.997; median total variation 2.884.
- LP2: median heating-to-cooling crossing 23.2 °C (2.5--97.5% 22.5--24.3; available in 1.000 of resamples); P[g(18)<0]=1.000; P[g(28)>0]=1.000; median total variation 0.7439.
- PSpline: median heating-to-cooling crossing 22.9 °C (2.5--97.5% 22.1--24.3; available in 1.000 of resamples); P[g(18)<0]=1.000; P[g(28)>0]=1.000; median total variation 0.4636.
- Hinge: median heating-to-cooling crossing 27.9 °C (2.5--97.5% 27.9--27.9; available in 1.000 of resamples); P[g(18)<0]=1.000; P[g(28)>0]=1.000; median total variation 0.4221.

## Interpretation

Held-out secant losses are noisy empirical implications of the coefficient field, not observations of derivative truth. The Monte Carlo sections use direct derivative MSE because truth is known there; this audit exists only because real-data derivative truth is latent. Stability, smoothness, and agreement are reported separately from held-out contrast accuracy.
