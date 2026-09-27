# BBLR Power Analysis v1.4

## Objective

Quantify the power of BBLR without broadening the DGP catalogue. The stage separates three questions:

1. **Does Stage 2 add value?** BBLR now contains exact `epsilon=0`, so `BBLR(r,0)=LLR(r)`.
2. **Can outcome information select useful coefficient smoothing?** Feasible minimum-Y, one-SE/low-smoothing, 0.5%-tolerance and 1%-tolerance selectors are compared with the simulation-only slope oracle.
3. **Is the support-regime advantage merely failure of global/support-unaware comparators?** Componentwise spline, adaptive local-linear and adaptive local-quadratic methods infer support components from training X only and outcome-tune the gap rule.

## Support-component inference

Adjacent sorted-X gaps are compared with the median positive spacing. Gap factors `16,20,30,inf` are candidates; `inf` means no split. A split is allowed only when all resulting components contain at least 15 observations. The gap factor is not informed by DGP truth.

## Two-stage workflow

### A. Existing-shard intensive analysis
Run `scripts/analyze_bblr_power_existing.py` against the authoritative `support_regime_v1` shard directory. This requires no new Monte Carlo and examines the complete historical BBLR tuning clouds.

### B. Landmark support-aware simulation
Run only the 12 cells in `POWER_LANDMARK_CELLS.csv`. These deliberately span spline-favourable, boundary, BBLR-favourable, soft-support and hard-gap regimes. The first run fixes n=500 and sigma=0.25 so differences are interpretable.

## Key outputs

- validation-Y versus slope-error correlation within BBLR clouds;
- prediction-flatness and latent slope opportunity;
- feasible low-smoothing selector regret;
- exact Stage-2 gain relative to epsilon=0/LLR;
- BBLR versus support-aware spline/AKLLR/AKLP2;
- inferred support-component diagnostics.
