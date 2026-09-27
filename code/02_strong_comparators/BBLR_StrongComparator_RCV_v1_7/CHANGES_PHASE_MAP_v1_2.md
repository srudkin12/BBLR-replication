# BBLR Univariate Phase Map v1.2 — changes

- Adds 36 controlled smooth DGPs across six parameterised families plus U01.
- Separates curvature, slope frequency, slope amplitude, transition sharpness,
  local-feature width, and local-feature location.
- Adds DGP family/parameter metadata to every result row.
- Adds richer shape diagnostics: slope RMS/max, curvature RMS/max, roughness
  length, mean range/SD, and slope total variation.
- Broadens BBLR r/epsilon grids through global-scale smoothing.
- Broadens smoothing-spline lambda grid.
- Extends summarisation to retain compact slope-oracle, mean-oracle and Pareto
  datasets without requiring raw-shard transfer.
- Adds phase-map analysis outputs for pairwise performance, prediction-selection
  slope regret, tuning-boundary audits and family-level summaries.
- Integrates compact and targeted handback tools.
- Adds four controlled-family unit tests.
- Comparator set deliberately unchanged for this DGP-mapping phase.
