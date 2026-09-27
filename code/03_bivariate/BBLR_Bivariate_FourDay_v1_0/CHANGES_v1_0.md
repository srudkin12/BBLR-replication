# BBLR Bivariate Four-Day v1.0

- Native two-dimensional BBLR with exact `epsilon=0` nesting of fixed-radius local linear regression.
- Exact analytical gradients for 16 bivariate DGPs, verified against finite differences.
- 18 support designs including soft sparse bands, hard gaps, central holes, correlated support, separated clusters and near-manifold support.
- Aggregate gradient RMSE plus component-specific slope RMSE.
- Repeated outcome-CV tuning; simulation gradients are evaluation-only.
- Strong comparator set: fixed/adaptive local polynomial, tensor spline, global linear/quadratic, and targeted componentwise tensor spline.
- 58-cell broad operating panel with noise/sample-size scaling and support/function alignment tests.
- Resumable shard-based HPC execution and compact handbacks.
- Three-stage unattended four-day runner so completed evidence is preserved even if later stages are still running.
