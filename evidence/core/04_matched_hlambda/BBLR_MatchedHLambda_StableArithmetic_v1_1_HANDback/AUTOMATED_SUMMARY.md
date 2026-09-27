# Stable-arithmetic matched-H-lambda audit

- Analysis audit: **CHECK**.
- Normal-form counts: {'BBLR_better': 73, 'W_better': 14, 'unclear': 5}.
- Centered-form counts: {'BBLR_better': 73, 'W_better': 14, 'unclear': 5}.
- Maximum per-replication max absolute test-slope difference: 0.00514806.
- 99th percentile of that diagnostic: 5.37823e-11.

## Theory-facing aggregates
- normal, all_positive: ratio=0.0370, win=0.672, delta=-1.13101 [-1.78917,-0.676994], BBLR_better.
- normal, lambda_eq1: ratio=0.6811, win=0.765, delta=-0.022615 [-0.0231455,-0.0221267], BBLR_better.
- normal, lambda_lt1: ratio=0.8679, win=0.508, delta=-0.00523901 [-0.00547682,-0.00499655], BBLR_better.
- normal, lambda_gt1: ratio=0.0027, win=0.959, delta=-23.4256 [-38.3474,-13.9253], BBLR_better.
- normal, absGamma_lt_0p01: ratio=0.7062, win=0.715, delta=-0.0183503 [-0.0187821,-0.0179325], BBLR_better.
- centered, all_positive: ratio=0.0370, win=0.672, delta=-1.13101 [-1.86585,-0.675126], BBLR_better.
- centered, lambda_eq1: ratio=0.6811, win=0.765, delta=-0.022615 [-0.0231016,-0.0220903], BBLR_better.
- centered, lambda_lt1: ratio=0.8679, win=0.508, delta=-0.00523901 [-0.0054938,-0.00499517], BBLR_better.
- centered, lambda_gt1: ratio=0.0027, win=0.959, delta=-23.4256 [-37.3069,-13.7149], BBLR_better.
- centered, absGamma_lt_0p01: ratio=0.7062, win=0.715, delta=-0.0183503 [-0.018774,-0.0179312], BBLR_better.
