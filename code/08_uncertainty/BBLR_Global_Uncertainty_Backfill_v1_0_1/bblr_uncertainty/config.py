from __future__ import annotations
import os

# Avoid nested BLAS/OpenMP parallelism on the 80-core workstation.
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, "1")

QUANTILES = [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]
METHOD_ALIASES = {
    "BALL-BASED LINEAR REGRESSION": "BBLR", "BALL BASED LINEAR REGRESSION": "BBLR",
    "BBLR": "BBLR", "LLR": "LLR", "AKLLR": "AKLLR", "AKLP2": "AKLP2",
    "SPLINE": "SPLINE", "CSPLINE": "CSPLINE", "CAKLLR": "CAKLLR", "CAKLP2": "CAKLP2",
    "LP2": "LP2", "OLS": "OLS", "RIDGE": "RIDGE", "LASSO": "LASSO",
    "LL-ALASSO": "LL-ALASSO", "LL_ALASSO": "LL-ALASSO", "W": "W", "MATCHED_W": "W",
    "WSTRICT": "WSTRICT", "WSAFE": "WSAFE", "WNORMAL": "WNORMAL", "WCENTER": "WCENTER",
    "SELECTED-REPEATED-CV": "BBLR", "LLR-SELECTED-REPEATED-CV": "LLR",
}
CELL_SYNONYMS = ["cell_id", "cell", "design_cell", "scenario_id", "scenario", "label", "landmark_label", "sigma", "noise_sd"]
REP_SYNONYMS = ["rep", "replication", "replicate", "rep_id", "replication_id", "simulation", "sim", "seed_rep"]
METHOD_SYNONYMS = ["method", "estimator", "selector", "algorithm", "model"]
METRIC_RE = r"(?i)(rmse|mse|mae|error|loss|regret|risk|bias|variance|var_|_var|alpha|gradient|slope|intercept|coefficient|nuisance|joint)"
EXCLUDE_METRIC_RE = r"(?i)(^n_|_n$|count|share|radius|^r$|eps|epsilon|lambda|penalty|fold|seed|time|seconds|dim|dimension|active_dim|noise_sd|temperature|rank|id$)"
