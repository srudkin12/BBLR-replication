# Changes — Strong Comparator Repeated CV v1.7

- Adds a 27-cell frozen adversarial panel from the v1.6 operating map.
- Adds outcome-only 5x5 repeated-CV tuning for all strong comparators.
- Reuses optimized BBLR repeated-CV evaluation; LLR remains the exact epsilon=0 subset.
- Adds adaptive local-linear/local-quadratic, spline, and support-aware versions under the same CV folds.
- Refits only outcome-selected configurations for truth evaluation, avoiding unnecessary oracle fitting of every comparator configuration.
- Adds paired BBLR-versus-comparator cell summaries and Pareto summaries.
- Adds compact strong-comparator handback creation.
- Adds tests for exact BBLR/LLR CV nesting, complete comparator registry, and finite selected evaluation.
