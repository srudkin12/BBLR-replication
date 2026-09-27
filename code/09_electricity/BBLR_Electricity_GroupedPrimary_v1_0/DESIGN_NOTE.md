# Grouped-week primary empirical refresh

Purpose: align the displayed electricity-demand coefficient curves and block-bootstrap uncertainty with the dependence-aware calendar-week-grouped CV specification requested by the v0.17 adversarial JoE review.

No tuning is redone. Parameters are frozen from the completed 1,000-repetition grouped-week CV audit:

- Peak: BBLR (r=3, eps=0); LLR r=3; LP2 r=8; PSpline (5,0.1); hinge (20,28).
- 18:00: BBLR (r=5, eps=4); LLR r=3; LP2 r=8; PSpline (4,0.01); hinge (20,28).

The package refits each frozen method to the full workday sample and runs 1,000 calendar-week block-bootstrap replications for coefficient uncertainty. It does not search over outcomes, clock times, temperature summaries, or tuning parameters.
