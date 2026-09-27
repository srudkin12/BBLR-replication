# Strong-comparator production run

Default production target:

- 27 frozen cells
- 1,000 replications per cell
- 5-fold CV x 5 repeats for every method
- 140 workers
- 5 replications per shard

Expected Monte Carlo replications: 27,000.
Expected chunks: 5,400.

Run `scripts/run_strong_comparator_rcv.py`, then
`scripts/summarize_strong_comparator_rcv.py`, then
`scripts/make_strong_comparator_handoff.py`.
