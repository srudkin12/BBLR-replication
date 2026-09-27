# Four-day unattended run

The recommended production command uses 5-fold CV repeated 5 times for all methods and 140 outer workers.

The staged shell runner first completes 250 reps/cell, then extends the same run to 500, 750 and 1000 reps/cell using `--resume`. After every completed stage it writes summaries and a compact handback. If the final stage is still running when you return, the most recent completed-stage handback remains available and all later shards are resumable.
