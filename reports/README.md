# Evaluation Reports

The raw evaluation logs are excluded from Git because they are large generated files.

- evaluation_summary.tsv combines chunks using a sample-count-weighted exact-match score. `scripts/combine_results.py <run>` adds or replaces a run's row, named after the run, with the evaluation's Slurm job ID and the date. `exact_match` is top-1; `top_2` to `top_10` are the top-n scores, blank for older rows and for runs with fewer outputs per spectrum. Older rows are named after the Slurm evaluation job and have no date.
- evaluation_chunks.tsv contains metrics extracted from each chunk of the older evaluations, keyed by log file. The `eval_samll_512` log files really have that name; the summary row is spelled `eval_small_512`.
- eval_check_5928345 is a 10-sample smoke test, not a full evaluation.
- Full evaluations contain 79,441 test samples.
- dataset_stats.json, when present, is written by `scripts/dataset_stats.py`.
