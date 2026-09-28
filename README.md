# NMR-to-SMILES with FLAN-T5

This repository contains training and evaluation code for predicting molecular SMILES strings from NMR-derived input sequences using FLAN-T5 models.

## Repository contents

- `t5_train.py`: shared training pipeline.
- `t5_base.py`, `t5_large.py`, `t5_xl.py`, `t5_nmr.py`: model-specific or earlier training implementations.
- `start_training_*.sh`: Slurm configurations for different model sizes, input lengths, batch sizes, and epoch counts.
- `evaluate_exact_match.py`: exact-match evaluation program.
- `evaluate_*.sh`: Slurm array jobs for model evaluation.
- `reports/`: chunk-level and aggregated exact-match results.
- `data/manifest.tsv`: dataset file sizes, line counts, and SHA-256 checksums.

## Environment

Create the Conda environment:

    conda env create -f environment.yml
    conda activate t5

The Slurm scripts contain cluster-specific CUDA modules, resource requests, and environment paths. Review them before running on another system.

## Data

The full datasets are excluded from Git because of their size and possible redistribution restrictions.

Expected local directories:

    alberts_2d/
    nmr_expt_data/

See `data/README.md` and `data/manifest.tsv` for split structure and integrity checksums.

## Training

Submit one training configuration through Slurm, for example:

    sbatch start_training_xl_1536_4x4_10ep.sh

Generated checkpoints are written under `outputs/` or `results*/` and are intentionally excluded from Git.

## Evaluation

Submit the corresponding evaluation array, for example:

    sbatch evaluate_xl_1536_10ep_array.sh

Raw Slurm logs are excluded from Git. Compact exact-match summaries are available in `reports/evaluation_summary.tsv`.

## Results

Full evaluations contain 79,441 test samples. The summary table combines array-job chunks using sample-count-weighted exact-match scores.

`eval_check_5928345` is a 10-sample smoke test and should not be compared with full evaluations.
