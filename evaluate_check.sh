#!/bin/bash

#SBATCH --job-name=eval_check
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1

set -e
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

module load cuda/12.6
source /home/b5an/jucloud.b5an/miniforge3/bin/activate
conda activate t5

nvidia-smi

python evaluate_exact_match.py \
  --model-path outputs/flan-t5-xl_nmr_input1024_bs16/final_model \
  --data-dir alberts_2d \
  --split test \
  --input-max-length 1024 \
  --batch-size 2 \
  --start-index 0 \
  --sample-size 10 \
  --output-file outputs/flan-t5-xl_nmr_input1024_bs16/test_check.json \
  --local-files-only