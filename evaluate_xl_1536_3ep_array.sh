#!/bin/bash

#SBATCH --job-name=eval_xl_1536_3ep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-3

set -e
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

module load cuda/12.6
source /home/b5an/jucloud.b5an/miniforge3/bin/activate
conda activate t5

CHUNK_SIZE=20000
START_INDEX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))

echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Start index: $START_INDEX"
echo "Chunk size: $CHUNK_SIZE"

python evaluate_exact_match.py \
  --model-path outputs/flan-t5-xl_nmr_input1536_bs4x4_3ep/final_model \
  --data-dir alberts_2d \
  --split test \
  --input-max-length 1536 \
  --batch-size 2 \
  --start-index "$START_INDEX" \
  --sample-size "$CHUNK_SIZE" \
  --local-files-only