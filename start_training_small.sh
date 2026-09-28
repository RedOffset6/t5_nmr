#!/bin/bash

#SBATCH --output=logs/%x_%j.out
#SBATCH --job-name=flan_t5_small_1536
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1

# Exit immediately if a command fails.
set -e

# Make Python output appear in the Slurm log immediately.
export PYTHONUNBUFFERED=1


# ===== Move to submission directory =====

cd "$SLURM_SUBMIT_DIR"


# ===== Experiment configuration =====

export MODEL_NAME="google/flan-t5-small"
export INPUT_MAX_LENGTH=1536
export TARGET_MAX_LENGTH=128

export TRAIN_BATCH_SIZE=4
export GRAD_ACCUMULATION_STEPS=4

# 4 × 4 × 1 GPU = effective batch size 16.
export EVAL_BATCH_SIZE=8
export GENERATION_BATCH_SIZE=8

export LEARNING_RATE="5e-5"
export WEIGHT_DECAY="0.01"
export NUM_EPOCHS=3
export SEED=42

export SAVE_STEPS=2000
export SAVE_TOTAL_LIMIT=2

# Start with gradient checkpointing disabled.
export GRADIENT_CHECKPOINTING=0

export TEST_SAMPLE_SIZE=1000
export GENERATION_MAX_NEW_TOKENS=128

# Explicit output directory prevents accidental checkpoint reuse.
export OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs/flan-t5-large_nmr_input1536_bs16"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29600
export WORLD_SIZE=$SLURM_NTASKS

echo "MASTER_ADDR=$MASTER_ADDR"
echo "WORLD_SIZE=$WORLD_SIZE"




# ===== Slurm information =====

echo "===== Slurm Job Info ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Submission directory: $SLURM_SUBMIT_DIR"
echo "Working directory: $(pwd)"


# ===== Environment activation =====

echo "===== Environment Before Activation ====="
echo "Python before activation: $(which python || true)"

module load cuda/12.6

source /home/b5an/jucloud.b5an/miniforge3/bin/activate
conda activate t5

echo "===== Environment After Activation ====="
echo "Python executable: $(which python)"
python --version


# ===== GPU information =====

echo "===== GPU Info ====="
nvidia-smi


# ===== Selected training configuration =====

echo "===== Shell Training Configuration ====="
echo "Model: $MODEL_NAME"
echo "Input max length: $INPUT_MAX_LENGTH"
echo "Train batch size per device: $TRAIN_BATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACCUMULATION_STEPS"
echo "Effective batch size: $((TRAIN_BATCH_SIZE * GRAD_ACCUMULATION_STEPS))"
echo "Evaluation batch size: $EVAL_BATCH_SIZE"
echo "Generation batch size: $GENERATION_BATCH_SIZE"
echo "Gradient checkpointing: $GRADIENT_CHECKPOINTING"
echo "Save steps: $SAVE_STEPS"
echo "Output directory: $OUTPUT_DIR"


# ===== Start training =====

echo "===== Start Training ====="

srun --cpu-bind=none bash -c '
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
python -u t5_train.py

echo "===== Job Finished Successfully ====="
echo "End time: $(date)"