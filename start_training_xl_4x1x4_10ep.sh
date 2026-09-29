#!/bin/bash

#SBATCH --output=logs/%x_%j.out
#SBATCH --job-name=flan_t5_xl_4x1x4_10ep
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# All 4 GH200s of the node, so the job never shares a node.
#SBATCH --gres=gpu:4
# One torchrun per node owns all 4 Grace CPUs (4 x 72 cores).
#SBATCH --cpus-per-task=288
# Whole-node host memory: rank 0 holds the fp32 XXL weights while loading
# and saving, and every rank memory-maps the tokenized dataset.
#SBATCH --mem=0
#SBATCH --time=1-00:00:00

# Exit immediately if a command fails.
set -e

# Make Python output appear in the Slurm log immediately.
export PYTHONUNBUFFERED=1


# ===== Move to submission directory =====

cd "$SLURM_SUBMIT_DIR"


# ===== Experiment configuration =====

export MODEL_NAME="google/flan-t5-xl"
export TARGET_MAX_LENGTH=128

export TRAIN_BATCH_SIZE=4
export GRAD_ACCUMULATION_STEPS=1

# 4 × 1 × 4 GPUs = effective batch size 16.
export EVAL_BATCH_SIZE=8
export GENERATION_BATCH_SIZE=8

export LEARNING_RATE="5e-5"
export WEIGHT_DECAY="0.01"
export NUM_EPOCHS=10
export SEED=42

export SAVE_STEPS=2000
export SAVE_TOTAL_LIMIT=2

export GRADIENT_CHECKPOINTING=0

export TEST_SAMPLE_SIZE=1000
export GENERATION_MAX_NEW_TOKENS=128

# Explicit output directory prevents accidental checkpoint reuse.
export OUTPUT_DIR="$SLURM_SUBMIT_DIR/outputs/flan-t5-xl_nmr_input1536_4x1x4_ep10"


# ===== Multi-GPU configuration =====

# ddp for small/base/large, fsdp for xl/xxl.
export PARALLEL_MODE=fsdp
export GPUS_PER_NODE=4
export DATALOADER_NUM_WORKERS=4

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
# INFO for the first multi-node run.
export NCCL_DEBUG=WARN


# ===== Slurm information =====

echo "===== Slurm Job Info ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NODELIST ($SLURM_JOB_NUM_NODES)"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
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
echo "Parallel mode: $PARALLEL_MODE"
echo "GPUs: $((GPUS_PER_NODE * SLURM_JOB_NUM_NODES))"
echo "Train batch size per device: $TRAIN_BATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACCUMULATION_STEPS"
echo "Effective batch size: $((TRAIN_BATCH_SIZE * GRAD_ACCUMULATION_STEPS * GPUS_PER_NODE * SLURM_JOB_NUM_NODES))"
echo "Evaluation batch size: $EVAL_BATCH_SIZE"
echo "Generation batch size: $GENERATION_BATCH_SIZE"
echo "Gradient checkpointing: $GRADIENT_CHECKPOINTING"
echo "Save steps: $SAVE_STEPS"
echo "Output directory: $OUTPUT_DIR"


# ===== Start training =====

echo "===== Start Training ====="

srun --ntasks-per-node=1 --cpu-bind=none \
  torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc-per-node="$GPUS_PER_NODE" \
    --rdzv-id="$SLURM_JOB_ID" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    t5_train.py

echo "===== Job Finished Successfully ====="
echo "End time: $(date)"
