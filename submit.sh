#!/bin/bash
# Submits a Slurm job for a run described by configs/train/<run>.env.
#
#   ./submit.sh train <run> [sbatch options]      train (resubmits itself until done)
#   ./submit.sh evaluate <run> [sbatch options]   full test-set evaluation (array job)
#   ./submit.sh check <run> [sbatch options]      evaluate 10 molecules
#   ./submit.sh test-gpu [sbatch options]         GPU and PyTorch check
#   ./submit.sh test-multi-gpu [sbatch options]   NCCL check on 4 GPUs (--nodes=2 for two nodes)
#
# Variables not set in the config can be given at submission, for example a
# smoke test: MAX_STEPS=50 ./submit.sh train xxl_2x2x4_10ep

set -e

cd "$(dirname "$0")"
source slurm/env.sh
mkdir -p logs

usage() {
  sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//' >&2
  exit 1
}

[ $# -ge 1 ] || usage
job="$1"
shift

case "$job" in
  test-gpu)
    exec sbatch "$@" slurm/test_gpu.sbatch
    ;;
  test-multi-gpu)
    exec sbatch "$@" slurm/test_multi_gpu.sbatch
    ;;
  train|evaluate|check)
    [ $# -ge 1 ] || usage
    export RUN="$1"
    shift
    source slurm/load_run.sh
    ;;
  *)
    usage
    ;;
esac

case "$job" in
  train)
    gpus="$GPUS_PER_NODE"
    if [ $((NODES * gpus)) -gt 1 ]; then
      # Whole-node memory: rank 0 holds the fp32 weights while loading and
      # saving, and every rank memory-maps the tokenized dataset.
      default_mem=0
    else
      default_mem=64G
    fi
    exec sbatch \
      --job-name="train_$RUN" \
      --nodes="$NODES" \
      --ntasks-per-node=1 \
      --gres="gpu:$gpus" \
      --cpus-per-task=$((CPUS_PER_GPU * gpus)) \
      --mem="${TRAIN_MEM:-$default_mem}" \
      --time="${TRAIN_TIME:-1-00:00:00}" \
      --export=ALL \
      "$@" slurm/train.sbatch
    ;;
  evaluate)
    chunks="${EVAL_CHUNKS:-4}"
    exec sbatch \
      --job-name="eval_$RUN" \
      --array="0-$((chunks - 1))" \
      --nodes=1 \
      --ntasks-per-node=1 \
      --gres=gpu:1 \
      --cpus-per-task="$CPUS_PER_GPU" \
      --mem="${EVAL_MEM:-64G}" \
      --time="${EVAL_TIME:-10:00:00}" \
      --export=ALL \
      "$@" slurm/evaluate.sbatch
    ;;
  check)
    exec sbatch \
      --job-name="check_$RUN" \
      --nodes=1 \
      --ntasks-per-node=1 \
      --gres=gpu:1 \
      --cpus-per-task=8 \
      --mem=32G \
      --time=00:20:00 \
      --export=ALL \
      "$@" slurm/evaluate_check.sbatch
    ;;
esac
