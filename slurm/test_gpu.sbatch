#!/bin/bash
# Prints GPU and PyTorch CUDA information for a node: ./submit.sh test-gpu
#SBATCH --job-name=test_gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8GB
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1

cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
setup_job_env

echo "===== Hostname ====="
hostname

echo "===== CUDA_VISIBLE_DEVICES ====="
echo $CUDA_VISIBLE_DEVICES

echo "===== nvidia-smi ====="
nvidia-smi

echo "===== Torch check ====="
python -c "import torch; print('torch version:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
