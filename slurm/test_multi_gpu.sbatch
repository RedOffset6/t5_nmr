#!/bin/bash
# Checks that torchrun and NCCL work on all GPUs of the job before training.
# One node: ./submit.sh test-multi-gpu
# Two nodes (Slingshot network check): ./submit.sh test-multi-gpu --nodes=2
#SBATCH --job-name=test_multi_gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --time=00:10:00

set -e

cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
setup_job_env

export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO

if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
  # The same Slingshot settings as two-node training.
  setup_multinode_nccl
fi

echo "===== Nodes ====="
echo "$SLURM_JOB_NODELIST ($SLURM_JOB_NUM_NODES), master $MASTER_ADDR:$MASTER_PORT"

echo "===== nvidia-smi ====="
nvidia-smi

# On shared storage, so every node of the job can read it.
mkdir -p "$SLURM_SUBMIT_DIR/tmp"
cat > "$SLURM_SUBMIT_DIR/tmp/allreduce_$SLURM_JOB_ID.py" <<'EOF'
import os
import time

import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
rank, world_size = dist.get_rank(), dist.get_world_size()

print(
    f"rank {rank}/{world_size} on {os.uname().nodename}, "
    f"GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}",
    flush=True,
)

# Correctness: the sum of all ranks' (rank + 1) is world_size * (world_size + 1) / 2.
value = torch.full((1,), rank + 1.0, device="cuda")
dist.all_reduce(value)
expected = world_size * (world_size + 1) / 2
assert value.item() == expected, f"all_reduce gave {value.item()}, expected {expected}"

# Bandwidth: 1 GiB fp32 all-reduce, reported as NCCL-tests "bus bandwidth".
tensor = torch.ones(256 * 1024 * 1024, device="cuda")
for _ in range(3):
    dist.all_reduce(tensor)
torch.cuda.synchronize()
iterations = 10
start = time.perf_counter()
for _ in range(iterations):
    dist.all_reduce(tensor)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / iterations
size_gb = tensor.numel() * tensor.element_size() / 1e9
bus_bandwidth = size_gb / elapsed * 2 * (world_size - 1) / world_size

if rank == 0:
    print("torch:", torch.__version__, "NCCL:", ".".join(map(str, torch.cuda.nccl.version())))
    print("all_reduce OK")
    print(f"1 GiB all_reduce: {elapsed * 1000:.1f} ms, bus bandwidth {bus_bandwidth:.1f} GB/s")

dist.destroy_process_group()
EOF

echo "===== torchrun all-reduce ====="
srun --ntasks-per-node=1 --cpu-bind=none \
  torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc-per-node="$GPUS_PER_NODE" \
    --rdzv-id="$SLURM_JOB_ID" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$SLURM_SUBMIT_DIR/tmp/allreduce_$SLURM_JOB_ID.py"
