#!/bin/bash
# Checks that torchrun and NCCL work on all GPUs of the job before training.
# One node: sbatch test_multi_gpu.sh
# Two nodes (Slingshot network check): sbatch --nodes=2 test_multi_gpu.sh
#SBATCH --job-name=test_multi_gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --time=00:10:00

set -e

module load cuda/12.6

source /home/b5an/jucloud.b5an/miniforge3/bin/activate
conda activate t5

export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO

if [ "$SLURM_JOB_NUM_NODES" -gt 1 ]; then
  # Same Slingshot settings as start_training_xxl_2x1x8_10ep.sh.
  module load brics/nccl brics/aws-ofi-nccl
  export NCCL_NET="AWS Libfabric"
  export NCCL_SOCKET_IFNAME=hsn
  export NCCL_NET_GDR_LEVEL=PHB
  export NCCL_CROSS_NIC=1
  export NCCL_MIN_NCHANNELS=4
  export NCCL_GDRCOPY_ENABLE=1
  export NCCL_NET_FORCE_FLUSH=0
  export FI_CXI_DEFAULT_CQ_SIZE=131072
  export FI_CXI_DEFAULT_TX_SIZE=2048
  export FI_HMEM_CUDA_USE_GDRCOPY=1
  export FI_CXI_RDZV_PROTO=alt_read
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
