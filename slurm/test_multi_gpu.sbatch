#!/bin/bash
# Checks that NCCL works on all GPUs of the job before training, with the
# same layout as training: one Slurm task per GPU.
# One node: ./submit.sh test-multi-gpu
# Two nodes (Slingshot network check): ./submit.sh test-multi-gpu --nodes=2
#SBATCH --job-name=test_multi_gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --time=00:10:00

set -e

cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
setup_job_env

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export OMP_NUM_THREADS=8
export WORLD_SIZE="$SLURM_NTASKS"
export NCCL_DEBUG=INFO
# Only the lines that say which transport NCCL chose.
export NCCL_DEBUG_SUBSYS=INIT,NET

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

# A task sees either all of the node's GPUs (then LOCAL_RANK picks one) or
# only its own (then it is device 0), as in t5_train.py.
visible = torch.cuda.device_count()
local_rank = int(os.environ["LOCAL_RANK"]) if visible > 1 else 0
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
rank, world_size = dist.get_rank(), dist.get_world_size()

print(
    f"rank {rank}/{world_size} on {os.uname().nodename}: sees {visible} GPU(s), "
    f"uses GPU {local_rank} ({torch.cuda.get_device_name(local_rank)}), "
    f"CPUs {sorted(os.sched_getaffinity(0))[0]}-{sorted(os.sched_getaffinity(0))[-1]}",
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
    if visible == 1:
        print("Each rank sees only its own GPU: hsdp will not work, use fsdp or ddp")
    print(f"1 GiB all_reduce: {elapsed * 1000:.1f} ms, bus bandwidth {bus_bandwidth:.1f} GB/s")

dist.destroy_process_group()
EOF

echo "===== all-reduce, one task per GPU ====="
srun --cpu-bind=cores bash -c '
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  exec python -u "$0"
' "$SLURM_SUBMIT_DIR/tmp/allreduce_$SLURM_JOB_ID.py"

echo "===== NCCL transport ====="
echo "Between nodes, look above for 'NET/AWS Libfabric' (Slingshot) rather than 'NET/Socket' (TCP)."
