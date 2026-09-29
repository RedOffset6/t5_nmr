# Load configs/train/$RUN.env and fill in the defaults every job needs.
# Sourced from the repository root by submit.sh and the Slurm jobs.

if [ -z "${RUN:-}" ]; then
  echo "RUN is not set" >&2
  exit 1
fi

RUN_CONFIG="configs/train/$RUN.env"
if [ ! -f "$RUN_CONFIG" ]; then
  echo "No config $RUN_CONFIG; available runs:" >&2
  ls configs/train | sed 's/\.env$//' >&2
  exit 1
fi

# Every variable in the config is exported to t5_train.py.
set -a
source "$RUN_CONFIG"
set +a

export NODES="${NODES:-1}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
export DATA_DIR="${DATA_DIR:-alberts_2d}"

# Output folders are named after the run, except for runs made before this
# layout, whose configs set OUTPUT_DIR to the existing folder.
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/$RUN}"

# A smoke test (MAX_STEPS set at submission) gets its own folder, so it never
# resumes from, or is resumed by, the real run.
if [ "${MAX_STEPS:-0}" -gt 0 ] && [ "${OUTPUT_DIR%_max*steps}" = "$OUTPUT_DIR" ]; then
  export OUTPUT_DIR="${OUTPUT_DIR}_max${MAX_STEPS}steps"
fi
