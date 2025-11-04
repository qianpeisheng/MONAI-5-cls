#!/usr/bin/env bash
set -euo pipefail

# Launch the two experiments concurrently:
# - scratch variant on GPU 0
# - pretrained variant on GPU 1

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

# Allow overrides
GPU_SCRATCH=${GPU_SCRATCH:-0}
GPU_PRETRAIN=${GPU_PRETRAIN:-1}

# Start immediately unless overridden
export DELAY_SECONDS=${DELAY_SECONDS:-0}

echo "Launching scratch (GPU ${GPU_SCRATCH}) and pretrained (GPU ${GPU_PRETRAIN}) runs in parallel..."

(
  export GPU_ID=${GPU_SCRATCH}
  bash ./run_train_pseudo_vs_points.sh
) &

(
  export GPU_ID=${GPU_PRETRAIN}
  bash ./run_train_pseudo_vs_points_pretrained.sh
) &

wait
echo "Both parallel runs finished."

