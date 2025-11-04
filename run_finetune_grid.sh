#!/usr/bin/env bash
# Schedule 6 WP5 fine-tune experiments on 2 GPUs:
#  - init in {scratch, pretrained}
#  - subset_ratio in {1.0 (100%), 0.1 (10%), 0.01 (1%)}
# Runs sequentially per GPU and in parallel across 2 GPUs.
#
# Usage:
#   bash run_finetune_grid.sh /path/to/pretrained.ckpt
# or set env vars:
#   PRETRAINED_CKPT=/path/to/pretrained.ckpt bash run_finetune_grid.sh
#
# Optional env overrides:
#   DATA_ROOT=/data3/wp5/wp5-code/dataloaders/wp5-dataset \
#   SPLIT_CFG=/data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json \
#   GPUS="0 1" EPOCHS=100 BATCH_SIZE=2 LR=1e-3 NUM_WORKERS=4 SEED=42 bash run_finetune_grid.sh

set -euo pipefail

# Inputs and defaults
DATA_ROOT=${DATA_ROOT:-/data3/wp5/wp5-code/dataloaders/wp5-dataset}
SPLIT_CFG=${SPLIT_CFG:-/data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json}
GPUS=${GPUS:-"0 1"}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-2}
LR=${LR:-1e-3}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-42}

SCRIPT=${SCRIPT:-train_finetune_wp5.py}
if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: $SCRIPT not found in current directory." >&2
  exit 1
fi

# Pretrained ckpt path from arg or env; if missing, try to download and then discover
DEFAULT_PRETRAINED_DIR=${DEFAULT_PRETRAINED_DIR:-pretrained_models/spleen_ct_segmentation}
DEFAULT_PRETRAINED_CKPT="$DEFAULT_PRETRAINED_DIR/models/model.pt"

# Allow override by env or arg (won't be final if we discover a different model.pt later)
PRETRAINED_CKPT=${PRETRAINED_CKPT:-${1:-$DEFAULT_PRETRAINED_CKPT}}

if [[ ! -f "$PRETRAINED_CKPT" ]]; then
  echo "Attempting to download MONAI bundle 'spleen_ct_segmentation' to $DEFAULT_PRETRAINED_DIR ..."
  mkdir -p "$DEFAULT_PRETRAINED_DIR"
  # Try a lightweight Python call to download the bundle
  DEFAULT_PRETRAINED_DIR="$DEFAULT_PRETRAINED_DIR" python - <<'PY'
import sys, os
try:
    from monai.bundle import download
    bundle_dir = os.environ.get('DEFAULT_PRETRAINED_DIR','pretrained_models/spleen_ct_segmentation')
    download(name='spleen_ct_segmentation', bundle_dir=bundle_dir)
    print('Downloaded MONAI bundle to', bundle_dir)
except Exception as e:
    print('Bundle download failed:', e)
    sys.exit(0)
PY
  # Re-check exact path, or search for any model.pt under the bundle dir
  if [[ -f "$DEFAULT_PRETRAINED_CKPT" ]]; then
    PRETRAINED_CKPT="$DEFAULT_PRETRAINED_CKPT"
    echo "Using downloaded checkpoint: $PRETRAINED_CKPT"
  else
    found_ckpt=$(find "$DEFAULT_PRETRAINED_DIR" -type f -name 'model.pt' | head -n1 || true)
    if [[ -n "$found_ckpt" ]]; then
      PRETRAINED_CKPT="$found_ckpt"
      echo "Using discovered checkpoint: $PRETRAINED_CKPT"
    else
      echo "WARNING: Could not locate a pretrained checkpoint after download attempt."
      echo "         Pretrained runs will be skipped unless you set PRETRAINED_CKPT to a valid path."
    fi
  fi
fi

# Derive bundle dir from checkpoint path (two directories up: <bundle_dir>/models/model.pt)
if [[ -f "$PRETRAINED_CKPT" ]]; then
  BUNDLE_DIR="$(dirname "$(dirname "$PRETRAINED_CKPT")")"
  echo "Using bundle dir: $BUNDLE_DIR"
fi

SUBSETS=("1.0" "0.1" "0.01")

tag_for_ratio() {
  case "$1" in
    1.0) echo "100" ;;
    0.1) echo "10" ;;
    0.01) echo "1" ;;
    *) echo "custom" ;;
  esac
}

run_chain() {
  local gpu="$1"; shift
  local init="$1"; shift

  export CUDA_VISIBLE_DEVICES="$gpu"
  echo "[GPU $gpu] Starting chain: init=$init"

  # Skip pretrained chain if checkpoint is unavailable
  if [[ "$init" == "pretrained" ]] && [[ ! -f "$PRETRAINED_CKPT" ]]; then
    echo "[GPU $gpu] Skipping pretrained chain: checkpoint not found at $PRETRAINED_CKPT"
    return 0
  fi

  for r in "${SUBSETS[@]}"; do
    local tag; tag=$(tag_for_ratio "$r")
    local out="runs/grid/${init}_subset_${tag}"
    mkdir -p "$out"
    echo "[GPU $gpu] Running: subset_ratio=$r -> $out"

    if [[ "$init" == "pretrained" ]]; then
      python -u "$SCRIPT" \
        --mode train \
        --data_root "$DATA_ROOT" \
        --split_cfg "$SPLIT_CFG" \
        --output_dir "$out" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --lr "$LR" \
        --lr_ft 3e-4 \
        --subset_ratio "$r" \
        --seed "$((SEED+1))" \
        --net basicunet \
        ${BUNDLE_DIR:+--bundle_dir "$BUNDLE_DIR"} \
        --init pretrained \
        --pretrained_ckpt "$PRETRAINED_CKPT" | tee "$out/train.log"
    else
      python -u "$SCRIPT" \
        --mode train \
        --data_root "$DATA_ROOT" \
        --split_cfg "$SPLIT_CFG" \
        --output_dir "$out" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --lr "$LR" \
        --subset_ratio "$r" \
        --seed "$SEED" \
        --net basicunet \
        --init scratch \
        --pretrained_ckpt "$PRETRAINED_CKPT" | tee "$out/train.log"
    fi
  done

  echo "[GPU $gpu] Chain complete: init=$init"
}

# Assign one chain to each GPU
GPU_ARR=($GPUS)
if [[ ${#GPU_ARR[@]} -lt 2 ]]; then
  echo "ERROR: Need at least 2 GPU IDs in GPUS (e.g., '0 1')." >&2
  exit 3
fi

(
  run_chain "${GPU_ARR[0]}" scratch
) &

(
  run_chain "${GPU_ARR[1]}" pretrained
) &

wait
echo "All experiments complete."
