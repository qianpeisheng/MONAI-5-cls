#!/usr/bin/env bash
set -euo pipefail

# Run two concurrent few-slices training jobs using precomputed selected slice masks.
# GPU 0 and 1 are used respectively. Output dirs include timestamps (handled by the trainer).
#
# Usage (adjust paths as needed):
#   bash scripts/run_few_slices_selected_dual.sh \
#     /data3/wp5/wp5-code/dataloaders/wp5-dataset \
#     /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json \
#     runs/selected_slices_1pct_20251009-172901/sup_masks
#
# Args:
#   $1 = DATA_ROOT (folder that contains 'data' subfolder with NIfTI pairs)
#   $2 = SPLIT_CFG (JSON config for train/test split with test_serial_numbers)
#   $3 = SUP_MASKS_DIR (directory with <id>_supmask.npy produced by select_informative_slices.py)

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 DATA_ROOT SPLIT_CFG SUP_MASKS_DIR" >&2
  exit 1
fi

DATA_ROOT="$1"
SPLIT_CFG="$2"
SUP_MASKS_DIR="$3"

if [[ ! -d "$SUP_MASKS_DIR" ]]; then
  echo "Sup masks dir not found: $SUP_MASKS_DIR" >&2
  exit 1
fi

# Pick python
PY="${PYTHON:-venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

# Common training args
COMMON=(
  train_finetune_wp5.py
  --mode train
  --data_root "$DATA_ROOT"
  --split_cfg "$SPLIT_CFG"
  --fewshot_mode few_slices
  --fewshot_static
  --sup_masks_dir "$SUP_MASKS_DIR"
  --subset_ratio 1.0
  --epochs 50
  --batch_size 2
  --num_workers 4
  --lr 1e-4
)

TS=$(date +%Y%m%d-%H%M%S)
LOG_DIR="runs/few_slices_from_selection_${TS}_logs"
mkdir -p "$LOG_DIR"

echo "Launching two jobs with sup_masks_dir=$SUP_MASKS_DIR"

# GPU 0
CUDA_VISIBLE_DEVICES=0 "$PY" "${COMMON[@]}" \
  --seed 42 \
  --output_dir "runs/few_slices_from_selection_gpu0" \
  >"$LOG_DIR/gpu0.log" 2>&1 &
PID0=$!
echo "GPU0 PID=$PID0 (log=$LOG_DIR/gpu0.log)"

# GPU 1
CUDA_VISIBLE_DEVICES=1 "$PY" "${COMMON[@]}" \
  --seed 43 \
  --output_dir "runs/few_slices_from_selection_gpu1" \
  >"$LOG_DIR/gpu1.log" 2>&1 &
PID1=$!
echo "GPU1 PID=$PID1 (log=$LOG_DIR/gpu1.log)"

echo "Both jobs started. Tail logs with: tail -f $LOG_DIR/gpu{0,1}.log"

