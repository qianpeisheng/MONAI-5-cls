#!/usr/bin/env bash
set -euo pipefail

# End-to-end: select informative slices emphasizing class 3, then train
# two concurrent jobs (GPU 0 and 1) using static few-slices supervision.
# Output folders include timestamps; nothing gets overwritten.
#
# Usage (adjust paths as needed):
#   bash scripts/run_select_and_train_emphC3_dual.sh \
#     /data3/wp5/wp5-code/dataloaders/wp5-dataset \
#     /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json \
#     0.01
#
# Args:
#   $1 = DATA_ROOT (folder that contains 'data' subfolder with NIfTI pairs)
#   $2 = SPLIT_CFG (JSON config for train/test split with test_serial_numbers)
#   $3 = PERCENT (fraction of all slices, across X+Y+Z, e.g., 0.01)

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 DATA_ROOT SPLIT_CFG PERCENT" >&2
  exit 1
fi

DATA_ROOT="$1"
SPLIT_CFG="$2"
PERCENT="$3"

# Emphasis config for class 3 (small/under-represented)
TARGET_CLASS=3
MIN_TARGET_SHARE=0.35
TARGET_BOOST=0.4
PER_VOLUME_CAP=12
NMS=3

# Trainer defaults (override by exporting env before running if desired)
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
SEED0="${SEED0:-42}"
SEED1="${SEED1:-43}"

# Pick python
PY="${PYTHON:-venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

# Optional delay before starting selection/training (default 1 hour)
SLEEP_SECONDS="${SLEEP_SECONDS:-3600}"
if [[ "$SLEEP_SECONDS" -gt 0 ]]; then
  echo "Delay requested: sleeping ${SLEEP_SECONDS}s before starting..." >&2
  sleep "$SLEEP_SECONDS"
fi

# 1) Selection (new timestamped folder)
TS_SEL=$(date +%Y%m%d-%H%M%S)
PCT_TAG=$(echo "$PERCENT" | sed 's/\./p/g')
SEL_OUT="runs/selected_slices_emphC3_${PCT_TAG}_${TS_SEL}"
mkdir -p "$SEL_OUT"

if [[ ! -f datalist_train.json ]];
then
  echo "datalist_train.json not found in CWD; point selector to your train list." >&2
  exit 1
fi

echo "[1/2] Selecting slices with class-3 emphasis into: $SEL_OUT"
"$PY" scripts/select_informative_slices.py \
  --train_list datalist_train.json \
  --out_dir "$SEL_OUT" \
  --percent "$PERCENT" \
  --per_volume_cap "$PER_VOLUME_CAP" \
  --nms "$NMS" \
  --save_sup_masks \
  --target_class "$TARGET_CLASS" \
  --min_target_share "$MIN_TARGET_SHARE" \
  --target_boost "$TARGET_BOOST"

SUP_MASKS_DIR="$SEL_OUT/sup_masks"
if [[ ! -d "$SUP_MASKS_DIR" ]]; then
  echo "Sup masks not found at $SUP_MASKS_DIR" >&2
  exit 1
fi

# 2) Train concurrently on GPU 0 and 1
TS_TRAIN=$(date +%Y%m%d-%H%M%S)
LOG_DIR="runs/few_slices_from_selection_emphC3_${TS_TRAIN}_logs"
mkdir -p "$LOG_DIR"

COMMON=(
  train_finetune_wp5.py
  --mode train
  --data_root "$DATA_ROOT"
  --split_cfg "$SPLIT_CFG"
  --fewshot_mode few_slices
  --fewshot_static
  --sup_masks_dir "$SUP_MASKS_DIR"
  --subset_ratio 1.0
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --lr "$LR"
)

echo "[2/2] Launching training on GPUs 0 and 1 using $SUP_MASKS_DIR"

CUDA_VISIBLE_DEVICES=0 "$PY" "${COMMON[@]}" \
  --seed "$SEED0" \
  --output_dir "runs/few_slices_from_selection_emphC3_gpu0" \
  >"$LOG_DIR/gpu0.log" 2>&1 &
PID0=$!
echo "GPU0 PID=$PID0 (log=$LOG_DIR/gpu0.log)"

CUDA_VISIBLE_DEVICES=1 "$PY" "${COMMON[@]}" \
  --seed "$SEED1" \
  --output_dir "runs/few_slices_from_selection_emphC3_gpu1" \
  >"$LOG_DIR/gpu1.log" 2>&1 &
PID1=$!
echo "GPU1 PID=$PID1 (log=$LOG_DIR/gpu1.log)"

echo "Both jobs started. Tail with: tail -f $LOG_DIR/gpu{0,1}.log"
