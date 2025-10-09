#!/usr/bin/env bash
set -euo pipefail

# Two extreme fixed-points experiments (ratio=0.0 and 1e-5) from-scratch
# Run concurrently on 2 GPUs (0 and 1 by default) using BasicUNet.
#
# Notes
# - Static few-points, uniform sampling over all voxels, no dilation, no pseudo-labels.
# - From-scratch model (no bundle, no pretrained checkpoint), explicit --net basicunet.
# - Output uses --no_timestamp to write to stable folders under runs/fixed_points_scratch50 (by default).

# Config (override via env)
DATA_ROOT=${DATA_ROOT:-/data3/wp5/wp5-code/dataloaders/wp5-dataset}
SPLIT_CFG=${SPLIT_CFG:-/data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-2}
LR=${LR:-1e-3}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-42}
NORM=${NORM:-clip_zscore}

# Output root (stable, no timestamp)
OUT_ROOT=${OUT_ROOT:-runs/fixed_points_scratch50}
TS_FLAG="--no_timestamp"

# GPU assignment (default to two GPUs: 0 and 1)
GPU0=${GPU0:-0}
GPU1=${GPU1:-1}

mkdir -p "$OUT_ROOT"

declare -a RATIOS=(
  0.0       # zero supervision
  0.00001   # 1e-5: extremely few seeds
)

run_job() {
  local gpu="$1"
  local ratio="$2"
  local outdir="${OUT_ROOT}/ratio_${ratio}"
  echo "[GPU ${gpu}] From-scratch few-points (no dilation) ratio=${ratio} -> ${outdir}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  python3 -u train_finetune_wp5.py \
    --mode train \
    --data_root "$DATA_ROOT" \
    --split_cfg "$SPLIT_CFG" \
    --output_dir "$outdir" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --lr "$LR" \
    --subset_ratio 1.0 \
    --seed "$SEED" \
    --norm "$NORM" \
    --init scratch \
    --net basicunet \
    --fewshot_mode few_points \
    --fewshot_ratio "$ratio" \
    --fewshot_static \
    --save_sup_masks \
    --fp_dilate_radius 0 \
    --fp_sample_mode uniform_all \
    ${FP_UNIFORM_EXCLUDE6:+--fp_uniform_exclude6} \
    --pseudo_weight 0.0 \
    --fg_crop_prob 0.0 \
    --coverage_mode seeds \
    $TS_FLAG
}

# Ensure child jobs are terminated if the script exits
trap 'jobs -p | xargs -r kill' EXIT

# Launch the two ratios concurrently on two GPUs
run_job "$GPU0" "${RATIOS[0]}" &
PID0=$!
run_job "$GPU1" "${RATIOS[1]}" &
PID1=$!

echo "Launched jobs: PID0=$PID0 (GPU $GPU0, ratio=${RATIOS[0]}), PID1=$PID1 (GPU $GPU1, ratio=${RATIOS[1]})"

wait $PID0 $PID1

echo "Extreme-ratio from-scratch runs complete. Output root: $OUT_ROOT"
