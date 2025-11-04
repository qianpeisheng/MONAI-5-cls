#!/usr/bin/env bash
set -euo pipefail

# Single experiment: few-points static with ratio=1e-6 (0.000001), from-scratch BasicUNet.
# Runs on GPU 1 by default.

# Config (override via env)
DATA_ROOT=${DATA_ROOT:-/data3/wp5/wp5-code/dataloaders/wp5-dataset}
SPLIT_CFG=${SPLIT_CFG:-/data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-2}
LR=${LR:-1e-3}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-42}
NORM=${NORM:-clip_zscore}
GPU=${GPU:-1}
RATIO=${RATIO:-0.000001}

# Output root (stable, no timestamp)
OUT_ROOT=${OUT_ROOT:-runs/fixed_points_scratch50}
TS_FLAG="--no_timestamp"

mkdir -p "$OUT_ROOT"

outdir="${OUT_ROOT}/ratio_${RATIO}"
echo "[GPU ${GPU}] From-scratch few-points ratio=${RATIO} -> ${outdir}"

CUDA_VISIBLE_DEVICES="${GPU}" \
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
  --fewshot_ratio "$RATIO" \
  --fewshot_static \
  --save_sup_masks \
  --fp_dilate_radius 0 \
  --fp_sample_mode uniform_all \
  ${FP_UNIFORM_EXCLUDE6:+--fp_uniform_exclude6} \
  --pseudo_weight 0.0 \
  --fg_crop_prob 0.0 \
  --coverage_mode seeds \
  --log_to_file \
  $TS_FLAG

echo "Run complete. Output: $outdir"
