#!/usr/bin/env bash
set -euo pipefail

# Fixed, static few-points with MONAI bundle backbone, 50 epochs, GPU 1.
# - Uniform random sampling over voxels (incl. background). Optionally exclude class 6.
# - No dilation (fp_dilate_radius=0), no pseudo-labels, no FG-biased crop.
# - Saves static masks to <run>/sup_masks and uses them every epoch (reproducible with --seed).

# Config (override via env)
DATA_ROOT=${DATA_ROOT:-/data3/wp5/wp5-code/dataloaders/wp5-dataset}
SPLIT_CFG=${SPLIT_CFG:-/data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json}
BUNDLE_DIR=${BUNDLE_DIR:-pretrained_models/spleen_ct_segmentation/spleen_ct_segmentation}
PRETRAINED_CKPT=${PRETRAINED_CKPT:-$BUNDLE_DIR/models/model.pt}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-2}
LR=${LR:-1e-3}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-42}
NORM=${NORM:-clip_zscore}

# Output root (stable, no timestamp)
OUT_ROOT=${OUT_ROOT:-runs/fixed_points_bundle50}
TS_FLAG="--no_timestamp"

# Ratios: 10%..100% step 10%, plus 5%, 2%, 1%, 0.5%, 0.25%
RATIOS=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05 0.02 0.01 0.005 0.0025)

# Pin to GPU 1
export CUDA_VISIBLE_DEVICES=1

mkdir -p "$OUT_ROOT"

for ratio in "${RATIOS[@]}"; do
  outdir="${OUT_ROOT}/ratio_${ratio}"
  echo "[GPU 1] Bundle few-points (no dilation) ratio=${ratio} -> ${outdir}"
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
    --bundle_dir "$BUNDLE_DIR" \
    --pretrained_ckpt "$PRETRAINED_CKPT" \
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
done

echo "All bundle fixed few-points runs (50 epochs) submitted on GPU 1. Output root: $OUT_ROOT"
