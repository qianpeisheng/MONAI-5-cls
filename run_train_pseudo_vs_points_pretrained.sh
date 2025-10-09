#!/usr/bin/env bash
set -euo pipefail

# Schedule two experiments sequentially on a chosen GPU (default 0) using pretrained weights:
# 1) Train with dense pseudo labels as GT (full supervision)
# 2) Train with sparse 1% selected points (few_points static, raw points, no dilation)

GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Optional: wait before starting (default 0 seconds)
DELAY_SECONDS=${DELAY_SECONDS:-0}
echo "Delaying start by ${DELAY_SECONDS} seconds (~$((DELAY_SECONDS/3600)) hours)."
echo "Now:   $(date)"
sleep "${DELAY_SECONDS}"
echo "Start: $(date)"

# Dataset/split
DATA_ROOT=${DATA_ROOT:-/data3/wp5/wp5-code/dataloaders/wp5-dataset}
SPLIT_CFG=${SPLIT_CFG:-/data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json}

# Pretrained MONAI bundle (BasicUNet family) and weights
BUNDLE_DIR=${BUNDLE_DIR:-pretrained_models/spleen_ct_segmentation/spleen_ct_segmentation}
PRETRAINED_CKPT=${PRETRAINED_CKPT:-$BUNDLE_DIR/models/model.pt}

# Paths produced by the sparse2dense pipeline
PSEUDO_DENSE_DIR=${PSEUDO_DENSE_DIR:-/data3/wp5/wp5-code/dataloaders/wp5-dataset/pseudo_labels_1pct_train}
PSEUDO_INTERMEDIATE_DIR=${PSEUDO_INTERMEDIATE_DIR:-/data3/wp5/wp5-code/dataloaders/wp5-dataset/pseudo_labels_1pct_train_intermediate}

# Common training knobs
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
LR_FT=${LR_FT:-3e-4}
SUBSET_RATIO=${SUBSET_RATIO:-1.0}

echo "[1/2] (pretrained) Training with dense pseudo labels as GT"
venv/bin/python train_finetune_wp5.py \
  --mode train \
  --data_root "${DATA_ROOT}" \
  --split_cfg "${SPLIT_CFG}" \
  --output_dir runs/train_with_dense_pseudo_pretrained \
  --train_label_override_dir "${PSEUDO_DENSE_DIR}" \
  --epochs "${EPOCHS}" --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
  --bundle_dir "${BUNDLE_DIR}" \
  --init pretrained --pretrained_ckpt "${PRETRAINED_CKPT}" \
  --lr_ft "${LR_FT}" \
  --subset_ratio "${SUBSET_RATIO}" \
  --no_timestamp

echo "[2/2] (pretrained) Training with sparse 1% selected points (raw, no dilation)"
venv/bin/python train_finetune_wp5.py \
  --mode train \
  --data_root "${DATA_ROOT}" \
  --split_cfg "${SPLIT_CFG}" \
  --output_dir runs/train_with_1pct_points_raw_pretrained \
  --epochs "${EPOCHS}" --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" \
  --bundle_dir "${BUNDLE_DIR}" \
  --init pretrained --pretrained_ckpt "${PRETRAINED_CKPT}" \
  --lr_ft "${LR_FT}" \
  --subset_ratio "${SUBSET_RATIO}" \
  --fewshot_mode few_points --fewshot_static \
  --fp_dilate_radius 0 \
  --selected_points_dir "${PSEUDO_INTERMEDIATE_DIR}" \
  --save_sup_masks \
  --no_timestamp

echo "All pretrained experiments completed."
