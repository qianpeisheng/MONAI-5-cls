#!/usr/bin/env bash
set -euo pipefail

# Schedule two experiments sequentially on a chosen GPU (default 0):
# 1) Train with dense pseudo labels (override training labels)
# 2) Train with sparse 1% selected points (few_points static), using selected points + dense pseudo as pseudo labels

GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# Optional: wait before starting (default 0 seconds)
DELAY_SECONDS=${DELAY_SECONDS:-0}
echo "Delaying start by ${DELAY_SECONDS} seconds (~$((DELAY_SECONDS/3600)) hours)."
echo "Now:   $(date)"
sleep "${DELAY_SECONDS}"
echo "Start: $(date)"

DATA_ROOT=${DATA_ROOT:-/data3/wp5/wp5-code/dataloaders/wp5-dataset}
SPLIT_CFG=${SPLIT_CFG:-/data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json}
# Use the same architecture as the pretrained run (bundle UNet), but initialize from scratch
BUNDLE_DIR=${BUNDLE_DIR:-pretrained_models/spleen_ct_segmentation/spleen_ct_segmentation}

# Paths produced by the sparse2dense pipeline
PSEUDO_DENSE_DIR=/data3/wp5/wp5-code/dataloaders/wp5-dataset/pseudo_labels_1pct_train
PSEUDO_INTERMEDIATE_DIR=/data3/wp5/wp5-code/dataloaders/wp5-dataset/pseudo_labels_1pct_train_intermediate

echo "[1/2] Training with dense pseudo labels as GT (full supervision)"
venv/bin/python train_finetune_wp5.py \
  --mode train \
  --data_root "${DATA_ROOT}" \
  --split_cfg "${SPLIT_CFG}" \
  --output_dir runs/train_with_dense_pseudo \
  --train_label_override_dir "${PSEUDO_DENSE_DIR}" \
  --epochs 50 --batch_size 2 --num_workers 4 \
  --bundle_dir "${BUNDLE_DIR}" --init scratch --lr 1e-4 \
  --subset_ratio 1.0 --no_timestamp

echo "[2/2] Training with sparse 1% selected points (few_points static, no dilation)"
venv/bin/python train_finetune_wp5.py \
  --mode train \
  --data_root "${DATA_ROOT}" \
  --split_cfg "${SPLIT_CFG}" \
  --output_dir runs/train_with_1pct_points_raw \
  --epochs 50 --batch_size 2 --num_workers 4 \
  --bundle_dir "${BUNDLE_DIR}" --init scratch --lr 1e-4 \
  --subset_ratio 1.0 \
  --fewshot_mode few_points --fewshot_static \
  --fp_dilate_radius 0 \
  --selected_points_dir "${PSEUDO_INTERMEDIATE_DIR}" \
  --save_sup_masks \
  --no_timestamp

echo "All experiments completed."
