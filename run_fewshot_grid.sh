#!/usr/bin/env bash
# Launch >= 20 scratch few-shot experiments across 2 GPUs.
# Varies few-shot mode (few_slices, few_points), ratios (10%, 1%), and hyperparameters.
# Uses clip_zscore normalization by default.

set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/data3/wp5/wp5-code/dataloaders/wp5-dataset}
SPLIT_CFG=${SPLIT_CFG:-/data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json}
GPUS=${GPUS:-"0 1"}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-2}
LR=${LR:-1e-3}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-42}
NORM=${NORM:-clip_zscore}

SCRIPT=${SCRIPT:-train_finetune_wp5.py}
OUT_ROOT=${OUT_ROOT:-runs/fewshot_grid_${NORM}}

mkdir -p "$OUT_ROOT"

GPU_ARR=($GPUS)
if [[ ${#GPU_ARR[@]} -lt 2 ]]; then
  echo "ERROR: Need at least 2 GPUs in GPUS (e.g., '0 1')." >&2
  exit 2
fi

run_job() {
  local gpu="$1"; shift
  export CUDA_VISIBLE_DEVICES="$gpu"
  echo "[GPU $gpu] $*"
  "$@"
}

CMDS=()

# Few-slices 10%: axes {z,y,x,multi}
for AX in z y x multi; do
  CMDS+=("python3 -u $SCRIPT --mode train --data_root $DATA_ROOT --split_cfg $SPLIT_CFG --output_dir $OUT_ROOT/slices_${AX}_10 --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --lr $LR --subset_ratio 1.0 --seed $SEED --net basicunet --norm $NORM --init scratch --fewshot_mode few_slices --fewshot_ratio 0.1 --fs_axis_mode $AX")
done

# Few-slices 1%: axes {z,y,x,multi}
for AX in z y x multi; do
  CMDS+=("python3 -u $SCRIPT --mode train --data_root $DATA_ROOT --split_cfg $SPLIT_CFG --output_dir $OUT_ROOT/slices_${AX}_01 --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --lr $LR --subset_ratio 1.0 --seed $SEED --net basicunet --norm $NORM --init scratch --fewshot_mode few_slices --fewshot_ratio 0.01 --fs_axis_mode $AX")
done

# Few-slices 10% with explicit K override
CMDS+=("python3 -u $SCRIPT --mode train --data_root $DATA_ROOT --split_cfg $SPLIT_CFG --output_dir $OUT_ROOT/slices_z_10_k12 --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --lr $LR --subset_ratio 1.0 --seed $SEED --net basicunet --norm $NORM --init scratch --fewshot_mode few_slices --fewshot_ratio 0.1 --fs_axis_mode z --fs_k_slices 12")

# Few-points 10%: dilate 1 & 2, balance {proportional, uniform}
for DIL in 1 2; do
  for BAL in proportional uniform; do
    CMDS+=("python3 -u $SCRIPT --mode train --data_root $DATA_ROOT --split_cfg $SPLIT_CFG --output_dir $OUT_ROOT/points_10_d${DIL}_${BAL} --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --lr $LR --subset_ratio 1.0 --seed $SEED --net basicunet --norm $NORM --init scratch --fewshot_mode few_points --fewshot_ratio 0.1 --fp_dilate_radius $DIL --fp_balance $BAL --fp_bg_frac 0.25")
  done
done

# Few-slices + intensity augs (add to both 10% and 1%)
# Few-slices + intensity augs (add to both 10% and 1%)
for AX in z y x multi; do
  CMDS+=("python3 -u $SCRIPT --mode train --data_root $DATA_ROOT --split_cfg $SPLIT_CFG --output_dir $OUT_ROOT/slices_${AX}_10_aug --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --lr $LR --subset_ratio 1.0 --seed $SEED --net basicunet --norm $NORM --init scratch --fewshot_mode few_slices --fewshot_ratio 0.1 --fs_axis_mode $AX --aug_intensity --aug_prob 0.2 --aug_noise_std 0.01 --aug_shift 0.1 --aug_scale 0.1")
done
for AX in z y x multi; do
  CMDS+=("python3 -u $SCRIPT --mode train --data_root $DATA_ROOT --split_cfg $SPLIT_CFG --output_dir $OUT_ROOT/slices_${AX}_01_aug --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --lr $LR --subset_ratio 1.0 --seed $SEED --net basicunet --norm $NORM --init scratch --fewshot_mode few_slices --fewshot_ratio 0.01 --fs_axis_mode $AX --aug_intensity --aug_prob 0.2 --aug_noise_std 0.01 --aug_shift 0.1 --aug_scale 0.1")
done

# Few-points 1% + FG-biased crop (dilate 1 & 2, balance {proportional, uniform})
for DIL in 1 2; do
  for BAL in proportional uniform; do
    CMDS+=("python3 -u $SCRIPT --mode train --data_root $DATA_ROOT --split_cfg $SPLIT_CFG --output_dir $OUT_ROOT/points_01_d${DIL}_${BAL}_fg --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --lr $LR --subset_ratio 1.0 --seed $SEED --net basicunet --norm $NORM --init scratch --fewshot_mode few_points --fewshot_ratio 0.01 --fp_dilate_radius $DIL --fp_balance $BAL --fp_bg_frac 0.25 --fg_crop_prob 0.7")
  done
done

# Few-points 10% + pseudo-labels (dilate=1, balance {proportional, uniform})
for BAL in proportional uniform; do
  CMDS+=("python3 -u $SCRIPT --mode train --data_root $DATA_ROOT --split_cfg $SPLIT_CFG --output_dir $OUT_ROOT/points_10_d1_${BAL}_pl --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --lr $LR --subset_ratio 1.0 --seed $SEED --net basicunet --norm $NORM --init scratch --fewshot_mode few_points --fewshot_ratio 0.1 --fp_dilate_radius 1 --fp_balance $BAL --fp_bg_frac 0.25 --pl_enable --pl_threshold 0.9 --pl_weight 0.2 --pl_warmup_epochs 5")
done
for DIL in 1 2; do
  for BAL in proportional uniform; do
    CMDS+=("python3 -u $SCRIPT --mode train --data_root $DATA_ROOT --split_cfg $SPLIT_CFG --output_dir $OUT_ROOT/points_01_d${DIL}_${BAL} --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --lr $LR --subset_ratio 1.0 --seed $SEED --net basicunet --norm $NORM --init scratch --fewshot_mode few_points --fewshot_ratio 0.01 --fp_dilate_radius $DIL --fp_balance $BAL --fp_bg_frac 0.25")
  done
done

# Total commands and split into 2 GPU chains
N=${#CMDS[@]}
chain0=()
chain1=()
for i in "${!CMDS[@]}"; do
  if (( i % 2 == 0 )); then
    chain0+=("${CMDS[$i]}")
  else
    chain1+=("${CMDS[$i]}")
  fi
done
echo "Prepared $N experiments (GPU ${GPU_ARR[0]}: ${#chain0[@]} runs, GPU ${GPU_ARR[1]}: ${#chain1[@]} runs)"

# Run chains concurrently, each GPU sequentially
run_chain() {
  local gpu="$1"; shift
  for cmd in "$@"; do
    run_job "$gpu" bash -lc "$cmd"
  done
}

(
  run_chain "${GPU_ARR[0]}" "${chain0[@]}"
) &
pid0=$!
(
  run_chain "${GPU_ARR[1]}" "${chain1[@]}"
) &
pid1=$!

wait "$pid0"
wait "$pid1"
echo "Few-shot grid complete. Output: $OUT_ROOT"
