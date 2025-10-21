#!/usr/bin/env bash
# Run WP5 verification experiments over subset ratios 20%..100% with full-label supervision (no dilation).
# Splits jobs across 2 GPUs and runs one job per GPU at a time.

set -euo pipefail

# --- Config (override via env) ---
DATA_ROOT=${DATA_ROOT:-/data3/wp5/wp5-code/dataloaders/wp5-dataset}
SPLIT_CFG=${SPLIT_CFG:-/data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-2}
NUM_WORKERS=${NUM_WORKERS:-4}
NET=${NET:-basicunet}
NORM=${NORM:-clip_zscore}
INIT=${INIT:-scratch}
# Subset ratios to test at 10% steps (10% .. 100%)
RATIOS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# GPU IDs to use
GPU0=${GPU0:-0}
GPU1=${GPU1:-1}
# Base output directory prefix (timestamp will be appended by the training script unless --no_timestamp is set)
BASE_OUT=${BASE_OUT:-runs/verify_basic_subset}

# If you want stable, non-timestamped directories, export NO_TS=1 before running.
TS_FLAG=""
if [[ "${NO_TS:-}" != "" ]]; then
  TS_FLAG="--no_timestamp"
fi

echo "DATA_ROOT=${DATA_ROOT}"
echo "SPLIT_CFG=${SPLIT_CFG}"
echo "Saving under base: ${BASE_OUT}_<ratio>[_TIMESTAMP]"
echo "GPUs: ${GPU0}, ${GPU1}"

run_series() {
  local gpu_id="$1"; shift
  for ratio in "$@"; do
    # Create ratio tag suitable for paths, e.g., 0.2 -> 020, 1.0 -> 100
    ratio_tag=$(printf "%03d" "$(awk -v r="$ratio" 'BEGIN{printf int(r*100)}')")
    outdir="${BASE_OUT}_${ratio_tag}"
    echo "[GPU ${gpu_id}] Starting subset_ratio=${ratio} -> ${outdir}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      set -x
      python3 -u train_finetune_wp5.py \
        --mode train \
        --data_root "${DATA_ROOT}" \
        --split_cfg "${SPLIT_CFG}" \
        --output_dir "${outdir}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --net "${NET}" \
        --norm "${NORM}" \
        --init "${INIT}" \
        --fewshot_mode few_samples \
        --subset_ratio "${ratio}" \
        --seed 42 \
        ${TS_FLAG}
    )
  done
}

# Split ratios alternately across the two GPUs
GPU0_RATS=()
GPU1_RATS=()
idx=0
for r in "${RATIOS[@]}"; do
  if (( idx % 2 == 0 )); then
    GPU0_RATS+=("$r")
  else
    GPU1_RATS+=("$r")
  fi
  idx=$((idx+1))
done

# Launch both GPU series in parallel (one job at a time per GPU)
( run_series "${GPU0}" "${GPU0_RATS[@]}" ) &
( run_series "${GPU1}" "${GPU1_RATS[@]}" ) &
wait

echo "All verification runs completed."
