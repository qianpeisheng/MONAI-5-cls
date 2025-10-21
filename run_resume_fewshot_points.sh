#!/usr/bin/env bash
# Resume/launch the remaining few-points experiments from run_fewshot_grid.sh.
# - Skips runs that already have metrics/summary.json
# - Distributes jobs across 2 GPUs (default GPUS="0 1")
# - Each job logs to <out_dir>/train.log and continues on errors

set -euo pipefail

# Config (override via env)
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
if [[ ${#GPU_ARR[@]} -lt 1 ]]; then
  echo "ERROR: Need at least 1 GPU in GPUS (e.g., '0 1')." >&2
  exit 2
fi
if [[ ${#GPU_ARR[@]} -lt 2 ]]; then
  echo "WARN: Only one GPU provided; jobs will run sequentially on GPU ${GPU_ARR[0]}" >&2
fi

COMMON="--mode train --data_root $DATA_ROOT --split_cfg $SPLIT_CFG --epochs $EPOCHS --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS --lr $LR --subset_ratio 1.0 --seed $SEED --net basicunet --norm $NORM --init scratch --fewshot_mode few_points"

# Helper: add cmd if run not finished
CMDS=()
add_job() {
  local outdir="$1"; shift
  local args="$*"
  if [[ -f "$outdir/metrics/summary.json" ]]; then
    echo "SKIP (finished): $outdir"
    return
  fi
  mkdir -p "$outdir"
  CMDS+=("mkdir -p '$outdir'; python3 -u $SCRIPT $COMMON --output_dir '$outdir' $args >> '$outdir/train.log' 2>&1")
}

# Remaining jobs to run (or re-run if not finished)
# 1) Few-points 1% + FG-biased crop (dilate 1 & 2, balance {proportional, uniform})
for DIL in 1 2; do
  for BAL in proportional uniform; do
    add_job "$OUT_ROOT/points_01_d${DIL}_${BAL}_fg" "--fewshot_ratio 0.01 --fp_dilate_radius $DIL --fp_balance $BAL --fp_bg_frac 0.25 --fg_crop_prob 0.7"
  done
done

# 2) Few-points 10% + pseudo-labels (dilate=1, balance {proportional, uniform})
for BAL in proportional uniform; do
  add_job "$OUT_ROOT/points_10_d1_${BAL}_pl" "--fewshot_ratio 0.1 --fp_dilate_radius 1 --fp_balance $BAL --fp_bg_frac 0.25 --pl_enable --pl_threshold 0.9 --pl_weight 0.2 --pl_warmup_epochs 5"
done

# 3) Few-points 1% baseline (no FG-biased crop)
for DIL in 1 2; do
  for BAL in proportional uniform; do
    add_job "$OUT_ROOT/points_01_d${DIL}_${BAL}" "--fewshot_ratio 0.01 --fp_dilate_radius $DIL --fp_balance $BAL --fp_bg_frac 0.25"
  done
done

N=${#CMDS[@]}
if (( N == 0 )); then
  echo "Nothing to run. All targeted runs have summary.json."
  exit 0
fi

echo "Prepared $N jobs under $OUT_ROOT"

# Split commands across GPUs (2-way split by index parity)
chain0=()
chain1=()
for i in "${!CMDS[@]}"; do
  if (( i % 2 == 0 )); then
    chain0+=("${CMDS[$i]}")
  else
    chain1+=("${CMDS[$i]}")
  fi
done

run_chain() {
  local gpu="$1"; shift
  local -n cmds_ref=$1
  local log="$OUT_ROOT/chain_gpu${gpu}.log"
  echo "Running ${#cmds_ref[@]} jobs on GPU $gpu (log: $log)"
  (
    set +e
    export CUDA_VISIBLE_DEVICES="$gpu"
    for cmd in "${cmds_ref[@]}"; do
      echo "[GPU $gpu] $cmd"
      bash -lc "$cmd" || echo "[GPU $gpu] JOB FAILED: $cmd"
    done
  ) >"$log" 2>&1 &
}

# Launch chains
if [[ ${#GPU_ARR[@]} -ge 2 ]]; then
  run_chain "${GPU_ARR[0]}" chain0
  run_chain "${GPU_ARR[1]}" chain1
else
  # Single GPU: run sequentially
  run_chain "${GPU_ARR[0]}" chain0
  # append second chain to same GPU
  run_chain "${GPU_ARR[0]}" chain1
fi

echo "Launched. Tail per-run logs like: tail -f $OUT_ROOT/points_01_d1_proportional_fg/train.log"

