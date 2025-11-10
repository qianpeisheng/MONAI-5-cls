#!/usr/bin/env bash
#
# Train models for all k variants in parallel across 2 GPUs.
#
# Usage:
#   bash scripts/train_all_k_variants.sh <k_variants_dir> [--dry-run] [--resume]
#
# Example:
#   bash scripts/train_all_k_variants.sh runs/sv_sparse_prop_0p1pct_strategic/k_variants
#
# Directory structure expected:
#   <k_variants_dir>/
#     k01/<case_id>_labels.npy
#     k03/<case_id>_labels.npy
#     ...
#     k50/<case_id>_labels.npy
#
# Distributes jobs:
#   - GPU 0: k01, k03, k05, k07, k10 (first 5)
#   - GPU 1: k15, k20, k25, k30, k50 (last 5)
#   - Sequential within each GPU, parallel across GPUs
#
# Each run trains for 20 epochs, saves to runs/train_sv_sparse_k<XX>/
#

set -e  # Exit on error

# Parse arguments
K_VARIANTS_DIR=${1:-"runs/sv_sparse_prop_0p1pct_strategic/k_variants"}
DRY_RUN=0
RESUME=0

for arg in "$@"; do
    if [ "$arg" == "--dry-run" ]; then
        DRY_RUN=1
    elif [ "$arg" == "--resume" ]; then
        RESUME=1
    fi
done

# Check k_variants directory exists
if [ ! -d "$K_VARIANTS_DIR" ]; then
    echo "ERROR: K variants directory not found: $K_VARIANTS_DIR"
    echo "Usage: bash scripts/train_all_k_variants.sh <k_variants_dir>"
    exit 1
fi

# Find all k directories (k01, k03, k05, ...)
K_DIRS=($(find "$K_VARIANTS_DIR" -maxdepth 1 -type d -name 'k*' | sort))

if [ ${#K_DIRS[@]} -eq 0 ]; then
    echo "ERROR: No k variant directories found in $K_VARIANTS_DIR"
    exit 1
fi

echo "=========================================================================="
echo "TRAINING ALL K VARIANTS"
echo "=========================================================================="
echo "K variants dir: $K_VARIANTS_DIR"
echo "Found ${#K_DIRS[@]} k variants:"
for k_dir in "${K_DIRS[@]}"; do
    echo "  - $(basename $k_dir)"
done
echo ""

if [ $DRY_RUN -eq 1 ]; then
    echo "DRY RUN MODE - Commands will not be executed"
    echo ""
fi

if [ $RESUME -eq 1 ]; then
    echo "RESUME MODE - Skipping completed runs"
    echo ""
fi

# Split k values across 2 GPUs
# First half to GPU 0, second half to GPU 1
HALF=$((${#K_DIRS[@]} / 2))
GPU0_DIRS=("${K_DIRS[@]:0:$HALF}")
GPU1_DIRS=("${K_DIRS[@]:$HALF}")

echo "Distribution:"
echo "  GPU 0 (${#GPU0_DIRS[@]} jobs): $(for d in "${GPU0_DIRS[@]}"; do basename $d; done | tr '\n' ' ')"
echo "  GPU 1 (${#GPU1_DIRS[@]} jobs): $(for d in "${GPU1_DIRS[@]}"; do basename $d; done | tr '\n' ' ')"
echo ""

# Training configuration
DATA_ROOT="/data3/wp5/wp5-code/dataloaders/wp5-dataset"
SPLIT_CFG="/data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json"
EPOCHS=20
BATCH_SIZE=2
NET="basicunet"
SEED=42

# Function to train one k variant
train_k_variant() {
    local gpu_id=$1
    local k_dir=$2
    local k_name=$(basename "$k_dir")

    # Output directory
    local output_dir="runs/train_sv_sparse_${k_name}"

    # Check if already completed (resume mode)
    if [ $RESUME -eq 1 ] && [ -f "${output_dir}/best.ckpt" ]; then
        echo "[GPU $gpu_id] Skipping $k_name (already completed)"
        return 0
    fi

    echo "[GPU $gpu_id] Training $k_name..."
    echo "[GPU $gpu_id]   Labels: $k_dir"
    echo "[GPU $gpu_id]   Output: $output_dir"

    local cmd="CUDA_VISIBLE_DEVICES=$gpu_id python3 -u train_finetune_wp5.py \
        --mode train \
        --data_root $DATA_ROOT \
        --split_cfg $SPLIT_CFG \
        --train_label_override_dir $k_dir \
        --output_dir $output_dir \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --num_workers 4 \
        --init scratch \
        --net $NET \
        --subset_ratio 1.0 \
        --seed $SEED \
        --norm clip_zscore \
        --roi_x 112 --roi_y 112 --roi_z 80 \
        --log_to_file"

    if [ $DRY_RUN -eq 1 ]; then
        echo "[GPU $gpu_id] DRY RUN: $cmd"
    else
        # Run training
        eval $cmd

        if [ $? -eq 0 ]; then
            echo "[GPU $gpu_id] ✓ $k_name completed successfully"
        else
            echo "[GPU $gpu_id] ✗ $k_name FAILED"
            return 1
        fi
    fi
}

# Function to train all k variants on one GPU sequentially
train_gpu_sequential() {
    local gpu_id=$1
    shift
    local k_dirs=("$@")

    echo ""
    echo "Starting GPU $gpu_id jobs..."
    echo ""

    for k_dir in "${k_dirs[@]}"; do
        train_k_variant $gpu_id "$k_dir"
    done

    echo ""
    echo "GPU $gpu_id completed all jobs"
}

# Start timestamp
START_TIME=$(date +%s)
echo "Started at: $(date)"
echo ""

if [ $DRY_RUN -eq 0 ]; then
    # Run GPU 0 and GPU 1 jobs in parallel
    train_gpu_sequential 0 "${GPU0_DIRS[@]}" &
    PID_GPU0=$!

    train_gpu_sequential 1 "${GPU1_DIRS[@]}" &
    PID_GPU1=$!

    # Wait for both GPUs to complete
    echo "Waiting for GPU 0 (PID $PID_GPU0) and GPU 1 (PID $PID_GPU1) to complete..."
    wait $PID_GPU0
    STATUS_GPU0=$?

    wait $PID_GPU1
    STATUS_GPU1=$?

    # End timestamp
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(((ELAPSED % 3600) / 60))

    echo ""
    echo "=========================================================================="
    echo "ALL K VARIANT TRAINING JOBS COMPLETED!"
    echo "=========================================================================="
    echo "Completed at: $(date)"
    echo "Total time: ${HOURS}h ${MINUTES}m"
    echo ""

    if [ $STATUS_GPU0 -eq 0 ] && [ $STATUS_GPU1 -eq 0 ]; then
        echo "✓ All jobs completed successfully"
        echo ""
        echo "Results saved in: runs/train_sv_sparse_k*/"
        echo ""
        echo "To summarize results, run:"
        echo "  python3 scripts/summarize_sv_training_results.py"
        echo "=========================================================================="
        exit 0
    else
        echo "✗ Some jobs failed"
        echo "  GPU 0 status: $STATUS_GPU0"
        echo "  GPU 1 status: $STATUS_GPU1"
        echo "=========================================================================="
        exit 1
    fi
else
    echo "=========================================================================="
    echo "DRY RUN COMPLETE"
    echo "=========================================================================="
    echo "No training was executed."
    echo "Remove --dry-run to start training."
    echo "=========================================================================="
fi
