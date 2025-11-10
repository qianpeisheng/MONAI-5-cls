#!/bin/bash
# Complete end-to-end pipeline: sampling + propagation + training all k variants
#
# Usage:
#   bash scripts/run_strategic_sparse_complete.sh [OPTIONS]
#
# This script runs:
#   1. Strategic sparse seed sampling (0.1% budget)
#   2. Multi-k label propagation (k=1,3,5,7,10,15,20,25,30,50)
#   3. Parallel training across 2 GPUs for all k variants

set -e  # Exit on error

# Default parameters
SV_DIR="/data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n12000_c0.05_s1.0_ras2_voted"
DATA_ROOT="/data3/wp5/wp5-code/dataloaders/wp5-dataset"
SPLIT_CFG="/data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json"
BUDGET_RATIO="0.001"
CLASS_WEIGHTS="0.1,1,1,2,2"
K_VALUES="1,3,5,7,10,15,20,25,30,50"
OUTPUT_DIR="runs/strategic_sparse_0p1pct_k_multi"
SEED="42"
SKIP_PIPELINE=false
SKIP_TRAINING=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-pipeline)
            SKIP_PIPELINE=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --help)
            echo "Usage: bash scripts/run_strategic_sparse_complete.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --output_dir DIR      Output directory (default: runs/strategic_sparse_0p1pct_k_multi)"
            echo "  --skip-pipeline       Skip sampling+propagation, go straight to training"
            echo "  --skip-training       Only run sampling+propagation, skip training"
            echo "  --help                Show this help message"
            echo ""
            echo "This script runs the complete pipeline:"
            echo "  1. Strategic seed sampling (0.1% budget, stratified by class)"
            echo "  2. Multi-k label propagation (k=1,3,5,7,10,15,20,25,30,50)"
            echo "  3. Parallel training across 2 GPUs for all 10 k variants"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

# Activate virtual environment if it exists
if [ -f "/home/peisheng/MONAI/venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source /home/peisheng/MONAI/venv/bin/activate
fi

# Print configuration
echo "============================================================"
echo "Strategic Sparse Supervoxel Pipeline - Complete Run"
echo "============================================================"
echo "Configuration:"
echo "  SV directory:    $SV_DIR"
echo "  Data root:       $DATA_ROOT"
echo "  Split config:    $SPLIT_CFG"
echo "  Budget ratio:    $BUDGET_RATIO (0.1%)"
echo "  Class weights:   $CLASS_WEIGHTS (0,1,2,3,4)"
echo "  K values:        $K_VALUES"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Random seed:     $SEED"
echo "============================================================"
echo ""

# Step 1: Run sampling + propagation pipeline
if [ "$SKIP_PIPELINE" = false ]; then
    echo "[1/2] Running strategic sparse pipeline (sampling + propagation)..."
    echo "This will take approximately 2-4 hours for 380 cases..."
    echo ""

    python3 scripts/pipeline_strategic_sparse_sv.py \
        --sv_dir "$SV_DIR" \
        --data_root "$DATA_ROOT" \
        --split_cfg "$SPLIT_CFG" \
        --budget_ratio "$BUDGET_RATIO" \
        --k_values "$K_VALUES" \
        --output_dir "$OUTPUT_DIR" \
        --seed "$SEED"

    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Pipeline failed at sampling/propagation stage"
        exit 1
    fi

    echo ""
    echo "✅ Pipeline (sampling + propagation) completed successfully!"
    echo ""
else
    echo "[1/2] Skipping pipeline (--skip-pipeline specified)"
    echo ""
fi

# Step 2: Train all k variants
if [ "$SKIP_TRAINING" = false ]; then
    echo "[2/2] Training all k variants across 2 GPUs..."
    echo "This will take approximately 10-15 hours (10 models, parallel on 2 GPUs)..."
    echo ""

    K_VARIANTS_DIR="$OUTPUT_DIR/k_variants"

    if [ ! -d "$K_VARIANTS_DIR" ]; then
        echo "ERROR: K variants directory not found: $K_VARIANTS_DIR"
        echo "Make sure the pipeline completed successfully before training."
        exit 1
    fi

    bash scripts/train_all_k_variants.sh "$K_VARIANTS_DIR"

    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Training failed"
        exit 1
    fi

    echo ""
    echo "✅ Training completed successfully!"
    echo ""
else
    echo "[2/2] Skipping training (--skip-training specified)"
    echo ""
fi

# Summary
echo "============================================================"
echo "✅ Complete pipeline finished successfully!"
echo "============================================================"
echo ""
echo "Outputs:"
echo "  Seeds:       $OUTPUT_DIR/strategic_seeds/"
echo "  Propagated:  $OUTPUT_DIR/cases/"
echo "  K variants:  $OUTPUT_DIR/k_variants/"
if [ "$SKIP_TRAINING" = false ]; then
    echo "  Trained:     runs/train_sv_sparse_k{01,03,05,07,10,15,20,25,30,50}/"
fi
echo ""
echo "Next steps:"
if [ "$SKIP_TRAINING" = false ]; then
    echo "  1. Evaluate models on test set"
    echo "  2. Compare performance across k values"
    echo "  3. Identify optimal k for sparse supervoxel labeling"
else
    echo "  1. Run training: bash scripts/train_all_k_variants.sh $OUTPUT_DIR/k_variants"
    echo "  2. Evaluate models on test set"
    echo "  3. Compare performance across k values"
fi
echo ""
