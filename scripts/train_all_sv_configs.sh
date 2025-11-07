#!/bin/bash
# Train segmentation models using all 30 supervoxel configurations
# Distributes jobs across 2 GPUs (0 and 1) with sequential execution per GPU

set -e  # Exit on error

# Configuration
SV_ROOT="/data3/wp5/monai-sv-sweeps"
DATA_ROOT="/data3/wp5/wp5-code/dataloaders/wp5-dataset"
SPLIT_CFG="${DATA_ROOT}/3ddl_split_config_20250801.json"
VENV_PATH="/home/peisheng/MONAI/venv/bin/activate"
EPOCHS=20
BATCH_SIZE=2
NUM_WORKERS=4

# Parse arguments
DRY_RUN=0
RESUME=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--dry-run] [--resume]"
      echo "  --dry-run: Show commands without executing"
      echo "  --resume:  Skip already completed runs"
      exit 1
      ;;
  esac
done

# Find all SV voted directories (not _eval)
echo "Finding supervoxel configurations in ${SV_ROOT}..."
SV_DIRS=($(find "${SV_ROOT}" -maxdepth 1 -type d -name "*_voted" ! -name "*_eval" | sort))

if [ ${#SV_DIRS[@]} -eq 0 ]; then
    echo "ERROR: No supervoxel directories found!"
    exit 1
fi

echo "Found ${#SV_DIRS[@]} supervoxel configurations"
echo ""

# Function to extract config from directory name
# Format: sv_fullgt_<MODE>_n<SEGMENTS>_c<COMPACTNESS>_s<SIGMA>_ras2_voted
extract_config() {
    local dir_name=$(basename "$1")

    # Extract mode (everything between sv_fullgt_ and _n)
    MODE=$(echo "$dir_name" | sed -E 's/sv_fullgt_(.+)_n[0-9]+.*/\1/')

    # Extract n_segments
    N_SEG=$(echo "$dir_name" | sed -E 's/.*_n([0-9]+)_.*/\1/')

    # Extract compactness
    COMP=$(echo "$dir_name" | sed -E 's/.*_c([0-9.]+)_.*/\1/')

    # Extract sigma
    SIGMA=$(echo "$dir_name" | sed -E 's/.*_s([0-9.]+)_.*/\1/')

    echo "${MODE}_n${N_SEG}_c${COMP}_s${SIGMA}"
}

# Function to run training for a given config
run_training() {
    local gpu_id=$1
    local sv_dir=$2
    local config_name=$3

    local output_dir="runs/train_sv_${config_name}_e${EPOCHS}"

    # Check if already completed (resume mode)
    if [ $RESUME -eq 1 ] && [ -d "$output_dir" ]; then
        if [ -f "${output_dir}/train.log" ]; then
            echo "  [GPU ${gpu_id}] SKIPPING (already exists): ${config_name}"
            return 0
        fi
    fi

    local cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python3 -u train_finetune_wp5.py \
  --mode train \
  --data_root ${DATA_ROOT} \
  --split_cfg ${SPLIT_CFG} \
  --train_label_override_dir ${sv_dir} \
  --output_dir ${output_dir} \
  --epochs ${EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS} \
  --init scratch \
  --net basicunet \
  --norm clip_zscore \
  --roi_x 112 --roi_y 112 --roi_z 80 \
  --log_to_file"

    if [ $DRY_RUN -eq 1 ]; then
        echo "  [GPU ${gpu_id}] DRY-RUN: ${config_name}"
        echo "    Output: ${output_dir}"
        echo "    Command: ${cmd}"
        echo ""
    else
        echo "  [GPU ${gpu_id}] STARTING: ${config_name}"
        echo "    SV dir: ${sv_dir}"
        echo "    Output: ${output_dir}"
        echo "    Started at: $(date)"

        # Activate venv and run
        source "${VENV_PATH}"
        eval $cmd

        echo "  [GPU ${gpu_id}] COMPLETED: ${config_name} at $(date)"
        echo ""
    fi
}

# Split configs between GPUs
echo "========================================================================"
echo "TRAINING CONFIGURATION"
echo "========================================================================"
echo "Total configs:    ${#SV_DIRS[@]}"
echo "GPU 0 jobs:       $((${#SV_DIRS[@]} / 2))"
echo "GPU 1 jobs:       $((${#SV_DIRS[@]} - ${#SV_DIRS[@]} / 2))"
echo "Epochs per run:   ${EPOCHS}"
echo "Batch size:       ${BATCH_SIZE}"
echo "Num workers:      ${NUM_WORKERS}"
echo "Dry run:          $([ $DRY_RUN -eq 1 ] && echo 'YES' || echo 'NO')"
echo "Resume mode:      $([ $RESUME -eq 1 ] && echo 'YES' || echo 'NO')"
echo "========================================================================"
echo ""

# Function to run all jobs for a specific GPU
run_gpu_jobs() {
    local gpu_id=$1
    local start_idx=$2
    local end_idx=$3

    echo ""
    echo "=========================================="
    echo "GPU ${gpu_id} - Starting $(($end_idx - $start_idx)) jobs"
    echo "=========================================="

    local job_num=1
    for ((i=$start_idx; i<$end_idx; i++)); do
        local sv_dir="${SV_DIRS[$i]}"
        local config_name=$(extract_config "$sv_dir")

        echo ""
        echo "[GPU ${gpu_id}] Job ${job_num}/$(($end_idx - $start_idx)): ${config_name}"

        run_training $gpu_id "$sv_dir" "$config_name"

        job_num=$((job_num + 1))
    done

    echo ""
    echo "=========================================="
    echo "GPU ${gpu_id} - ALL JOBS COMPLETED"
    echo "=========================================="
}

# Split jobs between GPUs
TOTAL=${#SV_DIRS[@]}
MID=$((TOTAL / 2))

# Preview all jobs
echo "JOB ASSIGNMENTS:"
echo "----------------"
echo ""
echo "GPU 0 jobs (${MID} configs):"
for ((i=0; i<$MID; i++)); do
    config_name=$(extract_config "${SV_DIRS[$i]}")
    output_dir="runs/train_sv_${config_name}_e${EPOCHS}"
    status=""
    if [ $RESUME -eq 1 ] && [ -d "$output_dir" ] && [ -f "${output_dir}/train.log" ]; then
        status=" [WILL SKIP - exists]"
    fi
    echo "  $((i+1)). ${config_name}${status}"
done

echo ""
echo "GPU 1 jobs ($((TOTAL - MID)) configs):"
for ((i=$MID; i<$TOTAL; i++)); do
    config_name=$(extract_config "${SV_DIRS[$i]}")
    output_dir="runs/train_sv_${config_name}_e${EPOCHS}"
    status=""
    if [ $RESUME -eq 1 ] && [ -d "$output_dir" ] && [ -f "${output_dir}/train.log" ]; then
        status=" [WILL SKIP - exists]"
    fi
    echo "  $((i-MID+1)). ${config_name}${status}"
done
echo ""

if [ $DRY_RUN -eq 1 ]; then
    echo "========================================================================"
    echo "DRY RUN MODE - No training will be executed"
    echo "========================================================================"
    echo ""
    run_gpu_jobs 0 0 $MID
    run_gpu_jobs 1 $MID $TOTAL
    echo ""
    echo "To execute for real, run without --dry-run flag"
    exit 0
fi

# Confirm before starting
echo "========================================================================"
echo "Ready to start training. This will take approximately 15-20 hours."
echo "Press Ctrl+C within 5 seconds to cancel..."
echo "========================================================================"
sleep 5

# Run both GPUs in parallel (but sequential within each GPU)
run_gpu_jobs 0 0 $MID &
GPU0_PID=$!

run_gpu_jobs 1 $MID $TOTAL &
GPU1_PID=$!

# Wait for both GPUs to complete
wait $GPU0_PID
wait $GPU1_PID

echo ""
echo "========================================================================"
echo "ALL TRAINING JOBS COMPLETED!"
echo "========================================================================"
echo "Completed at: $(date)"
echo "Results saved in: runs/train_sv_*_e${EPOCHS}/"
echo ""
echo "To summarize results, run:"
echo "  python3 scripts/summarize_sv_training_results.py"
echo "========================================================================"
