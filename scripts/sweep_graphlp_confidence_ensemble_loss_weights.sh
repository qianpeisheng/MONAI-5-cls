#!/usr/bin/env bash
set -euo pipefail

# Sweep loss-weight hyperparameters for Graph-LP confidence ensembles (agreement-count weighting).
#
# Prereq (build ensemble labels + agree_count maps):
#   bash scripts/run_graphlp_confidence_ensemble_experiments.sh
#
# This script schedules 40 training runs (2 GPUs: 0 and 1), sequential per GPU.
#
# Usage:
#   bash scripts/sweep_graphlp_confidence_ensemble_loss_weights.sh --dry-run
#   bash scripts/sweep_graphlp_confidence_ensemble_loss_weights.sh --resume --tag mytag

cd /home/peisheng/MONAI

# Optional: activate repo venv if it exists
if [ -f /home/peisheng/MONAI/venv/bin/activate ]; then . /home/peisheng/MONAI/venv/bin/activate; fi
PY="python3"
if [ -x /home/peisheng/MONAI/venv/bin/python ]; then PY="/home/peisheng/MONAI/venv/bin/python"; fi

DATA_ROOT="/data3/wp5_4_Dec_data/3ddl-dataset"
SPLIT_CFG="/data3/wp5_4_Dec_data/3ddl-dataset/data/dataset_config.json"
GT_SENTINEL=255

COMQ_DIR="runs/graph_lp_ens_vote_COMQ_tieQ"
CQ_DIR="runs/graph_lp_ens_vote_CQ_tieQ"

# Training defaults (override via env vars if desired)
EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LR="${LR:-0.001}"
ROI_X="${ROI_X:-112}"
ROI_Y="${ROI_Y:-112}"
ROI_Z="${ROI_Z:-80}"
SEED="${SEED:-42}"

DRY_RUN=0
RESUME=0
TAG=""
OUT_BASE="${OUT_BASE:-/data3/MONAI_experiments}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --tag)
      TAG="${2:-}"
      shift 2
      ;;
    --out-base)
      OUT_BASE="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--dry-run] [--resume] [--tag <name>] [--out-base <dir>]" >&2
      exit 2
      ;;
  esac
done

if [ -z "${TAG}" ]; then TAG="$(date +%Y%m%d-%H%M%S)"; fi
mkdir -p "${OUT_BASE}"
OUT_ROOT="${OUT_BASE}/sweep_graphlp_conf_ens_lossw_${TAG}"

if [ ! -d "${DATA_ROOT}" ]; then echo "ERROR: missing DATA_ROOT dir: ${DATA_ROOT}" >&2; exit 1; fi
if [ ! -f "${SPLIT_CFG}" ]; then echo "ERROR: missing SPLIT_CFG file: ${SPLIT_CFG}" >&2; exit 1; fi
if [ ! -d "${COMQ_DIR}/labels" ] || [ ! -d "${COMQ_DIR}/agreement" ]; then
  echo "ERROR: missing ${COMQ_DIR}/{labels,agreement} (run scripts/run_graphlp_confidence_ensemble_experiments.sh first)" >&2
  exit 1
fi
if [ ! -d "${CQ_DIR}/labels" ] || [ ! -d "${CQ_DIR}/agreement" ]; then
  echo "ERROR: missing ${CQ_DIR}/{labels,agreement} (run scripts/run_graphlp_confidence_ensemble_experiments.sh first)" >&2
  exit 1
fi

TRAIN_COMMON=(--mode train --data_root "${DATA_ROOT}" --split_cfg "${SPLIT_CFG}" --epochs "${EPOCHS}" --batch_size "${BATCH_SIZE}" --num_workers "${NUM_WORKERS}" --lr "${LR}" --roi_x "${ROI_X}" --roi_y "${ROI_Y}" --roi_z "${ROI_Z}" --norm clip_zscore --net basicunet --init scratch --seed "${SEED}")

printf -v LAST_EPOCH_FILE "epoch_%03d.json" "${EPOCHS}"

run_one() {
  local gpu_id="$1"
  local exp_name="$2"
  local ens="$3"          # COMQ or CQ
  local mode="$4"         # decoupled|table
  local scope="$5"        # dataset|batch
  local spec="$6"         # gamma_table (decoupled) or weight_table (table)

  local label_dir=""
  local agree_dir=""
  if [ "${ens}" = "COMQ" ]; then
    label_dir="${COMQ_DIR}/labels"
    agree_dir="${COMQ_DIR}"
  elif [ "${ens}" = "CQ" ]; then
    label_dir="${CQ_DIR}/labels"
    agree_dir="${CQ_DIR}"
  else
    echo "ERROR: unknown ensemble key: ${ens}" >&2
    return 2
  fi

  local out_dir="${OUT_ROOT}/${exp_name}"

  if [ "${RESUME}" -eq 1 ] && [ -f "${out_dir}/metrics/${LAST_EPOCH_FILE}" ] && [ -f "${out_dir}/last.ckpt" ]; then
    echo "  [GPU ${gpu_id}] SKIP (resume): ${exp_name}"
    return 0
  fi

  # NOTE: env-var prefixes like "CUDA_VISIBLE_DEVICES=0 python ..." must be parsed by bash.
  # When executing via an argv array ("${cmd[@]}"), the assignment would be treated as a literal
  # command name. Use `env` to make this robust.
  local cmd=(env CUDA_VISIBLE_DEVICES="${gpu_id}" "${PY}" train_finetune_wp5.py "${TRAIN_COMMON[@]}" --train_label_override_dir "${label_dir}" --train_label_agreement_dir "${agree_dir}" --agree_gt_sentinel "${GT_SENTINEL}" --agree_weight_mode "${mode}" --agree_imbalance_scope "${scope}" --output_dir "${out_dir}")
  if [ "${mode}" = "decoupled" ]; then
    cmd+=(--agree_gamma_table "${spec}")
  elif [ "${mode}" = "table" ]; then
    cmd+=(--agree_weight_table "${spec}")
  else
    echo "ERROR: unknown agree_weight_mode: ${mode}" >&2
    return 2
  fi

  if [ "${DRY_RUN}" -eq 1 ]; then
    echo "  [GPU ${gpu_id}] DRY-RUN: ${exp_name}"
    printf "    %q " "${cmd[@]}"
    echo ""
  else
    echo "  [GPU ${gpu_id}] START: ${exp_name} ($(date))"
    printf "    %q " "${cmd[@]}"
    echo ""
    "${cmd[@]}"
    echo "  [GPU ${gpu_id}] DONE:  ${exp_name} ($(date))"
  fi
}

# Experiment list:
# Format: name|ens|mode|scope|spec
EXPS=(
  # COMQ (K=4) decoupled: 10 gamma tables × 2 imbalance scopes = 20 runs
  "COMQ_dec_ds_g4_0p10_g3_0p05_g2_0p00|COMQ|decoupled|dataset|4:0.10,3:0.05,2:0.00,1:0.00"
  "COMQ_dec_bs_g4_0p10_g3_0p05_g2_0p00|COMQ|decoupled|batch|4:0.10,3:0.05,2:0.00,1:0.00"
  "COMQ_dec_ds_g4_0p15_g3_0p075_g2_0p00|COMQ|decoupled|dataset|4:0.15,3:0.075,2:0.00,1:0.00"
  "COMQ_dec_bs_g4_0p15_g3_0p075_g2_0p00|COMQ|decoupled|batch|4:0.15,3:0.075,2:0.00,1:0.00"
  "COMQ_dec_ds_g4_0p20_g3_0p10_g2_0p00|COMQ|decoupled|dataset|4:0.20,3:0.10,2:0.00,1:0.00"
  "COMQ_dec_bs_g4_0p20_g3_0p10_g2_0p00|COMQ|decoupled|batch|4:0.20,3:0.10,2:0.00,1:0.00"
  "COMQ_dec_ds_g4_0p25_g3_0p125_g2_0p00|COMQ|decoupled|dataset|4:0.25,3:0.125,2:0.00,1:0.00"
  "COMQ_dec_bs_g4_0p25_g3_0p125_g2_0p00|COMQ|decoupled|batch|4:0.25,3:0.125,2:0.00,1:0.00"
  "COMQ_dec_ds_g4_0p30_g3_0p15_g2_0p00|COMQ|decoupled|dataset|4:0.30,3:0.15,2:0.00,1:0.00"
  "COMQ_dec_bs_g4_0p30_g3_0p15_g2_0p00|COMQ|decoupled|batch|4:0.30,3:0.15,2:0.00,1:0.00"
  "COMQ_dec_ds_g4_0p20_g3_0p05_g2_0p00|COMQ|decoupled|dataset|4:0.20,3:0.05,2:0.00,1:0.00"
  "COMQ_dec_bs_g4_0p20_g3_0p05_g2_0p00|COMQ|decoupled|batch|4:0.20,3:0.05,2:0.00,1:0.00"
  "COMQ_dec_ds_g4_0p20_g3_0p15_g2_0p00|COMQ|decoupled|dataset|4:0.20,3:0.15,2:0.00,1:0.00"
  "COMQ_dec_bs_g4_0p20_g3_0p15_g2_0p00|COMQ|decoupled|batch|4:0.20,3:0.15,2:0.00,1:0.00"
  "COMQ_dec_ds_g4_0p20_g3_0p10_g2_0p01|COMQ|decoupled|dataset|4:0.20,3:0.10,2:0.01,1:0.00"
  "COMQ_dec_bs_g4_0p20_g3_0p10_g2_0p01|COMQ|decoupled|batch|4:0.20,3:0.10,2:0.01,1:0.00"
  "COMQ_dec_ds_g4_0p20_g3_0p10_g2_0p02|COMQ|decoupled|dataset|4:0.20,3:0.10,2:0.02,1:0.00"
  "COMQ_dec_bs_g4_0p20_g3_0p10_g2_0p02|COMQ|decoupled|batch|4:0.20,3:0.10,2:0.02,1:0.00"
  "COMQ_dec_ds_g4_0p30_g3_0p15_g2_0p02|COMQ|decoupled|dataset|4:0.30,3:0.15,2:0.02,1:0.00"
  "COMQ_dec_bs_g4_0p30_g3_0p15_g2_0p02|COMQ|decoupled|batch|4:0.30,3:0.15,2:0.02,1:0.00"

  # CQ (K=2) decoupled: 10 gamma tables × 2 imbalance scopes = 20 runs
  "CQ_dec_ds_g2_0p05_g1_0p00|CQ|decoupled|dataset|2:0.05,1:0.00"
  "CQ_dec_bs_g2_0p05_g1_0p00|CQ|decoupled|batch|2:0.05,1:0.00"
  "CQ_dec_ds_g2_0p10_g1_0p00|CQ|decoupled|dataset|2:0.10,1:0.00"
  "CQ_dec_bs_g2_0p10_g1_0p00|CQ|decoupled|batch|2:0.10,1:0.00"
  "CQ_dec_ds_g2_0p15_g1_0p00|CQ|decoupled|dataset|2:0.15,1:0.00"
  "CQ_dec_bs_g2_0p15_g1_0p00|CQ|decoupled|batch|2:0.15,1:0.00"
  "CQ_dec_ds_g2_0p20_g1_0p00|CQ|decoupled|dataset|2:0.20,1:0.00"
  "CQ_dec_bs_g2_0p20_g1_0p00|CQ|decoupled|batch|2:0.20,1:0.00"
  "CQ_dec_ds_g2_0p25_g1_0p00|CQ|decoupled|dataset|2:0.25,1:0.00"
  "CQ_dec_bs_g2_0p25_g1_0p00|CQ|decoupled|batch|2:0.25,1:0.00"
  "CQ_dec_ds_g2_0p30_g1_0p00|CQ|decoupled|dataset|2:0.30,1:0.00"
  "CQ_dec_bs_g2_0p30_g1_0p00|CQ|decoupled|batch|2:0.30,1:0.00"
  "CQ_dec_ds_g2_0p35_g1_0p00|CQ|decoupled|dataset|2:0.35,1:0.00"
  "CQ_dec_bs_g2_0p35_g1_0p00|CQ|decoupled|batch|2:0.35,1:0.00"
  "CQ_dec_ds_g2_0p20_g1_0p01|CQ|decoupled|dataset|2:0.20,1:0.01"
  "CQ_dec_bs_g2_0p20_g1_0p01|CQ|decoupled|batch|2:0.20,1:0.01"
  "CQ_dec_ds_g2_0p20_g1_0p02|CQ|decoupled|dataset|2:0.20,1:0.02"
  "CQ_dec_bs_g2_0p20_g1_0p02|CQ|decoupled|batch|2:0.20,1:0.02"
  "CQ_dec_ds_g2_0p25_g1_0p02|CQ|decoupled|dataset|2:0.25,1:0.02"
  "CQ_dec_bs_g2_0p25_g1_0p02|CQ|decoupled|batch|2:0.25,1:0.02"
)

echo "========================================================================"
echo "GraphLP Confidence-Ensemble Loss-Weight Sweep"
echo "========================================================================"
echo "Output base:     ${OUT_BASE}"
echo "Output root:     ${OUT_ROOT}"
echo "Total runs:      ${#EXPS[@]}"
echo "Epochs:          ${EPOCHS}"
echo "Batch size:      ${BATCH_SIZE}"
echo "LR:              ${LR}"
echo "ROI:             ${ROI_X},${ROI_Y},${ROI_Z}"
echo "Dry run:         $([ "${DRY_RUN}" -eq 1 ] && echo YES || echo NO)"
echo "Resume:          $([ "${RESUME}" -eq 1 ] && echo YES || echo NO)"
echo "========================================================================"

run_gpu_jobs() {
  local gpu_id="$1"
  shift
  local exps=( "$@" )

  echo ""
  echo "=========================================="
  echo "GPU ${gpu_id} - ${#exps[@]} runs"
  echo "=========================================="

  local i=1
  for spec in "${exps[@]}"; do
    IFS='|' read -r name ens mode scope table_or_gamma <<<"${spec}"
    echo ""
    echo "[GPU ${gpu_id}] ${i}/${#exps[@]}: ${name}"
    run_one "${gpu_id}" "${name}" "${ens}" "${mode}" "${scope}" "${table_or_gamma}"
    i=$((i+1))
  done
}

# Split exps across GPUs by index parity to balance COMQ/CQ workloads.
EXPS_GPU0=()
EXPS_GPU1=()
for i in "${!EXPS[@]}"; do
  if (( i % 2 == 0 )); then
    EXPS_GPU0+=( "${EXPS[$i]}" )
  else
    EXPS_GPU1+=( "${EXPS[$i]}" )
  fi
done

run_gpu_jobs 0 "${EXPS_GPU0[@]}" &
run_gpu_jobs 1 "${EXPS_GPU1[@]}" &
wait

echo ""
echo "DONE. Results are under: ${OUT_ROOT}"
