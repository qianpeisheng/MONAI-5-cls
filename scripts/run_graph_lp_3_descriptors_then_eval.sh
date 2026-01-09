#!/usr/bin/env bash
set -euo pipefail

# Runs Graph LP pseudo-label generation using 3 intensity descriptors (moments, quantiles16, hist32)
# AND evaluates each result vs train GT using scripts/eval_sv_voted_wp5.py.
#
# Notes:
# - Intensity descriptors are used *in addition to* the centroid-coordinate spatial term:
#   w_ij = exp(-||c_i-c_j||^2/(2*sigma_c^2)) * exp(-d(phi_i,phi_j)^2/(2*sigma_phi^2))
# - The old coords-only mode is still supported via: --descriptor_type none
#
# Usage (example):
#   bash scripts/run_graph_lp_3_descriptors_then_eval.sh
#
# Optional overrides (env vars):
#   PYTHON=python3
#   SV_DIR=runs/sv_fullgt_slic_n12000_new_ras
#   SEEDS_DIR=runs/strategic_sparse_0p1pct_new/strategic_seeds
#   DATALIST=datalist_train_new.json
#   DATA_ROOT=""  # only needed if datalist has relative paths
#   OUT_ROOT=runs/graph_lp_3desc_eval
#   CACHE_DIR=runs/cache_sv_descriptors
#   WORKERS=16
#   K=10
#   ALPHA=0.9
#   SIGMA_PHI=median
#   USE_COSINE=0  # set to 1 to use cosine for moments/quantiles
#   QUANTILES_INCLUDE_MAD=0
#   HIST_BINS=32
#   HIST_VMIN=-3
#   HIST_VMAX=3
#   MOMENTS_TRIM_RATIO=0.1
#   SAMPLE_EDGES_FOR_SIGMA=50000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"

SV_DIR="${SV_DIR:-runs/sv_fullgt_slic_n12000_new_ras}"
SEEDS_DIR="${SEEDS_DIR:-runs/strategic_sparse_0p1pct_new/strategic_seeds}"
DATALIST="${DATALIST:-datalist_train_new.json}"
DATA_ROOT="${DATA_ROOT:-}"

OUT_ROOT="${OUT_ROOT:-runs/graph_lp_3desc_eval}"
CACHE_DIR="${CACHE_DIR:-}"

WORKERS="${WORKERS:-16}"
K="${K:-10}"
ALPHA="${ALPHA:-0.9}"
NUM_CLASSES="${NUM_CLASSES:-5}"
SEED="${SEED:-42}"
IGNORE_CLASS="${IGNORE_CLASS:-6}"

SIGMA_PHI="${SIGMA_PHI:-median}"
USE_COSINE="${USE_COSINE:-0}"
QUANTILES_INCLUDE_MAD="${QUANTILES_INCLUDE_MAD:-0}"
HIST_BINS="${HIST_BINS:-32}"
HIST_VMIN="${HIST_VMIN:--3}"
HIST_VMAX="${HIST_VMAX:-3}"
MOMENTS_TRIM_RATIO="${MOMENTS_TRIM_RATIO:-0.1}"
SAMPLE_EDGES_FOR_SIGMA="${SAMPLE_EDGES_FOR_SIGMA:-50000}"

TS="$(date +%Y%m%d-%H%M%S)"
BASE_OUT="${OUT_ROOT}/graph_lp_3desc_${TS}_k${K}_a${ALPHA}_sigPhi${SIGMA_PHI}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "SV_DIR=${SV_DIR}"
echo "SEEDS_DIR=${SEEDS_DIR}"
echo "DATALIST=${DATALIST}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUT=${BASE_OUT}"
echo "Graph LP: k=${K} alpha=${ALPHA} classes=${NUM_CLASSES} seed=${SEED}"
echo "Descriptor: sigma_phi=${SIGMA_PHI} use_cosine=${USE_COSINE} quantiles_include_mad=${QUANTILES_INCLUDE_MAD}"
echo "Hist: bins=${HIST_BINS} range=[${HIST_VMIN},${HIST_VMAX}]"

mkdir -p "${BASE_OUT}"

run_one() {
  local name="$1"
  local dtype="$2"

  local run_dir="${BASE_OUT}/${name}"
  mkdir -p "${run_dir}"

  echo
  echo "======================================================================"
  echo "Graph LP: ${name} (descriptor_type=${dtype})"
  echo "======================================================================"

  cmd=(
    "${PYTHON}" scripts/propagate_graph_lp_multi_case.py
    --sv_dir "${SV_DIR}"
    --seeds_dir "${SEEDS_DIR}"
    --datalist "${DATALIST}"
    --output_dir "${run_dir}"
    --k "${K}"
    --alpha "${ALPHA}"
    --num_classes "${NUM_CLASSES}"
    --seed "${SEED}"
    --num_workers "${WORKERS}"
    --descriptor_type "${dtype}"
    --sigma_phi "${SIGMA_PHI}"
    --hist_bins "${HIST_BINS}"
    --hist_range "${HIST_VMIN}" "${HIST_VMAX}"
    --moments_trim_ratio "${MOMENTS_TRIM_RATIO}"
    --sample_edges_for_sigma "${SAMPLE_EDGES_FOR_SIGMA}"
  )

  if [[ -n "${DATA_ROOT}" ]]; then
    cmd+=(--data_root "${DATA_ROOT}")
  fi
  if [[ -n "${CACHE_DIR}" ]]; then
    cmd+=(--descriptor_cache_dir "${CACHE_DIR}")
  fi
  if [[ "${USE_COSINE}" == "1" ]]; then
    cmd+=(--use_cosine)
  fi
  if [[ "${QUANTILES_INCLUDE_MAD}" == "1" && "${dtype}" == "quantiles16" ]]; then
    cmd+=(--quantiles_include_mad)
  fi

  echo "Command: ${cmd[*]}"
  "${cmd[@]}"

  echo
  echo "======================================================================"
  echo "Eval: ${name} vs GT"
  echo "======================================================================"

  mkdir -p "${run_dir}/eval"
  eval_cmd=(
    "${PYTHON}" scripts/eval_sv_voted_wp5.py
    --sv-dir "${run_dir}/labels"
    --sv-ids-dir "${SV_DIR}"
    --datalist "${DATALIST}"
    --output_dir "${run_dir}/eval"
    --ignore-class "${IGNORE_CLASS}"
    --num_workers "${WORKERS}"
    --progress
    --log_to_file
  )
  if [[ -n "${DATA_ROOT}" ]]; then
    eval_cmd+=(--data-root "${DATA_ROOT}")
  fi

  echo "Command: ${eval_cmd[*]}"
  "${eval_cmd[@]}"

  # Print a compact summary from summary.json
  "${PYTHON}" -c "import json; p='${run_dir}/eval/metrics/summary.json'; j=json.load(open(p)); print('--- ${name} ---'); print('summary:', p); print('avg_dice:', j.get('average',{}).get('dice')); print('avg_iou:', j.get('average',{}).get('iou')); print('per_class_iou:', {k:v.get('iou') for k,v in j.get('per_class',{}).items()});"
}

run_one "moments" "moments"
run_one "quantiles16" "quantiles16"
run_one "hist32" "hist32"

echo
echo "DONE. Outputs:"
echo "  ${BASE_OUT}/moments/eval/metrics/summary.json"
echo "  ${BASE_OUT}/quantiles16/eval/metrics/summary.json"
echo "  ${BASE_OUT}/hist32/eval/metrics/summary.json"

