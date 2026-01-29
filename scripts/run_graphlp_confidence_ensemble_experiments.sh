#!/usr/bin/env bash
set -euo pipefail

# Graph-LP ensemble experiments with agreement-count weighting (2 GPUs: 0 and 1)
#
# This script runs a small, reproducible grid. Review/edit before running.
# Intended workflow:
#   1) Build ensemble pseudo labels + per-voxel agree_count maps
#   2) (Optional) Evaluate pseudo-label quality vs train GT
#   3) Train models using dedicated agree_count weighting support in train_finetune_wp5.py
#
# Run:
#   bash scripts/run_graphlp_confidence_ensemble_experiments.sh

cd /home/peisheng/MONAI

# Optional: activate repo venv if it exists
if [ -f /home/peisheng/MONAI/venv/bin/activate ]; then . /home/peisheng/MONAI/venv/bin/activate; fi
PY="python3"
if [ -x /home/peisheng/MONAI/venv/bin/python ]; then PY="/home/peisheng/MONAI/venv/bin/python"; fi

DATA_ROOT="/data3/wp5_4_Dec_data/3ddl-dataset"
SPLIT_CFG="/data3/wp5_4_Dec_data/3ddl-dataset/data/dataset_config.json"
DATALIST="datalist_train_new.json"
SV_IDS_DIR="runs/sv_fullgt_slic_n12000_new_ras"
GT_SENTINEL=255

# Input pseudo-label runs (do not modify unless you know what you're doing)
C_RUN="runs/graph_lp_prop_0p1pct_k10_a0.9_new_n12000"
O_RUN="runs/graph_lp_prop_0p1pct_k10_a0.9_new_n12000_outerbg_adaptive"
M_RUN="runs/graph_lp_3desc_eval/graph_lp_3desc_20260109-162446_k10_a0.9_sigPhimedian/moments"
Q_RUN="runs/graph_lp_3desc_eval/graph_lp_3desc_20260109-162446_k10_a0.9_sigPhimedian/quantiles16"
Q_SRC="runs/graph_lp_3desc_eval/graph_lp_3desc_20260109-162446_k10_a0.9_sigPhimedian/quantiles16/source_masks"

if [ ! -d "${DATA_ROOT}" ]; then echo "ERROR: missing DATA_ROOT dir: ${DATA_ROOT}" >&2; exit 1; fi
if [ ! -f "${SPLIT_CFG}" ]; then echo "ERROR: missing SPLIT_CFG file: ${SPLIT_CFG}" >&2; exit 1; fi
if [ ! -f "${DATALIST}" ]; then echo "ERROR: missing DATALIST file: ${DATALIST}" >&2; exit 1; fi
if [ ! -d "${SV_IDS_DIR}" ]; then echo "ERROR: missing SV_IDS_DIR dir: ${SV_IDS_DIR}" >&2; exit 1; fi
if [ ! -d "${Q_SRC}" ]; then echo "ERROR: missing Q_SRC dir: ${Q_SRC}" >&2; exit 1; fi
if [ ! -d "${C_RUN}" ]; then echo "ERROR: missing C_RUN dir: ${C_RUN}" >&2; exit 1; fi
if [ ! -d "${O_RUN}" ]; then echo "ERROR: missing O_RUN dir: ${O_RUN}" >&2; exit 1; fi
if [ ! -d "${M_RUN}" ]; then echo "ERROR: missing M_RUN dir: ${M_RUN}" >&2; exit 1; fi
if [ ! -d "${Q_RUN}" ]; then echo "ERROR: missing Q_RUN dir: ${Q_RUN}" >&2; exit 1; fi

echo "============================================================"
echo "STEP 1: Build ensemble pseudo labels + agree_count maps"
echo "============================================================"

# Outputs (per case):
# - labels/<id>_labels.npy
# - agreement/<id>_agree_count.npy
#   * pseudo voxels: agree_count in [1..K]
#   * GT-supported SV voxels (from --seed_source_mask_dir): agree_count=${GT_SENTINEL}
# - propagation_summary.json (records K and gt_sentinel)

# vote(C,O,M,Q) with tie-break=Q (K=4)
"${PY}" scripts/build_graph_lp_ensemble_labels.py --datalist "${DATALIST}" --out_dir runs/graph_lp_ens_vote_COMQ_tieQ --label_dir C="${C_RUN}" --label_dir O="${O_RUN}" --label_dir M="${M_RUN}" --label_dir Q="${Q_RUN}" --tie_break Q --seed_source_mask_dir "${Q_SRC}" --gt_sentinel "${GT_SENTINEL}" --write_source_masks

# vote(C,Q) with tie-break=Q (K=2) => output labels == Q everywhere; agree_count indicates agreement (2) vs disagreement (1).
"${PY}" scripts/build_graph_lp_ensemble_labels.py --datalist "${DATALIST}" --out_dir runs/graph_lp_ens_vote_CQ_tieQ --label_dir C="${C_RUN}" --label_dir Q="${Q_RUN}" --tie_break Q --seed_source_mask_dir "${Q_SRC}" --gt_sentinel "${GT_SENTINEL}" --write_source_masks

echo "============================================================"
echo "STEP 2 (optional): Evaluate pseudo-label quality vs train GT"
echo "============================================================"

"${PY}" scripts/eval_sv_voted_wp5.py --sv-dir runs/graph_lp_ens_vote_COMQ_tieQ/labels --sv-ids-dir "${SV_IDS_DIR}" --datalist "${DATALIST}" --output_dir runs/graph_lp_ens_vote_COMQ_tieQ/eval --ignore-class 6 --num_workers 16 --progress --log_to_file
"${PY}" scripts/eval_sv_voted_wp5.py --sv-dir runs/graph_lp_ens_vote_CQ_tieQ/labels --sv-ids-dir "${SV_IDS_DIR}" --datalist "${DATALIST}" --output_dir runs/graph_lp_ens_vote_CQ_tieQ/eval --ignore-class 6 --num_workers 16 --progress --log_to_file

echo "============================================================"
echo "STEP 3: Train models (GPU0 + GPU1 in parallel)"
echo "============================================================"

# Common training args
TRAIN_COMMON=(--mode train --data_root "${DATA_ROOT}" --split_cfg "${SPLIT_CFG}" --epochs 40 --batch_size 4 --num_workers 8 --lr 0.001 --roi_x 112 --roi_y 112 --roi_z 80 --norm clip_zscore --net basicunet --init scratch --seed 42)

# Agree-count weighting knobs (tune):
# - agree_count==${GT_SENTINEL} is treated as the GT tier (seed-supported SVs), weight fixed to 1.0
# - agree_count in [1..K] are pseudo voxels; weights are controlled by:
#   * --agree_weight_mode table --agree_weight_table "count:weight,..."
#   * or --agree_weight_mode decoupled --agree_gamma_table "count:gamma,..."
#
# In decoupled mode, gamma is interpreted as a *weight-mass ratio* per tier:
#   (w_c * N_c) / (w_gt * N_gt) ~= gamma_c
# which makes it easier to tune/report than raw per-voxel weights.

TS1="$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES=0 "${PY}" train_finetune_wp5.py "${TRAIN_COMMON[@]}" --train_label_override_dir runs/graph_lp_ens_vote_COMQ_tieQ/labels --train_label_agreement_dir runs/graph_lp_ens_vote_COMQ_tieQ --agree_gt_sentinel "${GT_SENTINEL}" --agree_weight_mode decoupled --agree_imbalance_scope dataset --agree_gamma_table "4:0.20,3:0.10,2:0.00,1:0.00" --output_dir "runs/train_graphlp_agree_COMQ_decoupled_gammaA_${TS1}_gpu0" &
CUDA_VISIBLE_DEVICES=1 "${PY}" train_finetune_wp5.py "${TRAIN_COMMON[@]}" --train_label_override_dir runs/graph_lp_ens_vote_COMQ_tieQ/labels --train_label_agreement_dir runs/graph_lp_ens_vote_COMQ_tieQ --agree_gt_sentinel "${GT_SENTINEL}" --agree_weight_mode decoupled --agree_imbalance_scope dataset --agree_gamma_table "4:0.30,3:0.15,2:0.02,1:0.00" --output_dir "runs/train_graphlp_agree_COMQ_decoupled_gammaB_${TS1}_gpu1" &
wait

TS2="$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES=0 "${PY}" train_finetune_wp5.py "${TRAIN_COMMON[@]}" --train_label_override_dir runs/graph_lp_ens_vote_CQ_tieQ/labels --train_label_agreement_dir runs/graph_lp_ens_vote_CQ_tieQ --agree_gt_sentinel "${GT_SENTINEL}" --agree_weight_mode table --agree_weight_table "2:0.20,1:0.00" --output_dir "runs/train_graphlp_agree_CQ_table_${TS2}_gpu0" &
CUDA_VISIBLE_DEVICES=1 "${PY}" train_finetune_wp5.py "${TRAIN_COMMON[@]}" --train_label_override_dir runs/graph_lp_ens_vote_CQ_tieQ/labels --train_label_agreement_dir runs/graph_lp_ens_vote_CQ_tieQ --agree_gt_sentinel "${GT_SENTINEL}" --agree_weight_mode table --agree_weight_table "2:0.20,1:0.02" --output_dir "runs/train_graphlp_agree_CQ_table_soft_${TS2}_gpu1" &
wait

echo "DONE"
