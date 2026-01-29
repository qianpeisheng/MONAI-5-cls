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

DATA_ROOT="/data3/wp5_4_Dec_data/3ddl-dataset"
SPLIT_CFG="/data3/wp5_4_Dec_data/3ddl-dataset/data/dataset_config.json"
DATALIST="datalist_train_new.json"
SV_IDS_DIR="runs/sv_fullgt_slic_n12000_new_ras"

# Input pseudo-label runs (do not modify unless you know what you're doing)
C_RUN="runs/graph_lp_prop_0p1pct_k10_a0.9_new_n12000"
O_RUN="runs/graph_lp_prop_0p1pct_k10_a0.9_new_n12000_outerbg_adaptive"
M_RUN="runs/graph_lp_3desc_eval/graph_lp_3desc_20260109-162446_k10_a0.9_sigPhimedian/moments"
Q_RUN="runs/graph_lp_3desc_eval/graph_lp_3desc_20260109-162446_k10_a0.9_sigPhimedian/quantiles16"
Q_SRC="runs/graph_lp_3desc_eval/graph_lp_3desc_20260109-162446_k10_a0.9_sigPhimedian/quantiles16/source_masks"

echo "============================================================"
echo "STEP 1: Build ensemble pseudo labels + agree_count maps"
echo "============================================================"

# vote(C,O,M,Q) with tie-break=Q (K=4)
python3 scripts/build_graph_lp_ensemble_labels.py --datalist "${DATALIST}" --out_dir runs/graph_lp_ens_vote_COMQ_tieQ --label_dir C="${C_RUN}" --label_dir O="${O_RUN}" --label_dir M="${M_RUN}" --label_dir Q="${Q_RUN}" --tie_break Q --seed_source_mask_dir "${Q_SRC}" --write_source_masks

# vote(C,Q) with tie-break=Q (K=2) => output labels == Q everywhere; agree_count indicates agreement (2) vs disagreement (1).
python3 scripts/build_graph_lp_ensemble_labels.py --datalist "${DATALIST}" --out_dir runs/graph_lp_ens_vote_CQ_tieQ --label_dir C="${C_RUN}" --label_dir Q="${Q_RUN}" --tie_break Q --seed_source_mask_dir "${Q_SRC}" --write_source_masks

echo "============================================================"
echo "STEP 2 (optional): Evaluate pseudo-label quality vs train GT"
echo "============================================================"

python3 scripts/eval_sv_voted_wp5.py --sv-dir runs/graph_lp_ens_vote_COMQ_tieQ/labels --sv-ids-dir "${SV_IDS_DIR}" --datalist "${DATALIST}" --output_dir runs/graph_lp_ens_vote_COMQ_tieQ/eval --ignore-class 6 --num_workers 16 --progress --log_to_file
python3 scripts/eval_sv_voted_wp5.py --sv-dir runs/graph_lp_ens_vote_CQ_tieQ/labels --sv-ids-dir "${SV_IDS_DIR}" --datalist "${DATALIST}" --output_dir runs/graph_lp_ens_vote_CQ_tieQ/eval --ignore-class 6 --num_workers 16 --progress --log_to_file

echo "============================================================"
echo "STEP 3: Train models (GPU0 + GPU1 in parallel)"
echo "============================================================"

# Common training args
TRAIN_COMMON=(--mode train --data_root "${DATA_ROOT}" --split_cfg "${SPLIT_CFG}" --epochs 40 --batch_size 4 --num_workers 8 --lr 0.001 --roi_x 112 --roi_y 112 --roi_z 80 --norm clip_zscore --net basicunet --init scratch --seed 42)

# Agree-count weighting knobs (tune):
# - agree_count==255 is treated as the GT tier (seed-supported SVs), weight fixed to 1.0
# - agree_count in [1..K] are pseudo voxels; weights are controlled by:
#   * --agree_weight_mode table --agree_weight_table "count:weight,..."
#   * or --agree_weight_mode decoupled --agree_gamma_table "count:gamma,..."

TS1="$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES=0 python3 train_finetune_wp5.py "${TRAIN_COMMON[@]}" --train_label_override_dir runs/graph_lp_ens_vote_COMQ_tieQ/labels --train_label_agreement_dir runs/graph_lp_ens_vote_COMQ_tieQ --agree_weight_mode decoupled --agree_imbalance_scope dataset --agree_gamma_table "4:0.20,3:0.10,2:0.02,1:0.00" --output_dir "runs/train_graphlp_agree_COMQ_decoupled_gammaA_${TS1}_gpu0" &
CUDA_VISIBLE_DEVICES=1 python3 train_finetune_wp5.py "${TRAIN_COMMON[@]}" --train_label_override_dir runs/graph_lp_ens_vote_COMQ_tieQ/labels --train_label_agreement_dir runs/graph_lp_ens_vote_COMQ_tieQ --agree_weight_mode decoupled --agree_imbalance_scope dataset --agree_gamma_table "4:0.30,3:0.15,2:0.03,1:0.00" --output_dir "runs/train_graphlp_agree_COMQ_decoupled_gammaB_${TS1}_gpu1" &
wait

TS2="$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES=0 python3 train_finetune_wp5.py "${TRAIN_COMMON[@]}" --train_label_override_dir runs/graph_lp_ens_vote_CQ_tieQ/labels --train_label_agreement_dir runs/graph_lp_ens_vote_CQ_tieQ --agree_weight_mode table --agree_weight_table "2:0.20,1:0.00" --output_dir "runs/train_graphlp_agree_CQ_table_${TS2}_gpu0" &
CUDA_VISIBLE_DEVICES=1 python3 train_finetune_wp5.py "${TRAIN_COMMON[@]}" --train_label_override_dir runs/graph_lp_ens_vote_CQ_tieQ/labels --train_label_agreement_dir runs/graph_lp_ens_vote_CQ_tieQ --agree_weight_mode table --agree_weight_table "2:0.20,1:0.02" --output_dir "runs/train_graphlp_agree_CQ_table_soft_${TS2}_gpu1" &
wait

echo "DONE"
