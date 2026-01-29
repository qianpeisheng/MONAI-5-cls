#!/usr/bin/env bash
set -euo pipefail

# Graph-LP confidence/ensemble experiments (2 GPUs: 0 and 1)
#
# This script only *lists* and *runs* the commands; review before running.
# Intended workflow:
#   1) Build ensemble pseudo labels + confidence/source masks
#   2) (Optional) Evaluate pseudo-label quality vs train GT
#   3) Train multiple models in parallel on GPU0/GPU1
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
echo "STEP 1: Build ensemble pseudo labels + confidence/source masks"
echo "============================================================"

# vote(C,O,M,Q) with tie-break=Q
python3 scripts/build_graph_lp_ensemble_labels.py --datalist "${DATALIST}" --out_dir runs/graph_lp_ens_vote_COMQ_tieQ_thr4 --label_dir C="${C_RUN}" --label_dir O="${O_RUN}" --label_dir M="${M_RUN}" --label_dir Q="${Q_RUN}" --tie_break Q --seed_source_mask_dir "${Q_SRC}" --confidence_threshold 4 --write_source_masks --write_confidence_maps --write_filtered_labels --ignore_label 6
python3 scripts/build_graph_lp_ensemble_labels.py --datalist "${DATALIST}" --out_dir runs/graph_lp_ens_vote_COMQ_tieQ_thr3 --label_dir C="${C_RUN}" --label_dir O="${O_RUN}" --label_dir M="${M_RUN}" --label_dir Q="${Q_RUN}" --tie_break Q --seed_source_mask_dir "${Q_SRC}" --confidence_threshold 3 --write_source_masks --write_confidence_maps --write_filtered_labels --ignore_label 6

# Keep labels exactly as Q, but derive confidence from agreement between C and Q (maxc==2 means C==Q).
# vote(C,Q) with tie-break=Q => output labels == Q everywhere; confidence map tells where C agrees.
python3 scripts/build_graph_lp_ensemble_labels.py --datalist "${DATALIST}" --out_dir runs/graph_lp_q_mask_agree_CQ_thr2 --label_dir C="${C_RUN}" --label_dir Q="${Q_RUN}" --tie_break Q --seed_source_mask_dir "${Q_SRC}" --confidence_threshold 2 --write_source_masks --write_confidence_maps --write_filtered_labels --ignore_label 6

echo "============================================================"
echo "STEP 2 (optional): Evaluate pseudo-label quality vs train GT"
echo "============================================================"

python3 scripts/eval_sv_voted_wp5.py --sv-dir runs/graph_lp_ens_vote_COMQ_tieQ_thr4/labels --sv-ids-dir "${SV_IDS_DIR}" --datalist "${DATALIST}" --output_dir runs/graph_lp_ens_vote_COMQ_tieQ_thr4/eval --ignore-class 6 --num_workers 16 --progress --log_to_file
python3 scripts/eval_sv_voted_wp5.py --sv-dir runs/graph_lp_ens_vote_COMQ_tieQ_thr3/labels --sv-ids-dir "${SV_IDS_DIR}" --datalist "${DATALIST}" --output_dir runs/graph_lp_ens_vote_COMQ_tieQ_thr3/eval --ignore-class 6 --num_workers 16 --progress --log_to_file
python3 scripts/eval_sv_voted_wp5.py --sv-dir runs/graph_lp_q_mask_agree_CQ_thr2/labels --sv-ids-dir "${SV_IDS_DIR}" --datalist "${DATALIST}" --output_dir runs/graph_lp_q_mask_agree_CQ_thr2/eval --ignore-class 6 --num_workers 16 --progress --log_to_file

echo "============================================================"
echo "STEP 3: Train models (GPU0 + GPU1 in parallel)"
echo "============================================================"

# Common training args
TRAIN_COMMON=(--mode train --data_root "${DATA_ROOT}" --split_cfg "${SPLIT_CFG}" --epochs 40 --batch_size 4 --num_workers 8 --lr 0.001 --roi_x 112 --roi_y 112 --roi_z 80 --norm clip_zscore --net basicunet --init scratch --seed 42)

# Loss weighting knobs (tune):
# - source mask is interpreted as \"reliable\" (1) vs \"uncertain\" (0) voxels
# - gamma = --source_lp_quality controls total weight-mass ratio of uncertain vs reliable voxels

TS1="$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES=0 python3 train_finetune_wp5.py "${TRAIN_COMMON[@]}" --train_label_override_dir runs/graph_lp_ens_vote_COMQ_tieQ_thr4/labels --train_label_source_dir runs/graph_lp_ens_vote_COMQ_tieQ_thr4/source_masks --source_weight_mode decoupled --source_imbalance_scope dataset --source_lp_quality 0.05 --output_dir "runs/train_graphlp_conf_COMQ_thr4_decoupled_dataset_gamma0.05_${TS1}_gpu0" &
CUDA_VISIBLE_DEVICES=1 python3 train_finetune_wp5.py "${TRAIN_COMMON[@]}" --train_label_override_dir runs/graph_lp_ens_vote_COMQ_tieQ_thr3/labels --train_label_source_dir runs/graph_lp_ens_vote_COMQ_tieQ_thr3/source_masks --source_weight_mode decoupled --source_imbalance_scope dataset --source_lp_quality 0.05 --output_dir "runs/train_graphlp_conf_COMQ_thr3_decoupled_dataset_gamma0.05_${TS1}_gpu1" &
wait

TS2="$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES=0 python3 train_finetune_wp5.py "${TRAIN_COMMON[@]}" --train_label_override_dir runs/graph_lp_ens_vote_COMQ_tieQ_thr4/labels --train_label_source_dir runs/graph_lp_ens_vote_COMQ_tieQ_thr4/source_masks --source_weight_mode decoupled --source_imbalance_scope dataset --source_lp_quality 0.10 --output_dir "runs/train_graphlp_conf_COMQ_thr4_decoupled_dataset_gamma0.10_${TS2}_gpu0" &
CUDA_VISIBLE_DEVICES=1 python3 train_finetune_wp5.py "${TRAIN_COMMON[@]}" --train_label_override_dir runs/graph_lp_ens_vote_COMQ_tieQ_thr3/labels --train_label_source_dir runs/graph_lp_ens_vote_COMQ_tieQ_thr3/source_masks --source_weight_mode decoupled --source_imbalance_scope dataset --source_lp_quality 0.10 --output_dir "runs/train_graphlp_conf_COMQ_thr3_decoupled_dataset_gamma0.10_${TS2}_gpu1" &
wait

# Agreement-mask variant: labels==Q, but downweight/ignore the C!=Q disagreement region via source masks.
TS3="$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES=0 python3 train_finetune_wp5.py "${TRAIN_COMMON[@]}" --train_label_override_dir runs/graph_lp_q_mask_agree_CQ_thr2/labels --train_label_source_dir runs/graph_lp_q_mask_agree_CQ_thr2/source_masks --source_weight_mode decoupled --source_imbalance_scope dataset --source_lp_quality 0.05 --output_dir "runs/train_graphlp_conf_Q_agreeCQ_decoupled_dataset_gamma0.05_${TS3}_gpu0" &
CUDA_VISIBLE_DEVICES=1 python3 train_finetune_wp5.py "${TRAIN_COMMON[@]}" --train_label_override_dir runs/graph_lp_q_mask_agree_CQ_thr2/labels --train_label_source_dir runs/graph_lp_q_mask_agree_CQ_thr2/source_masks --source_weight_mode decoupled --source_imbalance_scope dataset --source_lp_quality 0.10 --output_dir "runs/train_graphlp_conf_Q_agreeCQ_decoupled_dataset_gamma0.10_${TS3}_gpu1" &
wait

echo "============================================================"
echo "STEP 4 (optional): Hard-mask low-confidence voxels (set to ignore label 6)"
echo "============================================================"

# These runs use labels_filtered/ where low-confidence voxels are set to 6 (ignored in loss).
# We do NOT pass --train_label_source_dir here (keep it simple).
TS4="$(date +%Y%m%d-%H%M%S)"
CUDA_VISIBLE_DEVICES=0 python3 train_finetune_wp5.py "${TRAIN_COMMON[@]}" --train_label_override_dir runs/graph_lp_ens_vote_COMQ_tieQ_thr4/labels_filtered --output_dir "runs/train_graphlp_masked_COMQ_thr4_labelsFiltered_${TS4}_gpu0" &
CUDA_VISIBLE_DEVICES=1 python3 train_finetune_wp5.py "${TRAIN_COMMON[@]}" --train_label_override_dir runs/graph_lp_ens_vote_COMQ_tieQ_thr3/labels_filtered --output_dir "runs/train_graphlp_masked_COMQ_thr3_labelsFiltered_${TS4}_gpu1" &
wait

echo "DONE"

