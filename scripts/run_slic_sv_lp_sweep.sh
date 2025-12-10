#!/usr/bin/env bash
set -euo pipefail

# Config
DATA_ROOT=${DATA_ROOT:-/data3/wp5_4_Dec_data/3ddl-dataset}
DATALIST=${DATALIST:-datalist_train_new.json}
SEEDS_DIR=${SEEDS_DIR:-runs/strategic_sparse_0p1pct_new/strategic_seeds}
WORKERS=${WORKERS:-16}
MODE=${MODE:-slic}
COMPACTNESS=${COMPACTNESS:-0.05}
SIGMA=${SIGMA:-1.0}
K=${K:-10}
ALPHA=${ALPHA:-0.9}
NUM_CLASSES=${NUM_CLASSES:-5}
IGNORE_CLASS=${IGNORE_CLASS:-6}
SEED=${SEED:-42}

SEGMENTS=${SEGMENTS:-"6000 8000 10000 12000 14000 16000 18000 20000"}

echo "DATA_ROOT=$DATA_ROOT"
echo "DATALIST=$DATALIST"
echo "SEEDS_DIR=$SEEDS_DIR"
echo "WORKERS=$WORKERS"
echo "MODE=$MODE COMPACTNESS=$COMPACTNESS SIGMA=$SIGMA"
echo "Graph LP: k=$K alpha=$ALPHA classes=$NUM_CLASSES"
echo "Segments: $SEGMENTS"

for N in $SEGMENTS; do
  SV_OUT="runs/sv_fullgt_${MODE}_n${N}_new_ras"
  SV_EVAL="${SV_OUT}_eval"
  LP_OUT="runs/graph_lp_prop_0p1pct_k${K}_a${ALPHA}_new_n${N}"

  echo
  echo "=== n_segments=$N ==="

  # 1) Generate supervoxels with full-GT voting
  python3 scripts/gen_supervoxels_wp5.py \
    --data-root "$DATA_ROOT" \
    --out-dir "$SV_OUT" \
    --n-segments "$N" \
    --compactness "$COMPACTNESS" \
    --sigma "$SIGMA" \
    --mode "$MODE" \
    --datalist "$DATALIST" \
    --ref-header label \
    --assign-all-from-gt \
    --ignore-class "$IGNORE_CLASS" \
    --num-workers "$WORKERS"

  # 2) Evaluate SV-voted labels vs GT
  python3 scripts/eval_sv_voted_wp5.py \
    --sv-dir "$SV_OUT" \
    --sv-ids-dir "$SV_OUT" \
    --datalist "$DATALIST" \
    --data-root "$DATA_ROOT/data" \
    --output_dir "$SV_EVAL" \
    --ignore-class "$IGNORE_CLASS" \
    --num_workers "$WORKERS" \
    --progress \
    --log_to_file \
    --heavy \
    --hd_percentile 95

  # 3) Graph LP propagation + evaluation
  python3 scripts/pipeline_graph_lp_sv.py \
    --sv_dir "$SV_OUT" \
    --seeds_dir "$SEEDS_DIR" \
    --datalist "$DATALIST" \
    --data_root "$DATA_ROOT/data" \
    --output_dir "$LP_OUT" \
    --k "$K" \
    --alpha "$ALPHA" \
    --num_classes "$NUM_CLASSES" \
    --seed "$SEED" \
    --ignore_class "$IGNORE_CLASS" \
    --lp_num_workers "$WORKERS" \
    --eval_num_workers "$WORKERS" \
    --eval_heavy \
    --eval_hd_percentile 95 \
    --eval_progress \
    --eval_log_to_file
done

echo "Sweep complete."
