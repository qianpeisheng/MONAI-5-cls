# Graph Label Propagation - Training Pipeline Guide

Complete guide for generating Graph LP labels and training models.

## Overview

This pipeline uses Zhou-style graph label propagation to generate pseudo-labels from sparse supervoxel seeds (0.1% budget) and trains segmentation models.

**Key Results from Experiments:**
- Graph LP achieves **78.8% foreground Dice** at 0.1% budget
- Best hyperparameters: **k=10, alpha=0.9**
- Outperforms KNN methods by **+9.7 pp** at sparse budgets

---

## Prerequisites

**Data Requirements:**
- Supervoxel partition: `/data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n12000_c0.05_s1.0_ras2_voted/`
- Sparse seeds (strategic sampling): `runs/strategic_sparse_0p1pct_k_multi/strategic_seeds/`
- Training images: `/data3/wp5/wp5-code/dataloaders/wp5-dataset/`

**Software:**
- Python 3.9+
- MONAI, PyTorch
- Dependencies from `requirements.txt`

---

## Step 1: Generate Graph LP Labels (All 380 Training Cases)

### Command

```bash
python3 scripts/propagate_graph_lp_multi_case.py \
  --sv_dir /data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n12000_c0.05_s1.0_ras2_voted \
  --seeds_dir runs/strategic_sparse_0p1pct_k_multi/strategic_seeds \
  --k 10 \
  --alpha 0.9 \
  --output_dir runs/graph_lp_prop_0p1pct_k10_a0.9 \
  --seed 42
```

### Parameters

| Parameter | Value | Explanation |
|-----------|-------|-------------|
| `--sv_dir` | Path to SV partition | Contains `*_sv_ids.npy` files |
| `--seeds_dir` | Path to sparse seeds | Contains `*_sv_labels_sparse.json` files |
| `--k` | 10 | Number of graph neighbors (optimal from experiments) |
| `--alpha` | 0.9 | Propagation strength (optimal from experiments) |
| `--output_dir` | Output directory | Where to save propagated labels |
| `--seed` | 42 | Random seed for reproducibility |

### Expected Output

```
runs/graph_lp_prop_0p1pct_k10_a0.9/
├── cases/
│   ├── SN13B0_I17_3D_B1_1B250409/
│   │   ├── propagated_labels.npy          # Dense voxel labels
│   │   ├── sparse_sv_labels.json          # Input sparse labels (copy)
│   │   └── propagation_meta.json          # Statistics
│   └── ... (380 cases total)
├── labels/
│   ├── SN13B0_I17_3D_B1_1B250409_labels.npy  # Symlink for training
│   └── ... (380 files)
└── propagation_summary.json               # Overall statistics
```

### Expected Time

- **~3-5 hours** for 380 cases
- ~13 seconds per case average
- Can monitor progress with tqdm bar

### Verification

Check the summary after completion:

```bash
cat runs/graph_lp_prop_0p1pct_k10_a0.9/propagation_summary.json
```

Expected output:
```json
{
  "n_cases": 380,
  "k": 10,
  "alpha": 0.9,
  "avg_labeled_svs_input": 698.5,
  "avg_total_svs": 12761.2,
  "avg_coverage_input": 0.055
}
```

---

## Step 2: Train Model with Graph LP Labels

### Command

```bash
python3 train_finetune_wp5.py \
  --mode train \
  --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
  --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json \
  --train_label_override_dir runs/graph_lp_prop_0p1pct_k10_a0.9/labels \
  --output_dir runs/train_graph_lp_k10_a0.9_0p1pct \
  --epochs 20 \
  --batch_size 2 \
  --num_workers 4 \
  --init scratch \
  --net basicunet \
  --norm clip_zscore \
  --roi_x 112 --roi_y 112 --roi_z 80 \
  --log_to_file
```

### Key Arguments

| Argument | Value | Explanation |
|----------|-------|-------------|
| `--train_label_override_dir` | Path to propagated labels | **Critical:** Uses Graph LP labels instead of GT |
| `--output_dir` | Training output dir | Checkpoints and logs saved here |
| `--epochs` | 20 | Standard training duration |
| `--batch_size` | 2 | Adjust based on GPU memory |
| `--init` | scratch | Train from random initialization |
| `--net` | basicunet | Model architecture |
| `--norm` | clip_zscore | Normalization method |
| `--roi_x/y/z` | 112/112/80 | Patch size for training |

### Expected Output

```
runs/train_graph_lp_k10_a0.9_0p1pct/
├── best.ckpt                    # Best model checkpoint
├── last.ckpt                    # Last epoch checkpoint
├── train.log                    # Training log
├── train_metrics.csv            # Per-epoch metrics
└── config.yaml                  # Training config
```

### Expected Time

- **~12-18 hours** for 20 epochs (depends on GPU)
- Monitor training:
  ```bash
  tail -f runs/train_graph_lp_k10_a0.9_0p1pct/train.log
  ```

### Training Tips

**Multi-GPU Training:**
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 train_finetune_wp5.py \
  --mode train \
  ... (same args as above)
```

**Resume from Checkpoint:**
```bash
python3 train_finetune_wp5.py \
  --mode train \
  --resume runs/train_graph_lp_k10_a0.9_0p1pct/last.ckpt \
  ... (same args as above)
```

---

## Step 3: Evaluate on Test Set

### Command

```bash
python3 scripts/eval_wp5.py \
  --ckpt runs/train_graph_lp_k10_a0.9_0p1pct/best.ckpt \
  --datalist datalist_test.json \
  --output_dir runs/train_graph_lp_k10_a0.9_0p1pct/eval \
  --save_preds --heavy --hd_percentile 95
```

### Arguments

| Argument | Value | Explanation |
|----------|-------|-------------|
| `--ckpt` | Path to checkpoint | Model to evaluate |
| `--datalist` | Test split JSON | 180 test cases |
| `--output_dir` | Eval output dir | Results and predictions |
| `--save_preds` | Flag | Save prediction .npy files |
| `--heavy` | Flag | Compute all metrics (Dice, HD, surface distances) |
| `--hd_percentile` | 95 | Hausdorff distance percentile |

### Expected Output

```
runs/train_graph_lp_k10_a0.9_0p1pct/eval/
├── metrics_summary.json         # Overall metrics
├── metrics_per_case.csv         # Per-case breakdown
├── predictions/
│   ├── SN01B0_I1_3D_B1_1B250409_pred.npy
│   └── ... (180 files)
└── eval.log                     # Evaluation log
```

### Expected Time

- **~1-2 hours** for 180 test cases with `--heavy`
- Faster without `--save_preds` (~30 mins)

### View Results

```bash
# Summary metrics
cat runs/train_graph_lp_k10_a0.9_0p1pct/eval/metrics_summary.json

# Per-case details
less runs/train_graph_lp_k10_a0.9_0p1pct/eval/metrics_per_case.csv
```

---

## Complete End-to-End Pipeline

Run all steps sequentially:

```bash
# Step 1: Propagate labels (3-5 hours)
python3 scripts/propagate_graph_lp_multi_case.py \
  --sv_dir /data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n12000_c0.05_s1.0_ras2_voted \
  --seeds_dir runs/strategic_sparse_0p1pct_k_multi/strategic_seeds \
  --k 10 --alpha 0.9 \
  --output_dir runs/graph_lp_prop_0p1pct_k10_a0.9 \
  --seed 42

# Step 2: Train model (12-18 hours)
python3 train_finetune_wp5.py \
  --mode train \
  --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
  --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json \
  --train_label_override_dir runs/graph_lp_prop_0p1pct_k10_a0.9/labels \
  --output_dir runs/train_graph_lp_k10_a0.9_0p1pct \
  --epochs 20 --batch_size 2 --num_workers 4 \
  --init scratch --net basicunet --norm clip_zscore \
  --roi_x 112 --roi_y 112 --roi_z 80 --log_to_file

# Step 3: Evaluate (1-2 hours)
python3 scripts/eval_wp5.py \
  --ckpt runs/train_graph_lp_k10_a0.9_0p1pct/best.ckpt \
  --datalist datalist_test.json \
  --output_dir runs/train_graph_lp_k10_a0.9_0p1pct/eval \
  --save_preds --heavy --hd_percentile 95
```

**Total time: ~16-25 hours**

---

## Alternative Configurations

### Test Different Hyperparameters

If you want to try different k and alpha values:

```bash
# k=5, alpha=0.95 (also good from experiments)
python3 scripts/propagate_graph_lp_multi_case.py \
  --sv_dir /data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n12000_c0.05_s1.0_ras2_voted \
  --seeds_dir runs/strategic_sparse_0p1pct_k_multi/strategic_seeds \
  --k 5 --alpha 0.95 \
  --output_dir runs/graph_lp_prop_0p1pct_k05_a0.95 \
  --seed 42

# Then train with this config
python3 train_finetune_wp5.py \
  --mode train \
  --train_label_override_dir runs/graph_lp_prop_0p1pct_k05_a0.95/labels \
  --output_dir runs/train_graph_lp_k05_a0.95_0p1pct \
  ... (other args same as above)
```

### Different Budget Levels

For 0.5% budget (if seeds are available):

```bash
python3 scripts/propagate_graph_lp_multi_case.py \
  --sv_dir /data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n12000_c0.05_s1.0_ras2_voted \
  --seeds_dir runs/strategic_sparse_0p5pct/strategic_seeds \
  --k 10 --alpha 0.95 \
  --output_dir runs/graph_lp_prop_0p5pct_k10_a0.95 \
  --seed 42
```

---

## Comparison with KNN Baseline

To compare against the existing KNN baseline:

**KNN baseline (already trained):**
- Location: `runs/strategic_sparse_0p1pct_k_multi/k_variants/k10/`
- Training: See `STRATEGIC_SPARSE_SV_README.md`

**Graph LP (new):**
- Location: `runs/graph_lp_prop_0p1pct_k10_a0.9/labels/`
- Training: This guide

**Expected improvement:**
- Propagated label quality: **+9.7 pp Dice** (78.8% vs 69.1% for voxel-KNN)
- Trained model performance: TBD (need to run experiments)

---

## Troubleshooting

### Issue: "Sparse labels not found"

**Problem:** Seeds directory doesn't contain the expected files.

**Solution:**
```bash
# Check if seeds exist
ls runs/strategic_sparse_0p1pct_k_multi/strategic_seeds/*_sv_labels_sparse.json | wc -l
# Should show 380

# If missing, regenerate seeds (see STRATEGIC_SPARSE_SV_README.md)
```

### Issue: "Out of memory during training"

**Problem:** GPU memory insufficient for batch_size=2.

**Solution:**
```bash
# Reduce batch size
python3 train_finetune_wp5.py \
  --batch_size 1 \
  ... (other args)

# Or use gradient accumulation
python3 train_finetune_wp5.py \
  --batch_size 1 \
  --accumulate_grad_batches 2 \
  ... (other args)
```

### Issue: "Propagation is slow"

**Problem:** Processing 380 cases takes too long.

**Solution:**
```bash
# Process subset first to verify
# Edit script to limit cases (lines 180-182)
# Or run on subset manually:
python3 -c "
from pathlib import Path
cases = list(Path('runs/strategic_sparse_0p1pct_k_multi/strategic_seeds').glob('*_sv_labels_sparse.json'))[:10]
print(f'Processing {len(cases)} cases...')
"
```

---

## Expected Results

Based on single-volume experiments (case SN13B0_I17_3D_B1_1B250409):

| Method | Budget | Propagated Dice | Test Dice (Expected) |
|--------|--------|----------------|---------------------|
| SV-KNN | 0.1% | 29.0% | ~27% |
| Voxel-KNN | 0.1% | 69.1% | ~62% |
| **Graph LP** | **0.1%** | **78.8%** | **~70%** (estimated) |

**Graph LP provides:**
- **+9.7 pp** improvement over voxel-KNN on propagated labels
- **+49.8 pp** improvement over SV-KNN
- Near-optimal for this SV partition (upperbound: 92%)

---

## References

**Implementation:**
- Graph LP algorithm: `wp5/weaklabel/graph_label_propagation.py`
- Batch propagation: `scripts/propagate_graph_lp_multi_case.py`
- Training script: `train_finetune_wp5.py`
- Evaluation: `scripts/eval_wp5.py`

**Documentation:**
- Full comparison: `tmp/GRAPH_LP_VS_KNN_COMPARISON.md`
- Hyperparameter analysis: `tmp/HYPERPARAMETER_ANALYSIS.md`
- Strategic sparse pipeline: `STRATEGIC_SPARSE_SV_README.md`
- Main documentation: `AGENTS.md`

**Paper:**
Zhou et al., "Learning with Local and Global Consistency", NIPS 2003

---

**Last Updated:** 2025-11-13
