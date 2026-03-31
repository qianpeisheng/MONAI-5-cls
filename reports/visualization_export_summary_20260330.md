# Visualization Export Summary (2026-03-30)

## Objective

Generated publication-ready visualizations comparing multiple conditions on the 174-case test set (new WP5 dataset, 613 train / 174 test).

## Conditions

1. **Ground Truth** — dense expert labels
2. **Fully supervised** (0.903 Dice) — trained on 100% GT labels
3. **Sparse supervision** (0.847 Dice) — trained on 0.1% sparse annotations only (fewpoints baseline)
4. **Ours** (0.863 Dice) — trained on 0.1% sparse annotations + Graph-LP ensemble pseudo-labels with agreement-count weighting

## Prediction Generation

Predictions were generated via `scripts/eval_wp5.py --save_preds` using the test split from `dataset_config.json`.

| Run | Avg Dice | Checkpoint | Predictions |
|-----|----------|-----------|-------------|
| Fully supervised | 0.9038 | `runs/wp5_full_supervised_20251210-164804/best.ckpt` | `runs/wp5_full_supervised_20251210-164804/eval/preds/` (174 `.nii.gz`) |
| Sparse supervision | 0.8471 | `runs/wp5_fewpoints_0_1pct_global_20251210-191641/best.ckpt` | `runs/wp5_fewpoints_0_1pct_global_20251210-191641/eval/preds/` (174 `.nii.gz`) |
| Ours (COMQ ensemble) | 0.8629 | `/data3/MONAI_experiments/sweep_graphlp_conf_ens_lossw_sweep40/COMQ_dec_bs_g4_0p20_g3_0p15_g2_0p00_20260130-005424/best.ckpt` | `/data3/MONAI_experiments/sweep_graphlp_conf_ens_lossw_sweep40/COMQ_dec_bs_g4_0p20_g3_0p15_g2_0p00_20260130-005424/eval/preds/` (174 `.nii.gz`) |

Metrics saved alongside at `eval/metrics/summary.json` in each run directory.

## Exported Figures

### V2: 2D Slice Quints (z-axis) — 5-panel

- **Script:** `scripts/export_wp5_slice_quints.py` (multiprocessing-enabled)
- **Output:** `runs/exports_slice_quints_v2/` — 174 case folders
- **Layout:** 5-panel per slice: Data | Ground Truth | Fully supervised | Sparse supervision | Ours
- **Structure:** `<case_id>/z/<case_id>_z<NNN>.png`
- **Settings:** `--dpi 300 --alpha 0.4 --axes z --workers 14`

### V2: 3D Class Surface Plots — 4-panel

- **Script:** `scripts/export_wp5_3d_class_plots_v2.py` (multiprocessing-enabled)
- **Output:** `runs/exports_3d_class_plots_v2/` — 696 PNGs (4 classes x 174 cases)
- **Layout:** 4-panel per class: GT (green) | Fully supervised (red) | Sparse supervision (blue) | Ours (orange)
- **Structure:** `<case_id>/class_<N>.png`
- **Settings:** `--dpi 300 --step 2 --opacity 0.55 --classes 1 2 3 4 --workers 14`

### V1 (legacy): 2D Slice Quads (z-axis) — 4-panel

- **Script:** `scripts/export_wp5_slice_quads.py`
- **Output:** `runs/exports_slice_quads_all/` — 13,793 PNGs across 174 case folders
- **Layout:** 4-panel per slice: Data (grayscale) | Ground Truth (overlay) | Fully Supervised 0.903 (overlay) | COMQ 0.1% 0.863 (overlay)
- **Structure:** `<case_id>/z/<case_id>_z<NNN>.png`
- **Settings:** `--dpi 300 --alpha 0.4 --axes z`

### V1 (legacy): 3D Class Surface Plots — 3-panel

- **Script:** `scripts/export_wp5_3d_class_plots.py`
- **Output:** `runs/exports_3d_class_plots_all/` — 696 PNGs (4 classes x 174 cases)
- **Layout:** 3-panel per class: GT (green) | Fully Sup 0.903 (red) | COMQ 0.1% 0.863 (orange)
- **Structure:** `<case_id>/class_<N>.png`
- **Settings:** `--dpi 300 --step 2 --opacity 0.55 --classes 1 2 3 4`

### Test Exports (first 3 cases)

- `runs/exports_slice_quads_new/` — 2D slice test run (3 cases, z-axis)
- `runs/exports_3d_class_plots_new/` — 3D surface test run (3 cases)

## Key File Locations

| Resource | Path |
|----------|------|
| Test datalist (174 cases) | `datalist_test_new.json` |
| GT labels | `/data3/wp5_4_Dec_data/3ddl-dataset/data/labels/` |
| GT images | `/data3/wp5_4_Dec_data/3ddl-dataset/data/images/` |
| Split config | `/data3/wp5_4_Dec_data/3ddl-dataset/data/dataset_config.json` |
| 2D export script (v2) | `scripts/export_wp5_slice_quints.py` |
| 3D export script (v2) | `scripts/export_wp5_3d_class_plots_v2.py` |
| 2D export script (v1) | `scripts/export_wp5_slice_quads.py` |
| 3D export script (v1) | `scripts/export_wp5_3d_class_plots.py` |
| Eval script | `scripts/eval_wp5.py` |
| Streamlit viewer | `scripts/vis_wp5_streamlit.py` |

## V2 Changes (2026-03-31)

- Added **Sparse supervision** condition (fewpoints 0.1% baseline, no pseudo-labels)
- Renamed "COMQ 0.1%" to **Ours**
- Removed numerical Dice values from all panel titles
- 2D figures: 4-panel → 5-panel; 3D figures: 3-panel → 4-panel
- Scripts parallelized with multiprocessing (`--workers` flag)
- V1 exports preserved (not overwritten)

## Not Yet Done

- x and y axis slices (run `export_wp5_slice_quints.py --axes x y` if needed)
- Cherry-picked cases for paper figures (all 174 exported; manual selection still needed)
