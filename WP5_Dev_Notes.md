WP5 MONAI Pipeline Changes

- Date: 2025-10-06
- Context: Fix low metrics vs baseline by aligning normalization with wp5-code and correcting a metric bug.

Changes

- Per-sample clip+z-score normalization
  - Implemented `ClipZScoreNormalizeD` (custom MONAI MapTransform) to clip image intensities to [p1, p99] and apply z-score per sample.
  - Applied in both train and validation transforms after `Orientationd` and before padding/cropping.
  - File: `train_finetune_wp5.py` (class `ClipZScoreNormalizeD`, used in `get_transforms`).
  - Rationale: WP5 images have heavy-tailed, heterogeneous intensities; this matches `wp5-code`’s `clip_zscore` approach and the policy in `WP5_Segmentation_Data_Guide.md`.

- Metric computation bug fix
  - Fixed per-class masking in `compute_metrics`: for classes 1..4, use `pred == cls` (previously used `pred > 0` which collapsed all non-background classes into one and severely penalized Dice/IoU).
  - File: `train_finetune_wp5.py` (`compute_metrics`).

- Normalization flag and grid script
  - Added CLI flag `--norm` with choices: `clip_zscore` (default), `fixed_wp5` (legacy `[-3,8.5]→[0,1]`), or `none`.
  - Transforms now honor `--norm` in both training and inference paths.
  - New script `run_finetune_grid_v2.sh` to redo the 6 experiments with a chosen normalization; outputs to `runs/grid_<norm>/...`.

- Few-shot (configurable) in main trainer
  - New CLI: `--fewshot_mode {few_samples,few_slices,few_points}`, `--fewshot_ratio {0.1,0.01}`.
  - `few_samples` is the default; this corresponds to using a subset of volumes via `--subset_ratio` (full supervision on selected volumes).
  - Few-slices (3D masked): `--fs_axis_mode {z,y,x,multi}`, optional `--fs_k_slices`, `--fs_class_aware` (reserved; current uses FG counts).
  - Few-points (3D sparse masked): `--fp_dilate_radius`, `--fp_balance {proportional,uniform}`, `--fp_max_seeds`, `--fp_bg_frac`, `--fp_seed_strategy {random,boundary}` (boundary reserved).
  - Training: builds `sup_mask` per batch after crop; CE uses ignore_index=255 on unsupervised voxels and label==6; Dice masks `(label!=6) & (sup_mask==1)`.
  - Inference unchanged.

- Point-based few-shot enhancements (configurable)
  - Intensity augs (training only): `--aug_intensity` with `--aug_prob`, `--aug_noise_std`, `--aug_shift`, `--aug_scale`.
  - FG-biased cropping: `--fg_crop_prob`, `--fg_crop_margin` to sample crops centered around label foreground.
  - Pseudo-labels (self-training): `--pl_enable`, `--pl_threshold`, `--pl_weight`, `--pl_warmup_epochs` to add CE on high-confidence predictions for unlabeled voxels.
  - Defaults keep all enhancements off (no change to existing behavior).

Usage examples

- Full supervision or few-samples (default):
  - `python3 train_finetune_wp5.py --mode train --init scratch --subset_ratio 1.0 --output_dir runs/full ...`
  - Few-samples 10%: `--subset_ratio 0.1` (no change to `--fewshot_mode`, which defaults to `few_samples`).
- Few-slices (10%, Z axis):
  - `python3 train_finetune_wp5.py --mode train --init scratch --fewshot_mode few_slices --fewshot_ratio 0.1 --fs_axis_mode z --output_dir runs/fewslice_z_10pct ...`
- Few-slices (1%, multi-axis):
  - `python3 train_finetune_wp5.py --mode train --init scratch --fewshot_mode few_slices --fewshot_ratio 0.01 --fs_axis_mode multi --output_dir runs/fewslice_multi_1pct ...`
  - Alternate static selection (simpler heuristic): first pick slices where classes 0..4 all appear (ignore 6), sample randomly to 1% budget across all axes, then train with fixed sup masks:
    - `python3 scripts/select_informative_slices.py --train_list datalist_train.json --out_dir runs/selected_slices_allcls_1pct_$(date +%Y%m%d-%H%M%S) --percent 0.01 --selector all_classes_random --save_sup_masks`
    - Then launch training using the produced `sup_masks/` directory with `--fewshot_mode few_slices --fewshot_static --sup_masks_dir <that_dir>`.
- Few-points (10%, dilate=1):
  - `python3 train_finetune_wp5.py --mode train --init scratch --fewshot_mode few_points --fewshot_ratio 0.1 --fp_dilate_radius 1 --output_dir runs/fewpoints_10pct ...`

- Few-points (10%) + pseudo-labels:
  - `python3 train_finetune_wp5.py --mode train --init scratch --fewshot_mode few_points --fewshot_ratio 0.1 --fp_dilate_radius 1 --pl_enable --pl_threshold 0.9 --pl_weight 0.2 --pl_warmup_epochs 5 --output_dir runs/fewpoints_10pct_pl ...`

- Few-points (1%) + FG-biased crop:
  - `python3 train_finetune_wp5.py --mode train --init scratch --fewshot_mode few_points --fewshot_ratio 0.01 --fp_dilate_radius 1 --fg_crop_prob 0.7 --output_dir runs/fewpoints_01pct_fgcrop ...`

Virtualenv and quick sanity tests

- Activate the project virtual environment:
  - `source venv/bin/activate`
  - Confirm: `python --version` and `python -c "import torch, monai; print(torch.__version__)"`
- Quick unit check for slice supervision (runs on CPU):
  - `python - << 'PY'` then paste:
    ```python
    import torch
    from train_finetune_wp5 import _select_slices_mask_per_sample, build_slice_supervision_mask
    B,C,X,Y,Z=1,1,8,10,12
    lbl=torch.zeros((B,C,X,Y,Z),dtype=torch.long)
    lbl[:,:,2,:,:]=1; lbl[:,:,:,4,:]=2; lbl[:,:,:,:,7]=3
    for ax in (0,1,2):
        m=_select_slices_mask_per_sample(lbl,axis=ax,k=1)
        print('axis',ax,'shape',m.shape,'cov',float(m.float().mean()))
    m_all=build_slice_supervision_mask(lbl,roi=(X,Y,Z),axis_mode='multi',ratio=0.2,k_override=None)
    print('multi coverage',float(m_all.float().mean()))
    ```
    end with `PY` on its own line.
- Debugging device asserts (optional):
  - Prefix a single run with `CUDA_LAUNCH_BLOCKING=1` to catch the exact failing op.

Notes

- Consistency/entropy/pseudo-labeling are not enabled; can be added later behind flags if needed.

Notes

- Label policy remains unchanged and aligned with the guide: ignore voxels where `label == 6` for CE and Dice, and compute metrics over classes 0..4.
- Kept MONAI idioms (dict transforms, sliding-window inference, `(112,112,80)` patches) and avoided dataset-level assumptions.

Expected Impact

- Metrics should improve significantly due to correct per-class evaluation and more robust normalization.
- Training stability over variable-intensity scans should increase, especially across products/serials with different ranges.

References

- Source ideas for normalization: `/data3/wp5/wp5-code/dataloaders/wp5.py` (`get_normalization_transform('clip_zscore', per_sample=True)`).
- Policy reference: `WP5_Segmentation_Data_Guide.md` (robust per-sample normalization and ignore class 6).

---

Additions and fixes — 2025-10-09

- Configurable evaluation of both-empty cases (standardized)
  - CLI flag `--empty_pair_policy {exclude,count_as_one}` controls how per-class metrics aggregate cases where both prediction and GT are empty.
  - Default is now `count_as_one` (standard reporting): when a class is absent in both prediction and GT for a case, score 1.0 for Dice/IoU.
  - Use `exclude` only for exploratory analyses to avoid rare-class inflation.
  - Applied consistently in training-time eval and inference-time eval.
  - Files: `train_finetune_wp5.py` (`compute_metrics`, `evaluate`, CLI arg parsing).

- Orientation-safe mask precompute in static few-points
  - Internal loader for labels when precomputing static seed/supervision masks now uses MONAI `LoadImaged+Orientationd(axcodes='RAS')` for consistency with train-time transforms.
  - File: `train_finetune_wp5.py` (`_load_label_volume`).

- Old-semantics evaluator script
  - New helper: `scripts/eval_wp5_old_semantics.py` to re-evaluate existing predictions with the old both-empty=1 convention.
  - Usage:
    - `python scripts/eval_wp5_old_semantics.py --pred_dir <run>/preds --datalist datalist_test.json --out <run>/metrics/summary_old_semantics.json`.

- Streamlit viewer: few-shot comparison
  - Added a second predictions directory input to compare fully supervised vs few-shot outputs side-by-side (+GT) in 2D and 3D.
  - Default dirs:
    - Fully supervised: `/home/peisheng/MONAI/runs/grid_clip_zscore/pretrained_subset_100_eval/preds`.
    - Few-shot 0.001%: `/home/peisheng/MONAI/runs/fixed_points_scratch50/ratio_0.00001_infer_20251009-154947/preds`.
  - Files: `scripts/vis_wp5_streamlit.py` (adds `--pred_dir2`, triplet 2D view, dual 3D view, per-class Dice per pred).

- Extreme few-points run scripts
  - New: `scripts/run_fixed_points_ratio_1e6.sh` to run ratio=1e-6 (0.000001) few-points from scratch on a chosen GPU.
  - Updated: `scripts/run_fixed_points_bundle_extremes_50ep.sh` now runs from scratch (BasicUNet), launches two ratios concurrently, and writes to stable output dirs.

- Training loop robustness
  - Fixed an indentation bug around per-epoch evaluation in `train_finetune_wp5.py`.

Notes

- These changes keep MONAI idioms and WP5 label policy (ignore class 6) intact.
- For few-points `uniform_all` sampling with ignore-6, consider `--fp_uniform_exclude6` to avoid allocating seeds to class 6.

---

External Precompute Workflow (1% points)

- Precompute static masks (uniform per-scan 1% of voxels, no dilation):
  - `. /home/peisheng/MONAI/venv/bin/activate && python3 scripts/precompute_sup_masks.py --mode few_points_global --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json --subset_ratio 1.0 --ratio 0.01 --dilate_radius 0 --balance proportional --seed 42 --fp_sample_mode uniform_all --out_dir runs/sup_masks_1pct_uniform_all`
- Train with precomputed masks:
  - `. /home/peisheng/MONAI/venv/bin/activate && CUDA_VISIBLE_DEVICES=1 python3 -u train_finetune_wp5.py --mode train --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json --output_dir runs/fewpoints_01pct_static_from_dir --epochs 20 --batch_size 2 --num_workers 4 --init scratch --net basicunet --subset_ratio 1.0 --seed 42 --fewshot_mode few_points --fewshot_ratio 0.01 --fewshot_static --sup_masks_dir runs/sup_masks_1pct_uniform_all --pseudo_weight 0.0 --fg_crop_prob 0.0 --coverage_mode seeds --norm clip_zscore --roi_x 112 --roi_y 112 --roi_z 80 --log_to_file`

Evaluation (single evaluator; official semantics)

- Use `scripts/eval_wp5.py` for all evaluations; in-script trainer eval has been removed.
- Official policy: evaluate classes 0..4 (ignore label 6) and when both prediction and GT are empty for a class in a volume count the score as 1.0.
- Example command:
  - `. /home/peisheng/MONAI/venv/bin/activate && python3 scripts/eval_wp5.py --ckpt <run>/last.ckpt --datalist datalist_test.json --output_dir <run>_eval --no_timestamp --heavy --hd_percentile 95 --log_to_file`

Known-good environment (WP5)

- Python 3.9.5, torch 2.8.0+cu128, MONAI 1.5.1
- Confirm with: `. /home/peisheng/MONAI/venv/bin/activate && python -c "import torch, monai; print(torch.__version__, monai.__version__)"`
