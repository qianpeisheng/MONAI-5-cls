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
