Few‑Shot Policy (WP5)

Overview
- Budget definition: fewshot_ratio refers to the fraction of true labels used as seed points, measured globally across the entire training set (all voxels, including background).
- Static masks: Seeds and their dilated supervision regions are precomputed once per volume and reused throughout training. No per‑iteration resampling.
- Background: Background (class 0) receives a smaller share of the seed budget (configurable via --seed_bg_frac; default 0.10). Remaining seeds are allocated to classes 1–4 according to --fp_balance (proportional or uniform).
- Dilation: The dilated supervision region (radius --fp_dilate_radius) serves as context. We propagate seed labels into dilation (no overlap between seed neighborhoods) and apply a small loss weight.

Quick Start
- 1% seeds, cube dilation (3×3×3), static global budget, save masks, report seed coverage:
  - `CUDA_VISIBLE_DEVICES=0 python3 -u train_finetune_wp5.py --mode train \
     --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
     --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json \
     --output_dir runs/fewshot_static_01 --epochs 50 --batch_size 2 --num_workers 4 \
     --net basicunet --norm clip_zscore --init scratch \
     --fewshot_mode few_points --fewshot_ratio 0.01 --fewshot_static \
     --fp_dilate_radius 1 --dilation_shape cube --fp_balance proportional --seed_bg_frac 0.10 \
     --pseudo_weight 0.3 --coverage_mode seeds --fg_crop_prob 0.0 --save_sup_masks` 
- 10% seeds, cross dilation (Manhattan, size≈7 for r=1). Non‑overlap feasible; save masks, report seed coverage:
  - `CUDA_VISIBLE_DEVICES=1 python3 -u train_finetune_wp5.py --mode train \
     --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
     --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json \
     --output_dir runs/fewshot_static_10 --epochs 50 --batch_size 2 --num_workers 4 \
     --net basicunet --norm clip_zscore --init scratch \
     --fewshot_mode few_points --fewshot_ratio 0.10 --fewshot_static \
     --fp_dilate_radius 1 --dilation_shape cross --fp_no_overlap \
     --fp_balance proportional --seed_bg_frac 0.10 \
     --pseudo_weight 0.3 --coverage_mode seeds --fg_crop_prob 0.0 --save_sup_masks` 

Verifying budgets
- After precompute (automatic at the start of training), run:
  - 1%: `python3 scripts/verify_sup_masks.py --run_dir runs/fewshot_static_01 --expect_ratio 0.01 --dilate_radius 1 --shape cube`
  - 10%: `python3 scripts/verify_sup_masks.py --run_dir runs/fewshot_static_10 --expect_ratio 0.10 --dilate_radius 1 --shape cross`
- Seed fraction is checked against the expected budget; printed supervised (dilated) fraction is context only.

Training losses
- Seed supervision: CE + Dice on seed voxels only (strict ground truth, no budget leakage).
- Propagated pseudo labels: CE on dilated voxels using labels propagated from seeds; weighted by --pseudo_weight (default 0.3).
- Optional model PL: Optionally activate confidence‑based pseudo labels on voxels outside the dilated region (--pl_enable), same as before.

Coverage & Verification
- Seed fraction and dilated fraction per volume are saved to <run_dir>/sup_masks/*_supmask_stats.json.
- Verify script: scripts/verify_sup_masks.py reports dataset‑level mean/std/min/max for seeds and dilated coverage; checks against expected ratio.
- Training logs can report either seed coverage or dilated coverage via --coverage_mode {seeds,sup}.

Implementation notes
- Precomputation: Global seed budget is enforced across the train set. Seeds are spread to avoid overlap after dilation (--fp_no_overlap), so dilated pseudo labels are unambiguous.
- Transforms: Precomputed masks and propagated pseudo labels are loaded and carried through pad/flip/crop to stay aligned with the label.
- Dynamic few‑points resampling is disabled for clarity; use static masks.

Dilation shapes and overlap handling
- Shape selection via `--dilation_shape {auto,cube,cross}`:
  - auto: cross for ratio ≥ 0.1, else cube.
  - cube (Chebyshev): neighborhood size (2r+1)^3.
  - cross (Manhattan): neighborhood size ~ 1 + 6r.
- Feasibility: Maximum non‑overlap seed density is the inverse of neighborhood size. For r=1, cube≈3.7%, cross≈14.3%.
- Overlap policy when `--fp_no_overlap` is off:
  - cross shape uses nearest‑seed (Manhattan) tie‑break with deterministic FG‑first priority.
  - cube shape uses classwise dilation with FG‑first overwrite; in typical 1% settings overlaps are rare.

Recommended flags
- Global 1% few‑points (no label‑biased crops):
  --fewshot_mode few_points --fewshot_ratio 0.01 --fewshot_static \
  --fp_dilate_radius 1 --fp_no_overlap --seed_bg_frac 0.10 \
  --pseudo_weight 0.3 --fg_crop_prob 0.0 --save_sup_masks
