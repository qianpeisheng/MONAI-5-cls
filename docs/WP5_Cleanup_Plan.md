# WP5 Cleanup Plan (Immediate TODOs)

This plan captures the concrete actions to remove evaluation ambiguity, enforce logging via args, and fix model init confusion before the next development phase.

Immediate TODOs

- Remove broken eval mode in trainer
  - train_finetune_wp5.py: restrict `--mode` to `train` only; remove `infer(args)` implementation and update docstrings.
  - All references to in-script eval/infer are removed; use `scripts/eval_wp5.py` instead.

- Standardize evaluation semantics (both-empty → 1.0)
  - train_finetune_wp5.py: `compute_metrics` only implements “both-empty=1.0”; remove policy flag in `evaluate` and call sites.
  - scripts/eval_wp5.py: remove `--empty_pair_policy`; always use the same semantics via `tfw.compute_metrics`.
  - scripts/eval_wp5_old_semantics.py, scripts/verify_present_only_eval.py: mark deprecated and exit immediately with a clear message.

- Deduplicate helpers and make them canonical
  - Ensure single definitions for `_select_slices_mask_per_sample`, `build_slice_supervision_mask`, `build_points_supervision_mask`, `compute_metrics`, `evaluate`.

- Add args-driven logging (no shell tee required)
  - train_finetune_wp5.py: add `--log_to_file` (default true) and `--log_file_name` (default `train.log`). Implement a stdout/stderr tee to `<output_dir>/train.log`. Save `args.json`.
  - scripts/eval_wp5.py: add `--log_to_file` (default true) and `--log_file_name` (default `eval.log`). Implement tee and save metrics as before.

- Model init policy clarity (prefer larger BasicUNet)
  - train_finetune_wp5.py: default `--net basicunet` and keep it recommended. If `--init scratch`, ignore any `--pretrained_ckpt`. If `--init pretrained`, load weights without reinitializing. Log parameter count and config.

Documentation touch-ups (in this pass)

- Update docstrings in trainer and evaluator to reference a single evaluator and the official semantics.
- Note deprecation in legacy evaluator scripts with an early exit.

Follow-ups (next pass)

- Factor helpers into `wp5/` module (metrics, masks, transforms, model).
- Add CPU-only smoke tests for transforms, masks, and metrics.
- Pin known-good environment (torch 2.8.0+cu128, MONAI 1.5.1) in a short section in WP5_Dev_Notes.md.

