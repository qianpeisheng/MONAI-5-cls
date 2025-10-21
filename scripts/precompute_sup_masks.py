#!/usr/bin/env python3
"""
Precompute static supervision masks for WP5 few-shot experiments (step 1/2).

Generates per-case files:
  <out_dir>/<id>_seedmask.npy     (1,X,Y,Z) bool
  <out_dir>/<id>_supmask.npy      (1,X,Y,Z) bool
  <out_dir>/<id>_pseudolabel.npy  (optional, 1,X,Y,Z) int64
  <out_dir>/<id>_supmask_stats.json  summary per case

Two modes:
  - few_points_global: sample seeds from GT using a global budget (ratio) and BG fraction.
  - selected_points:   use preselected points from a directory; optional dense pseudo labels.

Example (global 1% with radius=1):
  venv/bin/python scripts/precompute_sup_masks.py \
    --mode few_points_global \
    --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
    --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json \
    --subset_ratio 1.0 --ratio 0.01 --dilate_radius 1 --seed_bg_frac 0.25 \
    --balance proportional --seed 42 \
    --out_dir runs/sup_masks_1pct_global_d1

Example (use 1% selected points + dense pseudo, radius=1):
  venv/bin/python scripts/precompute_sup_masks.py \
    --mode selected_points \
    --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
    --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json \
    --subset_ratio 1.0 --dilate_radius 1 \
    --selected_points_dir /data3/wp5/wp5-code/dataloaders/wp5-dataset/pseudo_labels_1pct_train_intermediate \
    --pseudo_label_dir /data3/wp5/wp5-code/dataloaders/wp5-dataset/pseudo_labels_1pct_train \
    --out_dir runs/sup_masks_1pct_points_pl_d1

Then train with:
  venv/bin/python train_finetune_wp5.py ... --fewshot_mode few_points --fewshot_static --sup_masks_dir <out_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

# Robust import from repo root in case script is executed from scripts/
try:
    from train_finetune_wp5 import (
        build_datalists,
        subset_datalist,
        precompute_static_global_seed_masks,
        precompute_sup_masks_from_selected_points,
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path as _Path
    repo_root = _Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from train_finetune_wp5 import (
        build_datalists,
        subset_datalist,
        precompute_static_global_seed_masks,
        precompute_sup_masks_from_selected_points,
    )


def summarize(out_dir: Path) -> dict:
    stats = list(out_dir.glob("*_supmask_stats.json"))
    fracs = []
    for p in stats:
        try:
            d = json.loads(p.read_text())
            fracs.append(float(d.get("sup_fraction", 0.0)))
        except Exception:
            pass
    if fracs:
        arr = np.asarray(fracs, dtype=float)
        return {
            "files": len(fracs),
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return {"files": 0, "mean": 0.0, "min": 0.0, "max": 0.0}


def main():
    ap = argparse.ArgumentParser(description="Precompute static supervision masks for WP5")
    ap.add_argument("--mode", choices=["few_points_global", "selected_points"], required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_cfg", required=True)
    ap.add_argument("--subset_ratio", type=float, default=1.0)
    ap.add_argument("--out_dir", required=True, help="Directory to write mask files into")
    # few_points_global
    ap.add_argument("--ratio", type=float, default=0.01)
    ap.add_argument("--seed_bg_frac", type=float, default=0.25)
    ap.add_argument("--dilate_radius", type=int, default=1)
    ap.add_argument("--balance", choices=["proportional", "uniform"], default="proportional")
    ap.add_argument("--no_overlap", action="store_true")
    ap.add_argument("--dilation_shape", choices=["auto", "cube", "cross"], default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp_sample_mode", choices=["stratified", "uniform_all"], default="stratified")
    ap.add_argument("--fp_uniform_exclude6", action="store_true")
    # selected_points
    ap.add_argument("--selected_points_dir", default="")
    ap.add_argument("--pseudo_label_dir", default="")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    split_cfg = Path(args.split_cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_list, _ = build_datalists(data_root / "data", split_cfg)
    train_list = subset_datalist(train_list, args.subset_ratio, args.seed)

    if args.mode == "selected_points":
        if not args.selected_points_dir:
            raise SystemExit("--selected_points_dir is required for mode=selected_points")
        pl_dir = Path(args.pseudo_label_dir) if args.pseudo_label_dir else None
        precompute_sup_masks_from_selected_points(
            train_list=train_list,
            selected_points_dir=Path(args.selected_points_dir),
            pseudo_label_dir=pl_dir,
            out_dir=out_dir,
            dilate_radius=args.dilate_radius,
        )
        cfg = {
            "mode": args.mode,
            "selected_points_dir": str(args.selected_points_dir),
            "pseudo_label_dir": str(args.pseudo_label_dir),
            "dilate_radius": int(args.dilate_radius),
            "subset_ratio": float(args.subset_ratio),
        }
    else:
        if args.ratio <= 0.0:
            raise SystemExit("--ratio must be > 0 for mode=few_points_global")
        precompute_static_global_seed_masks(
            train_list=train_list,
            masks_dir=out_dir,
            ratio=float(args.ratio),
            seed_bg_frac=float(args.seed_bg_frac),
            dilate_radius=int(args.dilate_radius),
            balance=str(args.balance),
            no_overlap=bool(args.no_overlap),
            dilation_shape=str(args.dilation_shape),
            seed=int(args.seed),
            sample_mode=str(args.fp_sample_mode),
            uniform_exclude6=bool(args.fp_uniform_exclude6),
        )
        cfg = {
            "mode": args.mode,
            "ratio": float(args.ratio),
            "seed_bg_frac": float(args.seed_bg_frac),
            "dilate_radius": int(args.dilate_radius),
            "balance": str(args.balance),
            "no_overlap": bool(args.no_overlap),
            "dilation_shape": str(args.dilation_shape),
            "seed": int(args.seed),
            "fp_sample_mode": str(args.fp_sample_mode),
            "fp_uniform_exclude6": bool(args.fp_uniform_exclude6),
            "subset_ratio": float(args.subset_ratio),
        }

    # Save a manifest for traceability
    manifest = {
        "config": cfg,
        "data_root": str(data_root),
        "split_cfg": str(split_cfg),
        "out_dir": str(out_dir),
        "summary": summarize(out_dir),
    }
    (out_dir / "sup_masks_config.json").write_text(json.dumps(manifest, indent=2))
    print("Completed mask precompute. Summary:", json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
