#!/usr/bin/env python3
"""
Re-evaluate a trained run with a 'present-only' metric policy.

Why: The default evaluation in train_finetune_wp5.py treats cases where a
given class is absent in both prediction and ground truth as perfect (Dice=1).
This can inflate scores for rare classes. This script recomputes metrics by
averaging per-class only over volumes where the class is present in GT.

Usage:
  python scripts/verify_present_only_eval.py \
    --run_dir runs/fixed_points_bundle50/ratio_0.0025 \
    --ckpt last.ckpt \
    --max_cases 30 \
    --device cuda

Notes:
  - Requires the same environment as training (PyTorch + MONAI).
  - Uses the dataset split/config from train_finetune_wp5 defaults unless overridden.
  - Saves a JSON summary next to the run under metrics/present_only_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference

import train_finetune_wp5 as tf


def compute_metrics_present_only(pred: torch.Tensor, gt: torch.Tensor, heavy: bool = False) -> Dict[int, Dict[str, float]]:
    """Compute per-class metrics, averaging only over volumes where class is present in GT.

    pred, gt: (B,1,X,Y,Z)
    Returns dict[class] = {dice, iou, n_cases}
    """
    B = pred.shape[0]
    ignore_mask = (gt != 6)
    classes = [0, 1, 2, 3, 4]
    out: Dict[int, Dict[str, float]] = {}
    for cls in classes:
        pred_mask = (pred == cls)
        gt_mask = (gt == cls)
        pm = (pred_mask & ignore_mask).squeeze(1).cpu().numpy().astype(np.uint8)
        gm = (gt_mask & ignore_mask).squeeze(1).cpu().numpy().astype(np.uint8)
        inter = (pm & gm).sum(axis=(1, 2, 3))
        psum = pm.sum(axis=(1, 2, 3))
        gsum = gm.sum(axis=(1, 2, 3))
        uni = (pm | gm).sum(axis=(1, 2, 3))

        # Only include samples where GT has the class
        mask_present = gsum > 0
        n = int(mask_present.sum())
        if n == 0:
            out[cls] = {"dice": float("nan"), "iou": float("nan"), "n_cases": 0}
            continue

        inter = inter[mask_present]
        psum = psum[mask_present]
        gsum = gsum[mask_present]
        uni = uni[mask_present]

        dice = np.where(psum + gsum > 0, (2.0 * inter) / (psum + gsum + 1e-8), 0.0)
        iou = np.where(uni > 0, inter / (uni + 1e-8), 0.0)
        out[cls] = {"dice": float(np.mean(dice)), "iou": float(np.mean(iou)), "n_cases": n}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to a training run directory (contains ckpts, metrics)")
    ap.add_argument("--ckpt", default="last.ckpt", help="Checkpoint filename in run_dir (last.ckpt or best.ckpt)")
    ap.add_argument("--data_root", default="/data3/wp5/wp5-code/dataloaders/wp5-dataset", help="Dataset root")
    ap.add_argument("--split_cfg", default="/data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json", help="Split config JSON")
    ap.add_argument("--roi", default="112,112,80", help="Sliding window ROI, e.g., 112,112,80")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_cases", type=int, default=0, help="Limit number of test cases (0=all)")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "cpu" else "cpu")
    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / args.ckpt
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Data
    train_list, test_list = tf.build_datalists(Path(args.data_root) / "data", Path(args.split_cfg))
    _, val_t = tf.get_transforms(roi=tuple(map(int, args.roi.split(','))), norm="clip_zscore")
    ds = Dataset(test_list, transform=val_t)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model (match bundle wrapper used in training script autodetecting from run)
    # Try to infer whether a bundle was used by presence of 'configs' under the known bundle path in train script.
    # Here we re-build from bundle config if the run was created by scripts/run_fixed_points_bundle_50ep.sh
    # which passes --bundle_dir pretrained_models/spleen_ct_segmentation/spleen_ct_segmentation
    bundle_dir = Path("pretrained_models/spleen_ct_segmentation/spleen_ct_segmentation")
    if bundle_dir.exists():
        net = tf.build_model_from_bundle(bundle_dir, out_channels=5).to(device)
    else:
        net = tf.build_model("basicunet").to(device)

    sd = torch.load(ckpt_path, map_location=device)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    net.load_state_dict(sd, strict=False)
    net.eval()

    roi = tuple(map(int, args.roi.split(',')))
    classes = [0, 1, 2, 3, 4]
    sums = {c: {"dice": 0.0, "iou": 0.0, "n": 0} for c in classes}

    with torch.no_grad():
        for i, batch in enumerate(dl):
            if args.max_cases and i >= args.max_cases:
                break
            img = batch["image"].to(device)
            gt = batch["label"].to(device)
            logits = sliding_window_inference(img, roi_size=roi, sw_batch_size=1, predictor=net, sw_device=device, device=device)
            pred = torch.argmax(logits, dim=1, keepdim=True)
            per_class = compute_metrics_present_only(pred.cpu(), gt.cpu())
            for c in classes:
                if per_class[c]["n_cases"] > 0:
                    sums[c]["dice"] += per_class[c]["dice"]
                    sums[c]["iou"] += per_class[c]["iou"]
                    sums[c]["n"] += 1

    # Finalize
    summary: Dict[str, Dict[str, float]] = {}
    for c in classes:
        n = max(1, sums[c]["n"])  # avoid div zero
        summary[str(c)] = {
            "dice": sums[c]["dice"] / n,
            "iou": sums[c]["iou"] / n,
            "n_cases": n,
        }
    avg = {
        "dice": float(np.nanmean([summary[str(c)]["dice"] for c in classes])),
        "iou": float(np.nanmean([summary[str(c)]["iou"] for c in classes])),
    }

    out = {"per_class": summary, "average": avg}
    out_path = run_dir / "metrics" / "present_only_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("Saved:", out_path)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

