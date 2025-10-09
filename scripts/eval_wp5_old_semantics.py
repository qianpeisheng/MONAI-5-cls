#!/usr/bin/env python3
"""
Evaluate WP5 predictions using the "old" semantics:
- Ignore label 6 voxels in ground truth for all metrics
- For per-class per-case metrics: if both prediction and GT are empty for the class, score = 1.0
- Macro-average per class across cases; then average over classes 0..4

Usage:
  python scripts/eval_wp5_old_semantics.py \
    --pred_dir /path/to/preds \
    --datalist datalist_test.json \
    --out /path/to/metrics/summary_old_semantics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import nibabel as nib  # type: ignore
except Exception:
    nib = None


def load_json(path: Path):
    return [] if not path.exists() else json.loads(path.read_text())


def load_label(path: Path) -> np.ndarray:
    if nib is None:
        raise RuntimeError("nibabel is required for loading NIfTI labels")
    img = nib.load(str(path))
    try:
        img = nib.as_closest_canonical(img)
    except Exception:
        pass
    arr = img.get_fdata().astype(np.int64)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def load_pred_any(path: Path) -> np.ndarray:
    s = path.suffix.lower()
    if s == ".npy" or path.name.endswith(".npy"):
        arr = np.load(str(path))
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        return arr.astype(np.int64)
    else:
        if nib is None:
            raise RuntimeError("nibabel is required for loading NIfTI predictions")
        img = nib.load(str(path))
        try:
            img = nib.as_closest_canonical(img)
        except Exception:
            pass
        arr = img.get_fdata().astype(np.int64)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        return arr


def center_pad_or_crop(vol: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    out = vol
    # Crop
    for axis in range(3):
        diff = out.shape[axis] - target_shape[axis]
        if diff > 0:
            start = diff // 2
            end = start + target_shape[axis]
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(start, end)
            out = out[tuple(sl)]
    # Pad
    pad_width = []
    for axis in range(3):
        diff = target_shape[axis] - out.shape[axis]
        if diff > 0:
            l = diff // 2
            r = diff - l
            pad_width.append((l, r))
        else:
            pad_width.append((0, 0))
    if any(p[0] > 0 or p[1] > 0 for p in pad_width):
        out = np.pad(out, pad_width, mode="constant", constant_values=0)
    return out


def find_pred_path(pred_dir: Path, image_path: Path, case_id: str) -> Optional[Path]:
    # Try image basename with _pred suffix first
    base = image_path.name
    base_no_ext = base[:-7] if base.endswith(".nii.gz") else (base[:-4] if base.endswith(".nii") else base)
    for ext in (".nii.gz", ".npy", ".nii"):
        cand = pred_dir / f"{base_no_ext}_pred{ext}"
        if cand.exists():
            return cand
    # Fallback: anything containing case id and 'pred'
    for pattern in ("*pred*.nii*", "*pred*.npy"):
        for p in sorted(pred_dir.glob(pattern)):
            if case_id and case_id in p.name:
                return p
    # Last fallback: first pred-like file
    any_pred = sorted(list(pred_dir.glob("*pred*.nii*")) + list(pred_dir.glob("*pred*.npy")))
    return any_pred[0] if any_pred else None


def dice_iou_old_semantics(pred: np.ndarray, gt: np.ndarray, cls: int, ignore_label: int = 6) -> Tuple[float, float]:
    mask = gt != ignore_label
    p = (pred == cls) & mask
    g = (gt == cls) & mask
    inter = float(np.logical_and(p, g).sum())
    psum = float(p.sum())
    gsum = float(g.sum())
    union = float(np.logical_or(p, g).sum())
    # Both empty -> 1.0
    if psum + gsum == 0:
        return 1.0, 1.0
    dice = (2.0 * inter) / (psum + gsum + 1e-8)
    iou = inter / (union + 1e-8) if union > 0 else 0.0
    return float(dice), float(iou)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", type=str, required=True)
    ap.add_argument("--datalist", type=str, default="datalist_test.json")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    datalist = load_json(Path(args.datalist))
    classes = [0, 1, 2, 3, 4]

    sums: Dict[int, Dict[str, float]] = {c: {"dice": 0.0, "iou": 0.0, "n": 0} for c in classes}

    for rec in datalist:
        img_p = Path(rec.get("image"))
        lbl_p = Path(rec.get("label"))
        cid = rec.get("id")
        if not (img_p and img_p.exists() and lbl_p and lbl_p.exists()):
            continue
        pred_p = find_pred_path(pred_dir, img_p, cid)
        if pred_p is None or not pred_p.exists():
            continue
        gt = load_label(lbl_p)
        pred = load_pred_any(pred_p)
        if pred.shape != gt.shape:
            pred = center_pad_or_crop(pred, gt.shape)
        for c in classes:
            d, j = dice_iou_old_semantics(pred, gt, c, ignore_label=6)
            sums[c]["dice"] += d
            sums[c]["iou"] += j
            sums[c]["n"] += 1

    per_class = {}
    for c in classes:
        n = max(1, int(sums[c]["n"]))
        per_class[str(c)] = {
            "dice": sums[c]["dice"] / n,
            "iou": sums[c]["iou"] / n,
            "hd": None,
            "asd": None,
        }
    avg = {
        "dice": float(np.mean([per_class[str(c)]["dice"] for c in classes])),
        "iou": float(np.mean([per_class[str(c)]["iou"] for c in classes])),
        "hd": None,
        "asd": None,
    }

    out = {"per_class": per_class, "average": avg}
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(out, indent=2))
    print(f"Wrote old-semantics metrics to: {out_p}")


if __name__ == "__main__":
    main()

