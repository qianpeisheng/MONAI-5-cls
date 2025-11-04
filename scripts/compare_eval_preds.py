#!/usr/bin/env python3
"""
Compare two sets of saved predictions against WP5 ground-truth labels and report Dice/IoU.

- Assumes predictions are saved by MONAI SaveImage as NIfTI files with suffix "_pred.nii.gz".
- Maps cases via the image filename base from the datalist ("image" field), e.g. <base>_pred.nii.gz.
- Computes per-case metrics over classes 0..4 and ignores voxels with GT == 6.
  Reports per-class and average Dice/IoU for each prediction directory.
- Also checks whether corresponding prediction files are byte-equal in pred1 and pred2 (shape/value).

Usage example:
  python3 scripts/compare_eval_preds.py \
    --pred1_dir runs/grid_clip_zscore/pretrained_subset_100_eval/preds \
    --pred2_dir runs/grid_clip_zscore/pretrained_subset_100_eval_backup/preds \
    --datalist datalist_test.json \
    --out runs/grid_clip_zscore/pred_compare_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib


def strip_niibase(name: str) -> str:
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    return name


def load_pred(path: Path) -> np.ndarray:
    if path.suffix in ('.npy', '.npz'):
        arr = np.load(path)
        if isinstance(arr, np.lib.npyio.NpzFile):
            # take first array
            key = list(arr.keys())[0]
            arr = arr[key]
        arr = np.asarray(arr)
    else:
        img = nib.load(str(path))
        arr = img.get_fdata(dtype=np.float32)
    arr = arr.astype(np.int64)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def load_label(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    arr = img.get_fdata(dtype=np.float32).astype(np.int64)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def dice_iou_per_class(pred: np.ndarray, gt: np.ndarray, classes=(0, 1, 2, 3, 4)) -> Dict[int, Tuple[float, float]]:
    # Ignore voxels with label == 6 entirely
    mask = (gt != 6)
    out: Dict[int, Tuple[float, float]] = {}
    eps = 1e-8
    for c in classes:
        pm = (pred == c) & mask
        gm = (gt == c) & mask
        inter = float((pm & gm).sum())
        psum = float(pm.sum())
        gsum = float(gm.sum())
        union = float((pm | gm).sum())
        dice = (2.0 * inter) / (psum + gsum + eps) if (psum + gsum) > 0 else 1.0
        iou = inter / (union + eps) if union > 0 else 1.0
        out[c] = (dice, iou)
    return out


def main():
    ap = argparse.ArgumentParser("Compare two prediction sets vs GT (Dice/IoU)")
    ap.add_argument('--pred1_dir', required=True)
    ap.add_argument('--pred2_dir', required=True)
    ap.add_argument('--datalist', default='datalist_test.json')
    ap.add_argument('--out', default='pred_compare_report.json')
    args = ap.parse_args()

    pred1_dir = Path(args.pred1_dir)
    pred2_dir = Path(args.pred2_dir)
    test_list: List[Dict] = json.loads(Path(args.datalist).read_text())

    cls_ids = [0, 1, 2, 3, 4]
    sums1 = {c: {'dice': 0.0, 'iou': 0.0, 'n': 0} for c in cls_ids}
    sums2 = {c: {'dice': 0.0, 'iou': 0.0, 'n': 0} for c in cls_ids}

    missing1 = []
    missing2 = []
    diffs = []
    compared = 0

    for rec in test_list:
        imgp = Path(rec['image'])
        lblp = Path(rec['label'])
        rid = rec.get('id') or strip_niibase(imgp.name)
        img_base = strip_niibase(imgp.name)
        def without_image_suffix(b: str) -> str:
            # Drop a trailing _image or _label if present
            for suf in ('_image', '_label'):
                if b.endswith(suf):
                    return b[: -len(suf)]
            return b
        img_base_wo = without_image_suffix(img_base)

        # Candidate names we actually see on disk in your runs: <id>_pred.npy
        # Also try image base variants and NIfTI just in case.
        candidates = [
            f'{rid}_pred.npy', f'{rid}_pred.nii.gz', f'{rid}_pred.nii',
            f'{img_base}_pred.npy', f'{img_base}_pred.nii.gz', f'{img_base}_pred.nii',
            f'{img_base_wo}_pred.npy', f'{img_base_wo}_pred.nii.gz', f'{img_base_wo}_pred.nii',
        ]
        p1 = next((pred1_dir / n for n in candidates if (pred1_dir / n).exists()), None)
        p2 = next((pred2_dir / n for n in candidates if (pred2_dir / n).exists()), None)
        if p1 is None:
            missing1.append(rid)
        if p2 is None:
            missing2.append(rid)
        if p1 is None or p2 is None:
            continue

        # Load arrays
        try:
            pred1 = load_pred(p1)
            pred2 = load_pred(p2)
            gt = load_label(lblp)
        except Exception as e:
            diffs.append({'base': base, 'error': str(e)})
            continue

        base = rid
        if pred1.shape != gt.shape:
            diffs.append({'base': base, 'mismatch': f'pred1 shape {pred1.shape} != gt {gt.shape}'})
            continue
        if pred2.shape != gt.shape:
            diffs.append({'base': base, 'mismatch': f'pred2 shape {pred2.shape} != gt {gt.shape}'})
            continue

        # Compare predictions equality (value-wise)
        same = np.array_equal(pred1, pred2)
        if not same:
            frac_diff = float((pred1 != pred2).mean())
            diffs.append({'base': base, 'same': False, 'diff_fraction': frac_diff})

        # Metrics
        per1 = dice_iou_per_class(pred1, gt, cls_ids)
        per2 = dice_iou_per_class(pred2, gt, cls_ids)
        for c in cls_ids:
            d1, i1 = per1[c]
            d2, i2 = per2[c]
            sums1[c]['dice'] += d1; sums1[c]['iou'] += i1; sums1[c]['n'] += 1
            sums2[c]['dice'] += d2; sums2[c]['iou'] += i2; sums2[c]['n'] += 1
        compared += 1

    def finalize(sums):
        out = {}
        for c in cls_ids:
            n = max(1, sums[c]['n'])
            out[str(c)] = {
                'dice': sums[c]['dice']/n,
                'iou': sums[c]['iou']/n,
            }
        avg = {
            'dice': float(np.mean([out[str(c)]['dice'] for c in cls_ids])),
            'iou': float(np.mean([out[str(c)]['iou'] for c in cls_ids])),
        }
        return out, avg

    per1, avg1 = finalize(sums1)
    per2, avg2 = finalize(sums2)

    report = {
        'pred1_dir': str(pred1_dir),
        'pred2_dir': str(pred2_dir),
        'cases_compared': compared,
        'missing_pred1': missing1,
        'missing_pred2': missing2,
        'pred_differences': diffs[:20],  # cap for brevity
        'preds_identical_all_cases': (len(diffs) == 0 and len(missing1) == 0 and len(missing2) == 0),
        'metrics_pred1': {'per_class': per1, 'average': avg1},
        'metrics_pred2': {'per_class': per2, 'average': avg2},
    }

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
