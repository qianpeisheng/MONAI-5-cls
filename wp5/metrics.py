from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric


def compute_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    heavy: bool = True,
    hd_percentile: float = 100.0,
) -> Dict[int, Dict[str, float]]:
    """Compute per-class Dice, IoU, and optionally HD/ASD for classes 0..4; ignore class 6.

    Policy: when a class is absent in both prediction and GT for a class/case, score 1.0 (both-empty=1.0)
    """
    B = pred.shape[0]
    ignore_mask = (gt != 6)
    classes = [0, 1, 2, 3, 4]
    if heavy:
        hd_metric = HausdorffDistanceMetric(percentile=float(hd_percentile), reduction="none")
        asd_metric = SurfaceDistanceMetric(symmetric=True, reduction="none")

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

        both_empty = (psum + gsum) == 0
        valid = ~both_empty
        dice = np.full(pred.shape[0], np.nan, dtype=np.float32)
        iou = np.full(pred.shape[0], np.nan, dtype=np.float32)
        dice[valid] = (2.0 * inter[valid]) / (psum[valid] + gsum[valid] + 1e-8)
        iou_valid = uni[valid] > 0
        iou_vals = np.zeros_like(inter[valid], dtype=np.float32)
        iou_vals[iou_valid] = inter[valid][iou_valid] / (uni[valid][iou_valid] + 1e-8)
        iou[valid] = iou_vals
        dice[both_empty] = 1.0
        iou[both_empty] = 1.0

        if heavy:
            hd_vals = np.full(B, np.nan, dtype=np.float32)
            asd_vals = np.full(B, np.nan, dtype=np.float32)
            for b in range(B):
                if psum[b] == 0 and gsum[b] == 0:
                    continue
                if psum[b] == 0 or gsum[b] == 0:
                    continue
                pt = torch.from_numpy(pm[b:b+1][None, ...].astype(np.float32))
                gt_t = torch.from_numpy(gm[b:b+1][None, ...].astype(np.float32))
                try:
                    hd_vals[b] = float(np.array(hd_metric(pt, gt_t)).reshape(-1)[0])
                except Exception:
                    hd_vals[b] = np.nan
                try:
                    asd_vals[b] = float(np.array(asd_metric(pt, gt_t)).reshape(-1)[0])
                except Exception:
                    asd_vals[b] = np.nan
            hd_mean = float(np.nanmean(hd_vals)) if np.any(~np.isnan(hd_vals)) else None
            asd_mean = float(np.nanmean(asd_vals)) if np.any(~np.isnan(asd_vals)) else None
        else:
            hd_mean = None
            asd_mean = None

        out[cls] = {
            "dice": float(np.nanmean(dice)) if np.any(~np.isnan(dice)) else 0.0,
            "iou": float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0,
            "hd": hd_mean,
            "asd": asd_mean,
        }

    return out

