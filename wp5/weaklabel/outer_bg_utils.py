#!/usr/bin/env python3
"""
Utilities for adaptive outer-background partitioning in WP5-style volumes.

These helpers are intentionally self-contained (NumPy + SciPy only) so they can
be imported from tests or scripts without pulling in MONAI or torch.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


def choose_outer_bg_distance(
    gt_labels: np.ndarray,
    target_outer_bg_frac: float = 0.7,
    min_distance: float = 1.0,
    max_distance: float = 32.0,
) -> Tuple[float, float]:
    """
    Choose a per-volume outer-background distance threshold based on GT.

    The goal is to mark approximately `target_outer_bg_frac` of *background voxels*
    (GT == 0) as "outer BG", i.e., sufficiently far from any foreground (classes 1-4).

    Args:
        gt_labels: (X, Y, Z) integer array with WP5 semantics.
        target_outer_bg_frac: Desired fraction of GT-background voxels that should be
            treated as outer background (0..1). Values <=0 disable outer BG.
        min_distance: Lower clamp for the chosen threshold (in voxels).
        max_distance: Upper clamp for the chosen threshold (in voxels).

    Returns:
        outer_bg_distance: Chosen distance threshold (float).
        actual_outer_bg_frac: Actual fraction of GT-background voxels whose distance
            to the nearest foreground voxel exceeds the chosen threshold.
    """
    target_outer_bg_frac = float(target_outer_bg_frac)
    if target_outer_bg_frac <= 0.0:
        return 0.0, 0.0

    gt = np.asarray(gt_labels)
    if gt.ndim != 3:
        raise ValueError(f"gt_labels must be 3D, got shape={gt.shape}")

    # Foreground: classes 1-4, ignore 6; background: class 0.
    fg_mask = (gt > 0) & (gt != 6)
    bg_mask = gt == 0

    num_bg = int(bg_mask.sum())
    if num_bg == 0:
        # Degenerate: no background voxels; fall back to minimal threshold.
        return float(min_distance), 0.0

    if not np.any(fg_mask):
        # Degenerate: no foreground; treat all background as outer BG.
        return float(max_distance), 1.0

    # Distance (in voxels) from each voxel to nearest foreground voxel.
    # This matches the convention used in scripts/sample_strategic_sv_seeds.py.
    dist_to_fg = distance_transform_edt(~fg_mask)
    bg_distances = dist_to_fg[bg_mask]

    # We want fraction(bg_distances > thr) ~= target_outer_bg_frac.
    # Let q = 1 - target; pick thr as q-quantile and clamp to [min_distance, max_distance].
    q = float(np.clip(1.0 - target_outer_bg_frac, 0.0, 1.0))
    thr_raw = float(np.quantile(bg_distances, q))
    thr = float(np.clip(thr_raw, min_distance, max_distance))

    actual_frac = float((bg_distances > thr).sum()) / float(num_bg)
    return thr, actual_frac


__all__ = ["choose_outer_bg_distance"]

