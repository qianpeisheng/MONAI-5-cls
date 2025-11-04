from __future__ import annotations

from typing import Dict, Iterable, Tuple, List, Optional

import numpy as np
from scipy.interpolate import Rbf

from .sampler import PointsForClass
from .geom_utils import bbox_from_mask


def _eval_rbf_grid(
    rbf: Rbf,
    shape: Tuple[int, int, int],
    region: Tuple[slice, slice, slice] | None = None,
    chunk_z: int = 32,
) -> np.ndarray:
    X, Y, Z = shape
    if region is None:
        sx, sy, sz = slice(0, X), slice(0, Y), slice(0, Z)
    else:
        sx, sy, sz = region
    out = np.zeros((sx.stop - sx.start, sy.stop - sy.start, sz.stop - sz.start), dtype=np.float32)
    # Evaluate in z-chunks to reduce peak memory/time
    for z0 in range(sz.start, sz.stop, chunk_z):
        z1 = min(z0 + chunk_z, sz.stop)
        zz = np.arange(z0, z1)
        yy = np.arange(sy.start, sy.stop)
        xx = np.arange(sx.start, sx.stop)
        Xg, Yg, Zg = np.meshgrid(xx, yy, zz, indexing="ij")
        vals = rbf(Xg.ravel(), Yg.ravel(), Zg.ravel()).reshape(Xg.shape)
        out[:, :, (z0 - sz.start) : (z1 - sz.start)] = vals.astype(np.float32)
    return out


def reconstruct_from_points(
    shape: Tuple[int, int, int],
    points: Dict[int, PointsForClass],
    rbf_kernel: str = "multiquadric",
    rbf_eps: float = 2.0,
    smooth: float = 1e-3,
    bbox_margin: int = 4,
    chunk_z: int = 32,
    constraints: Optional[Dict] = None,
) -> np.ndarray:
    """Reconstruct multi-class mask from sparse points using class-wise RBF SDFs.

    - Fits one RBF per class over integer voxel coordinates with signed distances.
    - Evaluates per-class SDF near each class bounding box (with margin) to save time.
    - Final label per voxel is argmax over positive SDFs; otherwise background (0).
    """
    X, Y, Z = shape
    sdf_maps: Dict[int, Tuple[Tuple[slice, slice, slice], np.ndarray]] = {}
    # First determine per-class support to compute bbox
    for c, pf in points.items():
        if pf.coords.shape[0] == 0:
            continue
        # RBF on voxel coordinates; to mitigate scale, standardize coordinates
        coords = pf.coords.astype(np.float32)
        sdf = pf.sdf.astype(np.float32)
        # Simple standardization
        mu = coords.mean(axis=0, keepdims=True)
        sig = coords.std(axis=0, keepdims=True) + 1e-6
        coords_n = (coords - mu) / sig
        rbf = Rbf(coords_n[:, 0], coords_n[:, 1], coords_n[:, 2], sdf, function=rbf_kernel, epsilon=rbf_eps, smooth=smooth)

        # Region of interest: bounding box around observed inside points
        # Use points with sdf >= 0 as proxy for interior/near-boundary
        inside_pts = coords[sdf >= 0]
        if inside_pts.shape[0] == 0:
            inside_pts = coords
        mask_pts = np.zeros(shape, dtype=bool)
        ip = np.clip(inside_pts.astype(int), 0, np.array(shape) - 1)
        mask_pts[ip[:, 0], ip[:, 1], ip[:, 2]] = True
        region = bbox_from_mask(mask_pts, margin=bbox_margin, shape=shape)

        # Evaluate RBF on normalized coordinates in region
        sx, sy, sz = region
        out_local = np.zeros((sx.stop - sx.start, sy.stop - sy.start, sz.stop - sz.start), dtype=np.float32)
        for z0 in range(sz.start, sz.stop, chunk_z):
            z1 = min(z0 + chunk_z, sz.stop)
            zz = np.arange(z0, z1)
            yy = np.arange(sy.start, sy.stop)
            xx = np.arange(sx.start, sx.stop)
            Xg, Yg, Zg = np.meshgrid(xx, yy, zz, indexing="ij")
            # normalize
            Xn = (Xg - mu[0, 0]) / sig[0, 0]
            Yn = (Yg - mu[0, 1]) / sig[0, 1]
            Zn = (Zg - mu[0, 2]) / sig[0, 2]
            vals = rbf(Xn.ravel(), Yn.ravel(), Zn.ravel()).reshape(Xg.shape)
            out_local[:, :, (z0 - sz.start) : (z1 - sz.start)] = vals.astype(np.float32)

        sdf_maps[c] = (region, out_local)

    # Build positive masks for fusion
    pos_masks: Dict[int, np.ndarray] = {}
    for c, pf in points.items():
        # If groups provided, reconstruct per-group to avoid merging separate instances (e.g., class 3 voids)
        if pf.groups is not None and pf.coords.shape[0] > 0 and np.unique(pf.groups).size > 1:
            cls_mask = np.zeros(shape, dtype=bool)
            for gid in np.unique(pf.groups):
                sel = pf.groups == gid
                if not np.any(sel):
                    continue
                coords = pf.coords[sel].astype(np.float32)
                sdf = pf.sdf[sel].astype(np.float32)
                mu = coords.mean(axis=0, keepdims=True)
                sig = coords.std(axis=0, keepdims=True) + 1e-6
                coords_n = (coords - mu) / sig
                rbf = Rbf(coords_n[:, 0], coords_n[:, 1], coords_n[:, 2], sdf, function=rbf_kernel, epsilon=rbf_eps, smooth=smooth)
                # region around this group's positive points
                inside_pts = coords[sdf >= 0]
                if inside_pts.shape[0] == 0:
                    inside_pts = coords
                mask_pts = np.zeros(shape, dtype=bool)
                ip = np.clip(inside_pts.astype(int), 0, np.array(shape) - 1)
                mask_pts[ip[:, 0], ip[:, 1], ip[:, 2]] = True
                region = bbox_from_mask(mask_pts, margin=bbox_margin, shape=shape)
                sx, sy, sz = region
                out_local = np.zeros((sx.stop - sx.start, sy.stop - sy.start, sz.stop - sz.start), dtype=np.float32)
                for z0 in range(sz.start, sz.stop, chunk_z):
                    z1 = min(z0 + chunk_z, sz.stop)
                    zz = np.arange(z0, z1)
                    yy = np.arange(sy.start, sy.stop)
                    xx = np.arange(sx.start, sx.stop)
                    Xg, Yg, Zg = np.meshgrid(xx, yy, zz, indexing="ij")
                    Xn = (Xg - mu[0, 0]) / sig[0, 0]
                    Yn = (Yg - mu[0, 1]) / sig[0, 1]
                    Zn = (Zg - mu[0, 2]) / sig[0, 2]
                    vals = rbf(Xn.ravel(), Yn.ravel(), Zn.ravel()).reshape(Xg.shape)
                    out_local[:, :, (z0 - sz.start) : (z1 - sz.start)] = vals.astype(np.float32)
                cur = np.zeros(shape, dtype=np.float32)
                cur[sx, sy, sz] = out_local
                cls_mask |= (cur > 0)
            pos_masks[c] = cls_mask
        else:
            region, sdf_loc = sdf_maps.get(c, (None, None))
            if region is None:
                continue
            sx, sy, sz = region
            cur = np.zeros(shape, dtype=np.float32)
            cur[sx, sy, sz] = sdf_loc
            pos_masks[c] = cur > 0

    if not constraints:
        # Default: argmax of positive SDFs
        label = np.zeros(shape, dtype=np.int16)
        best_val = np.zeros(shape, dtype=np.float32) - 1e9
        for c, (region, sdf_loc) in sdf_maps.items():
            sx, sy, sz = region
            cur = np.zeros(shape, dtype=np.float32)
            cur[sx, sy, sz] = sdf_loc
            mask_pos = cur > 0
            better = mask_pos & (cur > best_val)
            label[better] = int(c)
            best_val[better] = cur[better]
        return label

    # Fusion with constraints
    precedence: List[int] = list(constraints.get("precedence", []))
    disjoint = set(tuple(sorted(pair)) for pair in constraints.get("disjoint_pairs", []))
    inclusive = [tuple(pair) for pair in constraints.get("inclusive", [])]  # child,parent

    # Make a working copy of masks to enforce inclusive/disjoint
    masks = {c: m.copy() for c, m in pos_masks.items()}

    # Enforce inclusions:
    # 1) child must be subset of parent
    # 2) subtract child from parent to avoid double assignment
    for child, parent in inclusive:
        if child in masks and parent in masks:
            masks[child] = masks[child] & masks[parent]
            masks[parent] = masks[parent] & (~masks[child])

    # Compose by precedence (earlier wins)
    label = np.zeros(shape, dtype=np.int16)
    filled = np.zeros(shape, dtype=bool)
    for c in precedence:
        m = masks.get(c)
        if m is None:
            continue
        place = m & (~filled)
        label[place] = int(c)
        filled |= place
        # Enforce disjointness by clearing disjoint partners' masks in filled area
        for d_pair in list(disjoint):
            if c in d_pair:
                other = d_pair[0] if d_pair[1] == c else d_pair[1]
                if other in masks:
                    masks[other] = masks[other] & (~place)

    return label
