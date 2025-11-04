from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries


def class_mask(label: np.ndarray, cls: int) -> np.ndarray:
    return (label == cls)


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    # find_boundaries returns inner boundaries; ensure boolean
    b = find_boundaries(mask.astype(bool), connectivity=1, mode="inner")
    return b.astype(bool)


def interior_mask(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    if thickness <= 0:
        return mask.copy()
    # 3D cross-structure; use ndimage to support iterations
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)
    eroded = ndi.binary_erosion(mask, structure=structure, iterations=int(thickness))
    return eroded


def shell_inside(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    if thickness <= 0:
        return np.zeros_like(mask, dtype=bool)
    ero = interior_mask(mask, thickness)
    return (mask.astype(bool) & (~ero))


def shell_outside(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    if thickness <= 0:
        return np.zeros_like(mask, dtype=bool)
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)
    dil = ndi.binary_dilation(mask.astype(bool), structure=structure, iterations=int(thickness))
    return (dil & (~mask.astype(bool)))


def distance_inside(mask: np.ndarray) -> np.ndarray:
    # distance to boundary inside the object
    return ndi.distance_transform_edt(mask.astype(bool))


def distance_outside(mask: np.ndarray) -> np.ndarray:
    # distance outside the object (distance to mask from complement)
    return ndi.distance_transform_edt(~mask.astype(bool))


def estimate_surface_weight(mask: np.ndarray) -> float:
    # Use boundary voxel count as a proxy for surface area
    b = boundary_mask(mask)
    return float(b.sum())


def bbox_from_mask(mask: np.ndarray, margin: int = 0, shape: Tuple[int, int, int] | None = None) -> Tuple[slice, slice, slice]:
    idx = np.argwhere(mask)
    if idx.size == 0:
        return slice(0, 0), slice(0, 0), slice(0, 0)
    mins = idx.min(axis=0)
    maxs = idx.max(axis=0) + 1
    if shape is None:
        shape = mask.shape
    mins = np.maximum(mins - margin, 0)
    maxs = np.minimum(maxs + margin, shape)
    sx = slice(int(mins[0]), int(maxs[0]))
    sy = slice(int(mins[1]), int(maxs[1]))
    sz = slice(int(mins[2]), int(maxs[2]))
    return sx, sy, sz


def random_sample_coords(coords: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    if coords.shape[0] <= k:
        return coords
    idx = rng.choice(coords.shape[0], size=k, replace=False)
    return coords[idx]


def farthest_point_sampling(coords: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
    # Greedy FPS in voxel space; O(k * N). For large N, fallback to random.
    n = coords.shape[0]
    if k <= 0:
        return coords[:0]
    if n <= k or n == 0:
        return coords
    # If too large, random sample first to cap N
    cap = 100000
    if n > cap:
        coords = random_sample_coords(coords, cap, rng)
        n = coords.shape[0]
    sel_idx = np.zeros(k, dtype=int)
    sel_idx[0] = rng.randint(0, n)
    d2 = np.sum((coords - coords[sel_idx[0]]) ** 2, axis=1)
    for i in range(1, k):
        j = int(np.argmax(d2))
        sel_idx[i] = j
        d2 = np.minimum(d2, np.sum((coords - coords[j]) ** 2, axis=1))
    return coords[sel_idx]
