from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


def _maybe_squeeze_channel(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return arr[..., 0]
    return arr


def transpose_for_imshow(arr: np.ndarray) -> np.ndarray:
    """
    Safe transpose helper for matplotlib imshow.

    - For 2D arrays: returns `arr.T`
    - For RGB/RGBA arrays: returns `arr.transpose(1, 0, 2)`

    Using `.T` directly on RGB/RGBA produces invalid shapes like `(C, W, H)`.
    """
    a = np.asarray(arr)
    if a.ndim == 2:
        return a.T
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        return a.transpose(1, 0, 2)
    raise ValueError(f"Unsupported array shape for imshow transpose: {a.shape}")


def load_volume_ras(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Load a 3D volume from .nii/.nii.gz (canonicalized to closest RAS) or .npy.

    Returns:
      (volume, spacing_xyz) where spacing is in (x,y,z) order.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix == ".npy":
        arr = np.load(str(p))
        arr = np.asarray(arr)
        arr = _maybe_squeeze_channel(arr)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array from {path}, got shape={arr.shape}")
        return arr, (1.0, 1.0, 1.0)

    if p.suffix == ".nii" or p.name.endswith(".nii.gz"):
        import nibabel as nib  # local import: optional dependency for non-NIfTI use

        img = nib.load(str(p))
        try:
            img = nib.as_closest_canonical(img)
        except Exception:
            pass
        arr = np.asarray(img.get_fdata())
        arr = _maybe_squeeze_channel(arr)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D NIfTI from {path}, got shape={arr.shape}")
        A = getattr(img, "affine", None)
        if A is None:
            spacing_xyz = (1.0, 1.0, 1.0)
        else:
            spacing_xyz = tuple(float(np.linalg.norm(A[:3, i])) for i in range(3))
        return arr, spacing_xyz  # type: ignore[return-value]

    raise ValueError(f"Unsupported volume extension: {path}")


def counts_per_class(vol: np.ndarray, classes: Iterable[int]) -> Dict[int, int]:
    v = np.asarray(vol)
    return {int(c): int((v == c).sum()) for c in classes}


def valid_label_mask(
    gt: np.ndarray,
    *,
    ignore_label: Optional[int] = 6,
    unlabeled_value: Optional[int] = -1,
) -> np.ndarray:
    m = np.ones_like(gt, dtype=bool)
    if ignore_label is not None:
        m &= gt != int(ignore_label)
    if unlabeled_value is not None:
        m &= gt != int(unlabeled_value)
    return m


def error_mask(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    ignore_label: Optional[int] = 6,
    unlabeled_value: Optional[int] = -1,
) -> np.ndarray:
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred/gt shape mismatch: pred={pred.shape} gt={gt.shape}")
    m = valid_label_mask(gt, ignore_label=ignore_label, unlabeled_value=unlabeled_value)
    return m & (pred != gt)


def boundary_band_mask(
    labels: np.ndarray,
    *,
    radius: float,
    fg_classes: Sequence[int] = (1, 2, 3, 4),
    ignore_label: Optional[int] = 6,
) -> np.ndarray:
    """
    Compute a band around the FG boundary (FG vs non-FG).

    The band is defined as voxels with distance-to-boundary <= radius (in voxels).
    Boundary voxels are computed from the union foreground mask.
    """
    if radius < 0:
        raise ValueError("radius must be >= 0")

    lbl = np.asarray(labels)
    fg = np.isin(lbl, list(fg_classes))
    if ignore_label is not None:
        fg &= lbl != int(ignore_label)

    if not fg.any():
        return np.zeros_like(lbl, dtype=bool)

    from scipy.ndimage import binary_erosion, distance_transform_edt

    er = binary_erosion(fg, iterations=1)
    boundary = fg ^ er
    if not boundary.any():
        return np.zeros_like(lbl, dtype=bool)

    dist = distance_transform_edt(~boundary)
    return dist <= float(radius)


def outer_bg_distance_mask(
    labels: np.ndarray,
    *,
    min_distance: float,
    fg_classes: Sequence[int] = (1, 2, 3, 4),
    background_value: int = 0,
    background_only: bool = True,
) -> np.ndarray:
    """
    Diagnostic mask for "outer BG": background voxels far from FG.

    - If there is no FG, returns all background voxels as outer BG.
    - If there is no background, returns all-False.
    """
    if min_distance < 0:
        raise ValueError("min_distance must be >= 0")

    lbl = np.asarray(labels)
    bg = lbl == int(background_value)
    if not bg.any():
        return np.zeros_like(lbl, dtype=bool)

    fg = np.isin(lbl, list(fg_classes))
    if not fg.any():
        return bg if background_only else np.ones_like(lbl, dtype=bool)

    from scipy.ndimage import distance_transform_edt

    dist_to_fg = distance_transform_edt(~fg)
    mask = dist_to_fg >= float(min_distance)
    return (mask & bg) if background_only else mask
