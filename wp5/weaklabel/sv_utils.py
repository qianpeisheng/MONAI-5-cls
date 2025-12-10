import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    import nibabel as nib  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    nib = None

try:
    import SimpleITK as sitk  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    sitk = None


def _clip_zscore(x: np.ndarray, pmin: float = 1.0, pmax: float = 99.0) -> np.ndarray:
    """Robust per-volume normalization: clip to [pmin,pmax] percentiles, then z-score.

    Returns float32 array of the same shape.
    """
    x = x.astype(np.float32, copy=False)
    lo, hi = np.percentile(x, [pmin, pmax]).astype(np.float32)
    if hi <= lo:  # degenerate
        return np.zeros_like(x, dtype=np.float32)
    x = np.clip(x, lo, hi)
    m = float(x.mean())
    s = float(x.std())
    if s == 0.0:
        return np.zeros_like(x, dtype=np.float32)
    return (x - m) / s


def load_image(path: str, normalize: bool = True) -> np.ndarray:
    """Load 3D image volume as (Z, Y, X) float32.

    Supports .nii/.nii.gz via nibabel (preferred), SimpleITK as fallback, and .npy.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        # Accept (Z,Y,X) or (1,Z,Y,X) and squeeze leading singleton.
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D or 1x3D npy, got shape {arr.shape} for {path}")
        return _clip_zscore(arr) if normalize else arr.astype(np.float32)

    # NIfTI or other formats
    if nib is not None and (ext in {".nii", ".gz"} or path.endswith(".nii.gz")):
        img = nib.load(path)
        arr = np.asanyarray(img.dataobj)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D or 1x3D NIfTI, got shape {arr.shape} for {path}")
        return _clip_zscore(arr) if normalize else arr.astype(np.float32)

    if sitk is not None:
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)  # (Z,Y,X)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D image, got shape {arr.shape} for {path}")
        return _clip_zscore(arr) if normalize else arr.astype(np.float32)

    raise RuntimeError(
        f"Unsupported image format or missing loaders for {path}. Install nibabel or SimpleITK, or provide .npy."
    )


def load_seed_mask(path: str) -> np.ndarray:
    """Load seed mask as boolean array with shape (Z, Y, X) or (1,Z,Y,X).

    This baseline expects foreground-only seeds (True where seeded). If a (1,Z,Y,X)
    shape is provided, the channel dimension is removed.
    """
    arr = np.load(path)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D or 1x3D seeds, got shape {arr.shape} for {path}")
    # Accept integer masks (0/1) or bool
    if arr.dtype != bool:
        arr = arr.astype(np.int32) != 0
    return arr


def load_pseudolabel(path: str) -> np.ndarray:
    """Load dense per-voxel labels as int array (Z,Y,X) or (1,Z,Y,X)."""
    arr = np.load(path)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D or 1x3D labels, got shape {arr.shape} for {path}")
    return arr.astype(np.int32, copy=False)


def seed_labels_from_mask_and_pseudo(
    seed_mask: np.ndarray,
    pseudo_labels: np.ndarray,
    ignore_classes: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Construct sparse seed labels from a binary seed mask and a dense label volume.

    - seed voxels take their class from `pseudo_labels`.
    - non-seed voxels get -1 (unlabeled sentinel).
    - labels in `ignore_classes` (e.g., {6}) are treated as unlabeled (-1).
    Returns int32 array of shape (Z,Y,X) with -1 for unlabeled voxels.
    """
    if seed_mask.shape != pseudo_labels.shape:
        raise ValueError(
            f"Shape mismatch: seeds {seed_mask.shape} vs pseudo {pseudo_labels.shape}"
        )
    out = np.full(seed_mask.shape, -1, dtype=np.int32)
    if ignore_classes is None:
        ignore_classes = set()
    else:
        ignore_classes = set(ignore_classes)
    seeded_vals = pseudo_labels[seed_mask]
    if ignore_classes:
        keep = ~np.isin(seeded_vals, list(ignore_classes))
        idx = np.flatnonzero(seed_mask)
        idx = idx[keep]
        out.flat[idx] = seeded_vals[keep].astype(np.int32)
    else:
        out[seed_mask] = seeded_vals.astype(np.int32)
    return out


def majority_fill(
    seed_labels: np.ndarray,
    sv_ids: np.ndarray,
    *,
    unlabeled_values: Sequence[int] = (-1, 6),
    tie_policy: str = "skip",
    output_unlabeled_value: int = 6,
) -> Tuple[np.ndarray, int]:
    """Propagate majority class within each supervoxel.

    Args:
      seed_labels: int array (Z,Y,X) with -1 (or listed) where unlabeled.
      sv_ids: int array (Z,Y,X) of same shape; contiguous SV ids starting at 0.
      unlabeled_values: label values considered unlabeled/ignored in seeds (e.g., -1 and 6).
      tie_policy: one of {"skip","min","max","prefer_foreground"}.
      output_unlabeled_value: value to use for voxels in SVs with no labeled seeds or ties (default 6).

    Returns:
      (dense_labels, n_filled_svs)
    """
    if seed_labels.shape != sv_ids.shape:
        raise ValueError("seed_labels and sv_ids must have the same shape")

    flat_labels = seed_labels.ravel()
    flat_svid = sv_ids.ravel().astype(np.int64)
    out = np.full_like(flat_labels, output_unlabeled_value, dtype=np.int32)

    ignore_set = set(int(v) for v in unlabeled_values)
    unique_svs = np.unique(flat_svid)
    n_filled = 0
    for sv in unique_svs:
        idx = np.flatnonzero(flat_svid == sv)
        if idx.size == 0:
            continue
        vals = flat_labels[idx]
        # filter unlabeled/ignored
        m = ~np.isin(vals, list(ignore_set))
        if not np.any(m):
            # no labeled voxels in this SV
            continue
        labeled_vals = vals[m].astype(np.int64)
        # bincount over the observed classes
        max_class = int(labeled_vals.max())
        counts = np.bincount(labeled_vals, minlength=max_class + 1)
        # choose by policy
        if tie_policy == "skip":
            # detect ties at max
            top = counts.max()
            winners = np.flatnonzero(counts == top)
            if winners.size != 1:
                # leave unlabeled
                continue
            chosen = int(winners[0])
        elif tie_policy == "min":
            chosen = int(np.argmax(counts))  # np.argmax returns first (min class) on ties
        elif tie_policy == "max":
            top = counts.max()
            winners = np.flatnonzero(counts == top)
            chosen = int(winners[-1])
        elif tie_policy == "prefer_foreground":
            top = counts.max()
            winners = np.flatnonzero(counts == top)
            if winners.size == 1:
                chosen = int(winners[0])
            else:
                # prefer non-zero among winners
                nonzero = winners[winners != 0]
                chosen = int(nonzero[0] if nonzero.size else winners[0])
        else:
            raise ValueError(f"Unknown tie_policy: {tie_policy}")

        out[idx] = chosen
        n_filled += 1

    return out.reshape(seed_labels.shape), n_filled


def relabel_sequential(labels: np.ndarray) -> Tuple[np.ndarray, int]:
    """Relabel an integer array to have contiguous ids starting at 0.

    Returns (new_labels, n_labels).
    """
    labels = labels.astype(np.int64, copy=False)
    uniques = np.unique(labels)
    lut = {int(k): i for i, k in enumerate(uniques)}
    flat = labels.ravel()
    out = np.empty_like(flat, dtype=np.int32)
    for k, v in lut.items():
        out[flat == k] = v
    return out.reshape(labels.shape), len(uniques)


__all__ = [
    "load_image",
    "load_seed_mask",
    "load_pseudolabel",
    "seed_labels_from_mask_and_pseudo",
    "majority_fill",
    "relabel_sequential",
]

