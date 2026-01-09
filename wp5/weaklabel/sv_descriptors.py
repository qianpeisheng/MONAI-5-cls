from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np


DescriptorType = Literal["moments", "quantiles16", "hist32"]


def pclip_zscore(
    x: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Robust per-volume normalization: clip to [p_low,p_high] percentiles, then z-score.

    This matches the normalization used by `scripts/gen_supervoxels_wp5.py:pclip_zscore`.
    """
    arr = x.astype(np.float32, copy=False)
    flat = arr.reshape(-1)
    lo = np.percentile(flat, p_low)
    hi = np.percentile(flat, p_high)
    arr = np.clip(arr, lo, hi)
    mu = arr.mean()
    sd = arr.std()
    return (arr - mu) / (sd + eps)


def _np_quantile_linear(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """NumPy quantile with method='linear', with a fallback for older numpy."""
    try:
        return np.quantile(x, q, method="linear")
    except TypeError:
        return np.quantile(x, q, interpolation="linear")


def _grouped_quantiles_linear_from_sorted(
    vals_sorted: np.ndarray,
    counts: np.ndarray,
    starts: np.ndarray,
    q: np.ndarray,
) -> np.ndarray:
    """Compute linear quantiles per group given values sorted within each group.

    Implements the same "linear" definition as NumPy:
      pos = (n-1) * q
      out = v[floor(pos)] + (pos-floor(pos))*(v[ceil(pos)] - v[floor(pos)])
    """
    if q.ndim != 1:
        raise ValueError("q must be 1D")
    if vals_sorted.ndim != 1:
        raise ValueError("vals_sorted must be 1D")
    if counts.ndim != 1 or starts.ndim != 1:
        raise ValueError("counts and starts must be 1D")
    if counts.size != starts.size:
        raise ValueError("counts and starts must have the same length")

    n = counts.astype(np.float64)
    q = q.astype(np.float64, copy=False)
    pos = (n[:, None] - 1.0) * q[None, :]
    lo = np.floor(pos).astype(np.int64)
    hi = np.ceil(pos).astype(np.int64)
    w = (pos - lo).astype(np.float32)

    gl = starts[:, None] + lo
    gh = starts[:, None] + hi

    v_lo = vals_sorted[gl]
    v_hi = vals_sorted[gh]
    return v_lo + w * (v_hi - v_lo)


def _sort_by_sv_then_value(
    sv_ids: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat_sv = sv_ids.reshape(-1).astype(np.int64, copy=False)
    flat_val = values.reshape(-1).astype(np.float32, copy=False)

    order = np.lexsort((flat_val, flat_sv))
    sv_sorted = flat_sv[order]
    val_sorted = flat_val[order]

    unique_svs, counts = np.unique(sv_sorted, return_counts=True)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return unique_svs.astype(np.int64), counts.astype(np.int64), starts.astype(np.int64), sv_sorted, val_sorted


def _align_rows(unique_all: np.ndarray, rows_all: np.ndarray, unique_svs: np.ndarray) -> np.ndarray:
    """Align computed rows (for all SVs) to a requested SV id list."""
    unique_all = unique_all.astype(np.int64, copy=False)
    unique_svs = unique_svs.astype(np.int64, copy=False)
    if np.array_equal(unique_all, unique_svs):
        return rows_all
    idx = np.searchsorted(unique_all, unique_svs)
    if idx.size != unique_svs.size:
        raise ValueError("Failed to align descriptor rows: size mismatch")
    if not np.all((idx >= 0) & (idx < unique_all.size)):
        raise ValueError("Failed to align descriptor rows: out-of-range indices")
    if not np.all(unique_all[idx] == unique_svs):
        raise ValueError("Failed to align descriptor rows: SV id mismatch")
    return rows_all[idx]


def compute_sv_descriptor_moments(
    sv_ids: np.ndarray,
    img: np.ndarray,
    unique_svs: np.ndarray,
    *,
    trim_ratio: float = 0.1,
    quantile_method: Literal["linear"] = "linear",
) -> np.ndarray:
    """Robust moments descriptor per supervoxel.

    For each supervoxel S with voxel intensities I_v:
      [median, MAD, trimmed_mean(trim_ratio), P10, P25, P50, P75, P90]
    """
    if quantile_method != "linear":
        raise ValueError("Only quantile_method='linear' is supported (to match NumPy).")
    if sv_ids.shape != img.shape:
        raise ValueError(f"sv_ids and img must have the same shape, got {sv_ids.shape} vs {img.shape}")

    unique_all, counts, starts, sv_sorted, val_sorted = _sort_by_sv_then_value(sv_ids, img)

    q = np.array([0.10, 0.25, 0.50, 0.75, 0.90], dtype=np.float64)
    qu = _grouped_quantiles_linear_from_sorted(val_sorted, counts, starts, q).astype(np.float32)

    # qu columns: [P10, P25, P50, P75, P90]
    p10, p25, p50, p75, p90 = (qu[:, i] for i in range(5))
    median = p50

    # Trimmed mean over the sorted values within each group
    trim_ratio = float(np.clip(trim_ratio, 0.0, 0.49))
    k = np.floor(trim_ratio * counts.astype(np.float64)).astype(np.int64)
    keep = counts - 2 * k

    prefix = np.concatenate(([0.0], np.cumsum(val_sorted.astype(np.float64), dtype=np.float64)))
    start_keep = starts + k
    end_keep = starts + (counts - k)
    sum_keep = prefix[end_keep] - prefix[start_keep]

    # Fallback when keep<=0 (very small SVs): use mean over all voxels in that SV.
    end_all = starts + counts
    sum_all = prefix[end_all] - prefix[starts]
    mean_all = (sum_all / np.maximum(counts, 1)).astype(np.float32)
    trimmed_mean = np.where(keep > 0, (sum_keep / np.maximum(keep, 1)).astype(np.float32), mean_all)

    # MAD: median(|x - median(x)|) using the same "linear" quantile definition.
    group_idx = np.repeat(np.arange(unique_all.size, dtype=np.int64), counts)
    dev = np.abs(val_sorted - median[group_idx]).astype(np.float32)

    order_dev = np.lexsort((dev, sv_sorted))
    sv_sorted2 = sv_sorted[order_dev]
    dev_sorted = dev[order_dev]
    unique2, counts2 = np.unique(sv_sorted2, return_counts=True)
    if not np.array_equal(unique2, unique_all):
        raise RuntimeError("Internal error: SV ids changed during MAD sort.")
    starts2 = np.concatenate(([0], np.cumsum(counts2)[:-1])).astype(np.int64)
    mad = _grouped_quantiles_linear_from_sorted(
        dev_sorted.astype(np.float32), counts2.astype(np.int64), starts2, np.array([0.5], dtype=np.float64)
    ).reshape(-1).astype(np.float32)

    out_all = np.stack([median, mad, trimmed_mean, p10, p25, p50, p75, p90], axis=1).astype(np.float32)
    return _align_rows(unique_all, out_all, unique_svs.astype(np.int64, copy=False))


def compute_sv_descriptor_quantiles(
    sv_ids: np.ndarray,
    img: np.ndarray,
    unique_svs: np.ndarray,
    *,
    n_quantiles: int = 16,
    include_mad: bool = False,
    quantile_method: Literal["linear"] = "linear",
) -> np.ndarray:
    """Quantile signature descriptor per supervoxel.

    Default: 16 quantiles at q = 0, 1/15, ..., 1.0, with optional MAD appended.
    """
    if quantile_method != "linear":
        raise ValueError("Only quantile_method='linear' is supported (to match NumPy).")
    if sv_ids.shape != img.shape:
        raise ValueError(f"sv_ids and img must have the same shape, got {sv_ids.shape} vs {img.shape}")
    n_quantiles = int(n_quantiles)
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be >= 2")

    unique_all, counts, starts, sv_sorted, val_sorted = _sort_by_sv_then_value(sv_ids, img)

    q = np.linspace(0.0, 1.0, n_quantiles, dtype=np.float64)
    quant = _grouped_quantiles_linear_from_sorted(val_sorted, counts, starts, q).astype(np.float32)

    if not include_mad:
        return _align_rows(unique_all, quant, unique_svs.astype(np.int64, copy=False))

    # Compute MAD in the same way as in the moments descriptor.
    median = _grouped_quantiles_linear_from_sorted(
        val_sorted, counts, starts, np.array([0.5], dtype=np.float64)
    ).reshape(-1).astype(np.float32)
    group_idx = np.repeat(np.arange(unique_all.size, dtype=np.int64), counts)
    dev = np.abs(val_sorted - median[group_idx]).astype(np.float32)
    order_dev = np.lexsort((dev, sv_sorted))
    sv_sorted2 = sv_sorted[order_dev]
    dev_sorted = dev[order_dev]
    unique2, counts2 = np.unique(sv_sorted2, return_counts=True)
    if not np.array_equal(unique2, unique_all):
        raise RuntimeError("Internal error: SV ids changed during MAD sort.")
    starts2 = np.concatenate(([0], np.cumsum(counts2)[:-1])).astype(np.int64)
    mad = _grouped_quantiles_linear_from_sorted(
        dev_sorted.astype(np.float32), counts2.astype(np.int64), starts2, np.array([0.5], dtype=np.float64)
    ).reshape(-1).astype(np.float32)

    out_all = np.concatenate([quant, mad[:, None]], axis=1).astype(np.float32)
    return _align_rows(unique_all, out_all, unique_svs.astype(np.int64, copy=False))


def compute_sv_descriptor_hist(
    sv_ids: np.ndarray,
    img: np.ndarray,
    unique_svs: np.ndarray,
    *,
    bins: int = 32,
    vmin: float = -3.0,
    vmax: float = 3.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Histogram signature descriptor per supervoxel.

    - Uses a fixed range [vmin, vmax] (values are clipped into this range).
    - Outputs L1-normalized histograms (sum to 1 per SV).
    """
    if sv_ids.shape != img.shape:
        raise ValueError(f"sv_ids and img must have the same shape, got {sv_ids.shape} vs {img.shape}")
    bins = int(bins)
    if bins <= 0:
        raise ValueError("bins must be > 0")
    if not (vmax > vmin):
        raise ValueError("vmax must be > vmin")

    unique_svs_i64 = unique_svs.astype(np.int64, copy=False)

    flat_sv = sv_ids.reshape(-1).astype(np.int64, copy=False)
    flat_val = img.reshape(-1).astype(np.float32, copy=False)

    # Map SV ids to row indices (supports non-contiguous ids and/or subset unique_svs).
    if unique_svs_i64.size > 0 and unique_svs_i64[0] == 0 and np.array_equal(unique_svs_i64, np.arange(unique_svs_i64.size)):
        sv_idx = flat_sv
        in_subset = (sv_idx >= 0) & (sv_idx < unique_svs_i64.size)
    else:
        sv_idx = np.searchsorted(unique_svs_i64, flat_sv)
        in_subset = (sv_idx >= 0) & (sv_idx < unique_svs_i64.size) & (unique_svs_i64[sv_idx] == flat_sv)

    if not np.any(in_subset):
        return np.zeros((unique_svs_i64.size, bins), dtype=np.float32)

    sv_idx = sv_idx[in_subset].astype(np.int64, copy=False)
    vals = flat_val[in_subset].astype(np.float32, copy=False)

    # Clip to range and bin.
    vals = np.clip(vals, float(vmin), float(vmax))
    scale = float(bins) / float(vmax - vmin)
    b = np.floor((vals - float(vmin)) * scale).astype(np.int64)
    b = np.clip(b, 0, bins - 1)

    combined = sv_idx * bins + b
    counts = np.bincount(combined, minlength=int(unique_svs_i64.size * bins)).astype(np.float32)
    h = counts.reshape(int(unique_svs_i64.size), bins)
    denom = h.sum(axis=1, keepdims=True)
    denom = np.maximum(denom, 1.0)
    return (h / denom).astype(np.float32)


@dataclass(frozen=True)
class DescriptorConfig:
    descriptor_type: DescriptorType
    quantiles_include_mad: bool = False
    hist_bins: int = 32
    hist_range: Tuple[float, float] = (-3.0, 3.0)
    moments_trim_ratio: float = 0.1
    quantile_method: str = "linear"
    normalize: str = "pclip_zscore"

    def key(self) -> str:
        payload = {
            "descriptor_type": self.descriptor_type,
            "quantiles_include_mad": bool(self.quantiles_include_mad),
            "hist_bins": int(self.hist_bins),
            "hist_range": [float(self.hist_range[0]), float(self.hist_range[1])],
            "moments_trim_ratio": float(self.moments_trim_ratio),
            "quantile_method": str(self.quantile_method),
            "normalize": str(self.normalize),
        }
        return json.dumps(payload, sort_keys=True)


def save_descriptor_cache(path: Path, *, unique_svs: np.ndarray, phi: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        unique_svs=unique_svs.astype(np.int64, copy=False),
        phi=phi.astype(np.float32, copy=False),
    )


def load_descriptor_cache(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(str(path))
    unique_svs = data["unique_svs"].astype(np.int64, copy=False)
    phi = data["phi"].astype(np.float32, copy=False)
    return unique_svs, phi


def descriptors_equal_unique_svs(unique_a: np.ndarray, unique_b: np.ndarray) -> bool:
    unique_a = unique_a.astype(np.int64, copy=False)
    unique_b = unique_b.astype(np.int64, copy=False)
    return unique_a.shape == unique_b.shape and np.array_equal(unique_a, unique_b)
