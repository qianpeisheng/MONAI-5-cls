#!/usr/bin/env python3
"""
Unit tests for supervoxel intensity descriptors.

These tests are intentionally tiny and synthetic to validate exact descriptor
semantics without relying on external data.
"""

from __future__ import annotations

import numpy as np

from wp5.weaklabel.sv_descriptors import (
    compute_sv_descriptor_hist,
    compute_sv_descriptor_moments,
    compute_sv_descriptor_quantiles,
)


def _np_quantile(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """NumPy quantile with method='linear' (with backward-compatible fallback)."""
    try:
        return np.quantile(x, q, method="linear")
    except TypeError:
        # Older numpy uses 'interpolation' kwarg.
        return np.quantile(x, q, interpolation="linear")


def test_moments_descriptor_matches_numpy_linear_quantiles():
    # Two SVs with known intensities.
    v0 = np.arange(10, dtype=np.float32)  # 0..9
    v1 = np.full(4, 10.0, dtype=np.float32)  # constant

    sv_ids = np.array([0] * len(v0) + [1] * len(v1), dtype=np.int32).reshape(2, 1, 7)
    img = np.array(list(v0) + list(v1), dtype=np.float32).reshape(2, 1, 7)
    unique_svs = np.array([0, 1], dtype=np.int32)

    desc = compute_sv_descriptor_moments(
        sv_ids, img, unique_svs, trim_ratio=0.1, quantile_method="linear"
    )
    assert desc.shape == (2, 8)

    q = np.array([0.10, 0.25, 0.50, 0.75, 0.90], dtype=np.float64)

    med0 = float(_np_quantile(v0, np.array(0.50)))
    mad0 = float(_np_quantile(np.abs(v0 - med0), np.array(0.50)))
    k0 = int(np.floor(0.1 * v0.size))
    tmean0 = float(v0[k0 : v0.size - k0].mean())
    p0 = _np_quantile(v0, q).astype(np.float64)

    med1 = float(_np_quantile(v1, np.array(0.50)))
    mad1 = float(_np_quantile(np.abs(v1 - med1), np.array(0.50)))
    k1 = int(np.floor(0.1 * v1.size))
    tmean1 = float(v1[k1 : v1.size - k1].mean()) if (v1.size - 2 * k1) > 0 else float(v1.mean())
    p1 = _np_quantile(v1, q).astype(np.float64)

    exp0 = np.array([med0, mad0, tmean0, p0[0], p0[1], p0[2], p0[3], p0[4]], dtype=np.float64)
    exp1 = np.array([med1, mad1, tmean1, p1[0], p1[1], p1[2], p1[3], p1[4]], dtype=np.float64)

    np.testing.assert_allclose(desc[0], exp0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(desc[1], exp1, rtol=0, atol=1e-6)


def test_quantiles16_descriptor_matches_numpy():
    v0 = np.arange(10, dtype=np.float32)
    v1 = np.full(4, 10.0, dtype=np.float32)
    sv_ids = np.array([0] * len(v0) + [1] * len(v1), dtype=np.int32).reshape(2, 1, 7)
    img = np.array(list(v0) + list(v1), dtype=np.float32).reshape(2, 1, 7)
    unique_svs = np.array([0, 1], dtype=np.int32)

    desc = compute_sv_descriptor_quantiles(
        sv_ids,
        img,
        unique_svs,
        n_quantiles=16,
        include_mad=False,
        quantile_method="linear",
    )
    assert desc.shape == (2, 16)

    q = np.linspace(0.0, 1.0, 16, dtype=np.float64)
    exp0 = _np_quantile(v0, q).astype(np.float64)
    exp1 = _np_quantile(v1, q).astype(np.float64)
    np.testing.assert_allclose(desc[0], exp0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(desc[1], exp1, rtol=0, atol=1e-6)

    desc_mad = compute_sv_descriptor_quantiles(
        sv_ids,
        img,
        unique_svs,
        n_quantiles=16,
        include_mad=True,
        quantile_method="linear",
    )
    assert desc_mad.shape == (2, 17)
    med0 = float(_np_quantile(v0, np.array(0.50)))
    mad0 = float(_np_quantile(np.abs(v0 - med0), np.array(0.50)))
    med1 = float(_np_quantile(v1, np.array(0.50)))
    mad1 = float(_np_quantile(np.abs(v1 - med1), np.array(0.50)))
    np.testing.assert_allclose(desc_mad[0, -1], mad0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(desc_mad[1, -1], mad1, rtol=0, atol=1e-6)


def test_hist_descriptor_is_normalized_and_matches_expected_counts():
    v0 = np.arange(10, dtype=np.float32)
    v1 = np.full(4, 10.0, dtype=np.float32)
    sv_ids = np.array([0] * len(v0) + [1] * len(v1), dtype=np.int32).reshape(2, 1, 7)
    img = np.array(list(v0) + list(v1), dtype=np.float32).reshape(2, 1, 7)
    unique_svs = np.array([0, 1], dtype=np.int32)

    # Use a small number of bins and a wide range for deterministic expectations.
    h = compute_sv_descriptor_hist(sv_ids, img, unique_svs, bins=4, vmin=0.0, vmax=12.0)
    assert h.shape == (2, 4)

    # SV0: values 0..9 across 4 bins in [0,12] -> counts [3,3,3,1]
    exp0 = np.array([0.3, 0.3, 0.3, 0.1], dtype=np.float64)
    # SV1: all 10 -> last bin -> [0,0,0,1]
    exp1 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    np.testing.assert_allclose(h[0].sum(), 1.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(h[1].sum(), 1.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(h[0], exp0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(h[1], exp1, rtol=0, atol=1e-6)

