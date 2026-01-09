#!/usr/bin/env python3
"""
Unit tests for intensity-aware graph affinity construction.
"""

from __future__ import annotations

import numpy as np

from wp5.weaklabel.graph_affinity import build_spatial_intensity_affinity
from wp5.weaklabel.graph_label_propagation import _build_knn_affinity_matrix


def test_intensity_changes_weights_when_coords_identical():
    # Two nodes at identical coordinates -> spatial similarity should be 1.
    centroids = np.zeros((2, 3), dtype=np.float32)

    # Baseline (coords-only) affinity.
    W_base = _build_knn_affinity_matrix(centroids, k=1, sigma=None).toarray()
    assert W_base.shape == (2, 2)
    assert np.isclose(W_base[0, 1], W_base[1, 0])
    assert W_base[0, 1] > 0.0

    # Intensity descriptors differ -> combined affinity should drop below 1.
    phi = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    W = build_spatial_intensity_affinity(
        centroids=centroids,
        phi=phi,
        k=1,
        sigma_phi=1.0,
        use_cosine=False,
        metric="l2",
        sigma_c=None,
        sample_edges_for_sigma=1000,
        seed=0,
    ).toarray()

    assert W.shape == (2, 2)
    assert np.isclose(W[0, 1], W[1, 0])
    assert W[0, 1] < W_base[0, 1]
