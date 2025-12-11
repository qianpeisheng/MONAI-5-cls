#!/usr/bin/env python3
"""
Tests for adaptive outer-background distance selection utilities.

These tests focus on the pure NumPy/SciPy helpers in wp5.weaklabel.outer_bg_utils
so they do not require MONAI or torch.
"""

import numpy as np
import pytest

from wp5.weaklabel.outer_bg_utils import choose_outer_bg_distance


def _make_simple_volume():
    """
    Create a simple 3D volume with a compact foreground blob in the center.
    Background is class 0; foreground is class 1.
    """
    gt = np.zeros((32, 32, 32), dtype=np.int16)
    gt[12:20, 12:20, 12:20] = 1
    return gt


def test_choose_outer_bg_distance_basic_fraction():
    """Chosen distance should yield an outer BG fraction close to the target."""
    gt = _make_simple_volume()

    target = 0.7  # want ~70% of BG voxels to be outer BG
    thr, actual = choose_outer_bg_distance(
        gt_labels=gt,
        target_outer_bg_frac=target,
        min_distance=1.0,
        max_distance=32.0,
    )

    # Basic sanity: threshold is within the requested clamp range.
    assert 1.0 <= thr <= 32.0

    # Recompute actual fraction explicitly to verify consistency.
    from scipy.ndimage import distance_transform_edt

    fg_mask = gt > 0
    bg_mask = gt == 0
    dist_to_fg = distance_transform_edt(~fg_mask)
    vals = dist_to_fg[bg_mask]
    frac = float((vals > thr).sum()) / float(vals.size)

    # The actual fraction should be reasonably close to the target.
    assert abs(frac - target) < 0.1
    # And should match what the helper returned.
    assert pytest.approx(frac, rel=1e-6) == actual


def test_choose_outer_bg_distance_no_background():
    """When there is no background, the helper should degrade gracefully."""
    gt = np.ones((8, 8, 8), dtype=np.int16)  # all foreground

    thr, actual = choose_outer_bg_distance(gt_labels=gt, target_outer_bg_frac=0.5)

    # With no background, we expect minimal threshold and zero fraction.
    assert thr == pytest.approx(1.0)
    assert actual == pytest.approx(0.0)


def test_choose_outer_bg_distance_no_foreground():
    """When there is no foreground, all background should be outer BG."""
    gt = np.zeros((8, 8, 8), dtype=np.int16)  # all background

    thr, actual = choose_outer_bg_distance(gt_labels=gt, target_outer_bg_frac=0.5)

    # With no FG, the helper should mark all BG as outer, independent of target.
    assert thr == pytest.approx(32.0)
    assert actual == pytest.approx(1.0)


def test_choose_outer_bg_distance_disabled():
    """Target fraction <=0 should disable outer BG and return zero threshold."""
    gt = _make_simple_volume()

    thr, actual = choose_outer_bg_distance(gt_labels=gt, target_outer_bg_frac=0.0)
    assert thr == 0.0
    assert actual == 0.0

