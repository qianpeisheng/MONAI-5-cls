#!/usr/bin/env python3
"""
Unit tests for lightweight visualization helpers used by Streamlit apps.
"""

import numpy as np
import pytest

from wp5.weaklabel.vis_utils import boundary_band_mask, error_mask, outer_bg_distance_mask, transpose_for_imshow


def test_error_mask_respects_ignore_and_unlabeled():
    gt = np.array(
        [
            [0, 1, 6],
            [-1, 2, 0],
        ],
        dtype=np.int16,
    )
    pred = np.array(
        [
            [0, 2, 3],  # mismatch at (0,1) counts; (0,2) ignored because gt==6
            [4, 2, 1],  # (1,0) ignored because gt==-1; mismatch at (1,2) counts
        ],
        dtype=np.int16,
    )

    em = error_mask(pred, gt, ignore_label=6, unlabeled_value=-1)
    assert em.dtype == bool
    assert em.shape == gt.shape
    assert bool(em[0, 1]) is True
    assert bool(em[0, 2]) is False  # ignored label 6
    assert bool(em[1, 0]) is False  # unlabeled -1
    assert bool(em[1, 2]) is True


def test_boundary_band_mask_monotonic_with_radius():
    lbl = np.zeros((16, 16, 16), dtype=np.int16)
    lbl[6:10, 6:10, 6:10] = 1

    b0 = boundary_band_mask(lbl, radius=0.0)
    b1 = boundary_band_mask(lbl, radius=1.0)
    b2 = boundary_band_mask(lbl, radius=2.0)

    assert b0.any()
    assert b1.any()
    assert b2.any()
    assert int(b0.sum()) <= int(b1.sum()) <= int(b2.sum())
    assert np.all(b0 <= b1)
    assert np.all(b1 <= b2)


def test_boundary_band_mask_no_foreground_returns_empty():
    lbl = np.zeros((8, 8, 8), dtype=np.int16)
    b = boundary_band_mask(lbl, radius=2.0)
    assert b.shape == lbl.shape
    assert b.sum() == 0


def test_outer_bg_distance_mask_no_foreground_marks_all_background():
    lbl = np.zeros((8, 8, 8), dtype=np.int16)
    m = outer_bg_distance_mask(lbl, min_distance=5.0)
    assert m.shape == lbl.shape
    assert m.sum() == lbl.size


def test_outer_bg_distance_mask_no_background_returns_empty():
    lbl = np.ones((8, 8, 8), dtype=np.int16)
    m = outer_bg_distance_mask(lbl, min_distance=1.0)
    assert m.shape == lbl.shape
    assert m.sum() == 0


def test_outer_bg_distance_mask_invalid_distance():
    lbl = np.zeros((8, 8, 8), dtype=np.int16)
    with pytest.raises(ValueError):
        outer_bg_distance_mask(lbl, min_distance=-1.0)


def test_transpose_for_imshow_2d():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = transpose_for_imshow(a)
    assert t.shape == (4, 3)
    assert np.all(t == a.T)


def test_transpose_for_imshow_rgba():
    a = np.zeros((7, 11, 4), dtype=np.float32)
    a[2, 3, :] = np.array([1.0, 0.5, 0.25, 0.75], dtype=np.float32)
    t = transpose_for_imshow(a)
    assert t.shape == (11, 7, 4)
    assert np.all(t[3, 2, :] == a[2, 3, :])


def test_transpose_for_imshow_invalid():
    with pytest.raises(ValueError):
        transpose_for_imshow(np.zeros((2, 2, 2), dtype=np.float32))
