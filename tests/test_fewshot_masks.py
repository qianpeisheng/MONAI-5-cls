import os
import sys

import numpy as np
import torch

from train_finetune_wp5 import (
    build_points_supervision_mask,
    build_slice_supervision_mask,
)


def _make_synthetic_labels(batch=2, shape=(32, 32, 24), fg_prob=0.2, seed=123):
    rng = np.random.RandomState(seed)
    X, Y, Z = shape
    lbl = np.zeros((batch, 1, X, Y, Z), dtype=np.int64)
    # probabilities: bg 1-fg_prob, split fg_prob across 1..4, rare class 6
    probs = np.array([
        1.0 - fg_prob,        # class 0 (background)
        fg_prob * 0.25,       # class 1
        fg_prob * 0.25,       # class 2
        fg_prob * 0.25,       # class 3
        fg_prob * 0.24,       # class 4
        fg_prob * 0.01,       # class 6 (ignored)
    ], dtype=np.float64)
    probs = probs / probs.sum()  # just in case of rounding
    classes = np.array([0, 1, 2, 3, 4, 6], dtype=np.int64)
    for b in range(batch):
        cls_map = rng.choice(classes, size=(X, Y, Z), p=probs)
        lbl[b, 0] = cls_map
    return torch.from_numpy(lbl)


def test_points_mask_coverage_approx_ratio():
    lbl = _make_synthetic_labels(batch=2, shape=(32, 32, 24), fg_prob=0.3, seed=1).to(torch.device("cpu"))
    # target ~1% coverage with dilation radius 1
    ratio = 0.01
    sup = build_points_supervision_mask(
        labels=lbl,
        ratio=ratio,
        dilate_radius=1,
        balance="proportional",
        max_seeds=-1,
        bg_frac=0.25,
        seed_strategy="random",
    )
    cov = sup.float().mean().item()
    # Coverage should be > 0 and not grossly exceed target (allow some tolerance)
    assert cov > 0.0
    assert cov <= ratio * 1.5


def test_points_mask_coverage_tiny_ratio_no_dilate():
    lbl = _make_synthetic_labels(batch=1, shape=(24, 24, 16), fg_prob=0.2, seed=2)
    ratio = 1e-4
    sup = build_points_supervision_mask(
        labels=lbl,
        ratio=ratio,
        dilate_radius=0,
        balance="uniform",
        max_seeds=1,
        bg_frac=0.0,
        seed_strategy="random",
    )
    cov = sup.float().mean().item()
    # With max_seeds=1 and no dilation, ensure extremely low coverage
    total = float(np.prod(lbl.shape[2:]))
    assert cov <= 1.0 / total + 1e-6


def test_slice_mask_k_from_ratio():
    lbl = _make_synthetic_labels(batch=1, shape=(20, 24, 28), fg_prob=0.2, seed=3)
    # ask for ~10% of Z slices supervised
    Z = lbl.shape[-1]
    ratio = 0.1
    sup = build_slice_supervision_mask(labels=lbl, roi=(20, 24, Z), axis_mode="z", ratio=ratio, k_override=None)
    # Count supervised slices along Z
    z_mask = sup[0, 0].any(dim=(0, 1))  # (Z)
    k = z_mask.long().sum().item()
    assert 1 <= k <= max(1, int(np.ceil(ratio * Z)))
