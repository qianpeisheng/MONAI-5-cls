#!/usr/bin/env python3
"""
Unit tests for Graph LP source masks and reliability-weighted losses.

Covered behaviors:
- propagate_graph_lp_multi_case.propagate_case emits a voxelwise source_mask
  that correctly flags SVs with GT-derived seeds.
- create_training_symlinks creates both labels/ and source_masks/ symlinks.
- New loss helpers in train_finetune_wp5 produce expected results:
  * cross_entropy_with_voxel_weights matches a manual weighted average.
  * dice_loss_masked_weighted reduces to dice_loss_masked when weights are 1.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from scripts.propagate_graph_lp_multi_case import (
    propagate_case,
    create_training_symlinks,
)
from train_finetune_wp5 import (
    cross_entropy_with_voxel_weights,
    dice_loss_masked,
    dice_loss_masked_weighted,
)


def test_propagate_case_emits_source_mask(tmp_path: Path):
    """propagate_case should emit source_mask.npy flagging SVs with GT seeds."""
    sv_dir = tmp_path / "sv"
    seeds_dir = tmp_path / "seeds"
    out_dir = tmp_path / "out"
    sv_dir.mkdir()
    seeds_dir.mkdir()

    case_id = "CASE1"

    # Build a tiny 2x2x2 volume with two SVs: id 0 in one half, id 1 in the other.
    sv_ids = np.zeros((2, 2, 2), dtype=np.int32)
    sv_ids[0] = 0  # x=0 slice -> SV 0
    sv_ids[1] = 1  # x=1 slice -> SV 1
    np.save(sv_dir / f"{case_id}_sv_ids.npy", sv_ids)

    # Sparse labels: only SV 0 has a GT-derived seed (class 1).
    sparse = {"sv_labels": {"0": 1}}
    (seeds_dir / f"{case_id}_sv_labels_sparse.json").write_text(
        __import__("json").dumps(sparse, indent=2)
    )

    meta = propagate_case(
        case_id=case_id,
        sv_dir=sv_dir,
        seeds_dir=seeds_dir,
        output_dir=out_dir,
        k=1,
        alpha=0.9,
        num_classes=2,
    )
    assert meta is not None
    assert meta["n_labeled_svs_input"] == 1
    assert meta["n_sv_with_gt"] == 1
    assert meta["n_sv_graph_only"] == meta["n_total_svs"] - 1

    case_dir = out_dir / "cases" / case_id
    assert (case_dir / "propagated_labels.npy").exists()
    assert (case_dir / "source_mask.npy").exists()

    source_mask = np.load(case_dir / "source_mask.npy")
    assert source_mask.shape == sv_ids.shape

    # Voxels in SV 0 should be flagged as 1; SV 1 as 0.
    assert np.all(source_mask[sv_ids == 0] == 1)
    assert np.all(source_mask[sv_ids == 1] == 0)


def test_create_training_symlinks_for_source_masks(tmp_path: Path):
    """create_training_symlinks should create both labels/ and source_masks/ symlinks."""
    out_dir = tmp_path / "out"
    case_id = "CASE2"
    case_dir = out_dir / "cases" / case_id
    case_dir.mkdir(parents=True)

    # Minimal label and source files.
    lbl = np.zeros((2, 2, 2), dtype=np.int16)
    src = np.ones((2, 2, 2), dtype=np.uint8)
    np.save(case_dir / "propagated_labels.npy", lbl)
    np.save(case_dir / "source_mask.npy", src)

    create_training_symlinks(out_dir, [case_id])

    labels_dir = out_dir / "labels"
    src_dir = out_dir / "source_masks"
    lbl_link = labels_dir / f"{case_id}_labels.npy"
    src_link = src_dir / f"{case_id}_source.npy"

    assert lbl_link.is_symlink()
    assert src_link.is_symlink()
    assert lbl_link.resolve() == case_dir / "propagated_labels.npy"
    assert src_link.resolve() == case_dir / "source_mask.npy"


def test_propagate_case_outer_bg_split_skips_outer_sv_lp(tmp_path: Path, monkeypatch):
    """
    When outer-background splitting is enabled, Graph LP should only see ROI SVs
    and outer SVs should be hard-assigned to background (label 0).
    """
    import scripts.propagate_graph_lp_multi_case as glp_mod

    sv_dir = tmp_path / "sv"
    seeds_dir = tmp_path / "seeds"
    out_dir = tmp_path / "out"
    sv_dir.mkdir()
    seeds_dir.mkdir()

    case_id = "CASE_ROI"

    # Tiny 2x1x1 volume with two SVs: id 0 (ROI), id 1 (outer background).
    sv_ids = np.zeros((2, 1, 1), dtype=np.int32)
    sv_ids[0, 0, 0] = 0
    sv_ids[1, 0, 0] = 1
    np.save(sv_dir / f"{case_id}_sv_ids.npy", sv_ids)

    # Sparse labels: only SV 0 has a seed (class 2).
    sparse = {"sv_labels": {"0": 2}}
    (seeds_dir / f"{case_id}_sv_labels_sparse.json").write_text(
        __import__("json").dumps(sparse, indent=2)
    )

    # Seeds metadata with outer background SV id 1.
    seeds_meta = {
        "case_id": case_id,
        "outer_bg_sv_ids": [1],
    }
    (seeds_dir / f"{case_id}_seeds_meta.json").write_text(
        __import__("json").dumps(seeds_meta, indent=2)
    )

    # Stub graph_label_propagation to record how many SVs it sees.
    called = {}

    def fake_graph_lp(features, sv_labels, num_classes, k, alpha, sigma=None):
        called["N"] = features.shape[0]
        # Predict class 2 for all ROI SVs regardless of seeds.
        return np.full(len(sv_labels), 2, dtype=np.int64)

    monkeypatch.setattr(glp_mod, "graph_label_propagation", fake_graph_lp)

    meta = glp_mod.propagate_case(
        case_id=case_id,
        sv_dir=sv_dir,
        seeds_dir=seeds_dir,
        output_dir=out_dir,
        k=1,
        alpha=0.9,
        num_classes=3,
        use_outer_bg_split=True,
    )

    assert meta is not None
    # Graph LP should only see the ROI SV (id 0).
    assert called.get("N") == 1

    case_dir = out_dir / "cases" / case_id
    dense_labels = np.load(case_dir / "propagated_labels.npy")
    assert dense_labels.shape == sv_ids.shape

    # ROI SV (0) should take the propagated label (2).
    assert np.all(dense_labels[sv_ids == 0] == 2)
    # Outer background SV (1) should be forced to background (0), not 2.
    assert np.all(dense_labels[sv_ids == 1] == 0)


def test_cross_entropy_with_voxel_weights_matches_manual():
    """cross_entropy_with_voxel_weights should match a manual weighted CE."""
    torch.manual_seed(0)
    # Shape: B=1, C=3, X=2, Y=1, Z=1 => 2 voxels
    logits = torch.randn(1, 3, 2, 1, 1)
    # Target and weights with shape (1,1,2,1,1)
    target = torch.tensor([0, 2], dtype=torch.long).view(1, 1, 2, 1, 1)
    weights = torch.tensor([1.0, 2.0], dtype=torch.float32).view(1, 1, 2, 1, 1)

    ce_map = F.cross_entropy(logits, target.squeeze(1), reduction="none")  # (1,2,1,1)
    w = weights.squeeze(1)
    manual = (ce_map * w).sum() / w.sum()

    loss = cross_entropy_with_voxel_weights(
        logits=logits,
        target=target,
        weight_map=weights,
        ignore_index=255,
    )
    assert torch.allclose(loss, manual, atol=1e-6)


def test_dice_loss_weighted_equals_unweighted_for_unit_weights():
    """dice_loss_masked_weighted should reduce to dice_loss_masked for unit weights."""
    torch.manual_seed(0)
    B, C, X, Y, Z = 2, 5, 4, 4, 4
    logits = torch.randn(B, C, X, Y, Z)
    target = torch.randint(low=0, high=5, size=(B, 1, X, Y, Z), dtype=torch.long)
    ignore_mask = torch.ones_like(target, dtype=torch.bool)
    weights = torch.ones_like(target, dtype=torch.float32)

    base = dice_loss_masked(logits, target, ignore_mask)
    weighted = dice_loss_masked_weighted(logits, target, ignore_mask, weights)

    assert torch.allclose(base, weighted, atol=1e-6)
