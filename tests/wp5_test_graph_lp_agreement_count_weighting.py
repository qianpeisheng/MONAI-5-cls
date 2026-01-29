#!/usr/bin/env python3
"""
Unit tests for agreement-count (multi-level confidence) pseudo-label training support.

This extends the existing Graph-LP binary source-mask weighting to a count-based scheme:
- agree_count in [1..K] encodes raw agreement among K voters.
- agree_count==255 encodes the GT-supported tier (SV contains >=1 GT seed voxel).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from train_finetune_wp5 import (
    build_agree_weight_map_decoupled,
    build_agree_weight_map_table,
    parse_int_float_table,
)


def test_parse_int_float_table_basic():
    assert parse_int_float_table("") == {}
    assert parse_int_float_table("1:0,2:0.02, 4:1") == {1: 0.0, 2: 0.02, 4: 1.0}

    with pytest.raises(ValueError):
        parse_int_float_table("1:0,1:0.1")

    with pytest.raises(ValueError):
        parse_int_float_table("nope")


def test_table_weight_map_defaults_to_zero_and_keeps_gt_tier():
    # 4 voxels: [GT, count4, count2, count1]
    agree = torch.tensor([255, 4, 2, 1], dtype=torch.uint8).view(1, 1, 2, 2, 1)
    valid = torch.ones_like(agree, dtype=torch.bool)

    w = build_agree_weight_map_table(
        agree_count=agree,
        valid_mask=valid,
        weight_table={2: 0.2, 4: 0.4},
        gt_sentinel=255,
        gt_weight=1.0,
    )

    expected = torch.tensor([1.0, 0.4, 0.2, 0.0], dtype=torch.float32)
    assert torch.allclose(w.view(-1), expected, atol=1e-6)

    # Invalid voxels should have zero weight even if GT tier.
    valid2 = valid.clone()
    valid2.view(-1)[0] = False
    w2 = build_agree_weight_map_table(
        agree_count=agree,
        valid_mask=valid2,
        weight_table={2: 0.2, 4: 0.4},
        gt_sentinel=255,
        gt_weight=1.0,
    )
    assert float(w2.view(-1)[0].item()) == 0.0


def test_decoupled_by_count_matches_closed_form_for_ce():
    """
    With w_gt fixed to 1 and w_c = (|G|/|C|)*gamma_c, the weighted-mean CE should satisfy:
        CE = (mean_G + sum_c gamma_c * mean_C) / (1 + sum_c gamma_c)
    where G is the GT tier (agree_count==255) and C are the count tiers.
    """
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 2, 2, 1)  # 4 voxels
    target = torch.tensor([0, 1, 2, 0], dtype=torch.long).view(1, 1, 2, 2, 1)

    # 2 GT-tier voxels, 1 voxel with agree_count=2, 1 voxel with agree_count=1
    agree = torch.tensor([255, 255, 2, 1], dtype=torch.uint8).view(1, 1, 2, 2, 1)
    valid = torch.ones_like(agree, dtype=torch.bool)

    gamma = {2: 0.5, 1: 0.2}
    w = build_agree_weight_map_decoupled(
        agree_count=agree,
        valid_mask=valid,
        gamma_table=gamma,
        gt_sentinel=255,
        gt_weight=1.0,
        fallback_w_by_count=None,
    )

    ce = (
        (F.cross_entropy(logits, target.squeeze(1), reduction="none") * w.squeeze(1)).sum()
        / w.squeeze(1).sum()
    )

    ce_map = F.cross_entropy(logits, target.squeeze(1), reduction="none").view(-1)  # 4 voxels
    agree_flat = agree.view(-1)
    g = (agree_flat == 255)
    c2 = (agree_flat == 2)
    c1 = (agree_flat == 1)

    mean_g = ce_map[g].mean()
    mean_c2 = ce_map[c2].mean()
    mean_c1 = ce_map[c1].mean()

    expected = (mean_g + gamma[2] * mean_c2 + gamma[1] * mean_c1) / (1.0 + gamma[2] + gamma[1])
    assert torch.allclose(ce, expected, atol=1e-6)


def test_ensemble_builder_writes_agree_count_and_encodes_gt_tier(tmp_path: Path):
    """
    The ensemble builder should:
    - write labels/<id>_labels.npy
    - write agreement/<id>_agree_count.npy
    - set agree_count==255 on GT-tier voxels if seed_source_mask_dir is provided
    - write propagation_summary.json with K and gt_sentinel
    """
    from scripts import build_graph_lp_ensemble_labels as ens

    # One case in the datalist.
    datalist = tmp_path / "datalist.json"
    datalist.write_text(json.dumps([{"id": "A"}], indent=2))

    # Two voters (K=2) with different labels in one voxel.
    d1 = tmp_path / "v1"
    d2 = tmp_path / "v2"
    (d1 / "labels").mkdir(parents=True)
    (d2 / "labels").mkdir(parents=True)

    lbl1 = np.zeros((2, 2, 1), dtype=np.int16)
    lbl2 = np.zeros((2, 2, 1), dtype=np.int16)
    lbl2[0, 0, 0] = 1  # disagreement at (0,0,0)
    np.save(d1 / "labels" / "A_labels.npy", lbl1)
    np.save(d2 / "labels" / "A_labels.npy", lbl2)

    # GT-tier mask marks a single voxel.
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = np.zeros((2, 2, 1), dtype=np.uint8)
    src[1, 1, 0] = 1
    np.save(src_dir / "A_source.npy", src)

    out_dir = tmp_path / "out"

    ens.main(
        argv=[
            "--datalist",
            str(datalist),
            "--out_dir",
            str(out_dir),
            "--label_dir",
            f"V1={d1}",
            "--label_dir",
            f"V2={d2}",
            "--tie_break",
            "V1",
            "--seed_source_mask_dir",
            str(src_dir),
        ]
    )

    agree_path = out_dir / "agreement" / "A_agree_count.npy"
    assert agree_path.exists()
    agree_out = np.load(agree_path)

    # K=2 => pseudo voxels should have agree_count 1 or 2. GT-tier voxel should be 255.
    assert agree_out.dtype == np.uint8
    assert agree_out[1, 1, 0] == 255
    assert agree_out[0, 0, 0] == 1  # disagreement between V1 and V2
    assert agree_out[0, 1, 0] == 2  # unanimous background

    summary = json.loads((out_dir / "propagation_summary.json").read_text())
    assert summary["K"] == 2
    assert summary["gt_sentinel"] == 255
