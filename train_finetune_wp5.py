#!/usr/bin/env python3
"""
WP5 fine-tuning script (full or partial data) with reproducible training, per-epoch evaluation, and checkpoint/inference utilities.

Usage examples:

Train on full data and evaluate each epoch (defaults: BasicUNet, scratch init):
  python train_finetune_wp5.py --mode train \
    --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
    --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json \
    --output_dir runs/wp5_finetune_full --subset_ratio 1.0 --epochs 50 --batch_size 2 --lr 1e-4

Train on 10% (deterministic subset) and evaluate:
  python train_finetune_wp5.py --mode train \
    --subset_ratio 0.1 --seed 42 --output_dir runs/wp5_finetune_10pct

Optionally, train from a pretrained checkpoint (non-strict load for head):
  python train_finetune_wp5.py --mode train \
    --init pretrained --pretrained_ckpt path/to/pretrained.ckpt \
    --output_dir runs/wp5_finetune_pretrained

Train from scratch explicitly:
  python train_finetune_wp5.py --mode train --init scratch --output_dir runs/wp5_finetune_scratch

Evaluate a saved checkpoint on the test set:
  # Use the dedicated evaluator (timestamps + HD95/HD + policy metadata)
  python scripts/eval_wp5.py \
    --ckpt runs/wp5_finetune_full/best.ckpt \
    --datalist datalist_test.json \
    --output_dir runs/wp5_finetune_full/eval \
    --save_preds --heavy --hd_percentile 95

Notes:
- Label policy: evaluate classes 0..4 (background included), ignore class 6 in loss/metrics.
- Saves per-epoch metrics JSON and overall CSV under <output_dir>/metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.networks.nets import UNet, BasicUNet
from monai.bundle import ConfigParser
import torch.nn as nn
from monai.networks import one_hot  # still used elsewhere if needed
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandSpatialCropd,
    SpatialPadd,
    SaveImage,
    MapTransform,
    ScaleIntensityRanged,
    RandGaussianNoised,
    RandShiftIntensityd,
    RandScaleIntensityd,
)
from monai.utils import set_determinism
import sys
import logging
try:
    from monai.optimizers import Novograd  # preferred optimizer per MONAI bundle
except Exception:  # pragma: no cover
    Novograd = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_determinism(seed)


def build_datalists(data_dir: Path, cfg_path: Path) -> Tuple[List[Dict], List[Dict]]:
    cfg = json.loads(cfg_path.read_text())
    test_serials = set(cfg.get("test_serial_numbers", []))

    def serial_from_name(n: str):
        m = re.match(r"^SN(\d+)", n)
        return int(m.group(1)) if m else None

    pairs: Dict[str, Tuple[str, str, int]] = {}
    for n in os.listdir(data_dir):
        if n.endswith("_image.nii"):
            base = n[:-10]
            img = str(data_dir / f"{base}_image.nii")
            lbl = str(data_dir / f"{base}_label.nii")
            if os.path.exists(lbl):
                pairs[base] = (img, lbl, serial_from_name(n))

    train, test = [], []
    for k, (img, lbl, serial) in pairs.items():
        rec = {"image": img, "label": lbl, "id": k}
        (test if serial in test_serials else train).append(rec)
    return train, test


def override_train_labels(train_list: List[Dict], override_dir: Path) -> List[Dict]:
    """Override training label paths with files from override_dir mirroring basenames.

    Keeps 'image' unchanged. Only adjusts 'label' for training records.
    """
    out: List[Dict] = []
    for rec in train_list:
        lbl_path = Path(rec["label"]) if rec.get("label") else None
        if lbl_path is None:
            out.append(rec)
            continue
        new_lbl = override_dir / lbl_path.name
        r = dict(rec)
        r["label"] = str(new_lbl)
        out.append(r)
    return out


def subset_datalist(datalist: List[Dict], ratio: float, seed: int) -> List[Dict]:
    if ratio >= 0.999:
        return list(datalist)
    n = max(1, int(len(datalist) * ratio))
    rng = random.Random(seed)
    idxs = list(range(len(datalist)))
    rng.shuffle(idxs)
    idxs = sorted(idxs[:n])
    return [datalist[i] for i in idxs]


def _load_label_volume(path: str) -> np.ndarray:
    """Load label volume using MONAI loader to ensure RAS orientation.

    Returns channel-first array (1,X,Y,Z), dtype int64.
    Using MONAI's LoadImaged + EnsureChannelFirstd + Orientationd provides
    consistent orientation with the training pipeline, avoiding mismatches
    with any nibabel default orientation.
    """
    from monai.transforms import Compose as _Compose, LoadImaged as _LoadImaged, EnsureChannelFirstd as _EnsureChannelFirstd, Orientationd as _Orientationd
    data = {"label": path}
    t = _Compose([
        _LoadImaged(keys=["label"]),
        _EnsureChannelFirstd(keys=["label"]),
        _Orientationd(keys=["label"], axcodes="RAS"),
    ])
    d = t(data)
    arr = d["label"].astype(np.int64)
    return arr


def precompute_static_global_seed_masks(
    train_list: List[Dict],
    masks_dir: Path,
    ratio: float,
    seed_bg_frac: float,
    dilate_radius: int,
    balance: str,
    no_overlap: bool,
    dilation_shape: str = "auto",
    seed: int = 42,
    sample_mode: str = "stratified",
    uniform_exclude6: bool = False,
) -> None:
    """Precompute global seed and dilated supervision masks, plus propagated pseudo labels.

    - Global seed budget: total_seeds = round(ratio * sum_voxels)
    - Background seeds fraction given by seed_bg_frac (rest distributed to classes 1..4 proportionally or uniformly per volume)
    - Enforces no_overlap across seeds within each volume if requested.
    - Saves: <id>_seedmask.npy, <id>_supmask.npy, <id>_pseudolabel.npy, and stats JSON.
    """
    rng = np.random.RandomState(seed)
    # First pass: gather shapes and per-class counts
    infos = []
    total_vox = 0
    for rec in train_list:
        lblp = rec["label"]
        arr = _load_label_volume(lblp)  # (1,X,Y,Z)
        X, Y, Z = arr.shape[1:]
        total_vox += X * Y * Z
        counts = {c: int((arr == c).sum()) for c in [0, 1, 2, 3, 4]}
        infos.append({"id": rec["id"], "shape": (X, Y, Z), "path": lblp, "counts": counts})

    total_seeds = int(np.round(max(1e-6, ratio) * total_vox))
    bg_seeds_total = int(np.floor(total_seeds * np.clip(seed_bg_frac, 0.0, 0.9)))
    fg_seeds_total = max(0, total_seeds - bg_seeds_total)

    # Allocate per-volume seeds proportional to volume voxels
    per_vol = []
    for inf in infos:
        vol_vox = np.prod(inf["shape"])
        vol_share = vol_vox / total_vox
        seeds_v = int(np.floor(total_seeds * vol_share))
        bg_v = int(np.floor(bg_seeds_total * vol_share))
        fg_v = max(0, seeds_v - bg_v)
        per_vol.append({"id": inf["id"], "seeds": seeds_v, "bg": bg_v, "fg": fg_v})

    # Sampling helper on numpy arrays, using torch for dilation
    import torch
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    # shape selection
    shape = dilation_shape
    if shape == "auto":
        shape = "cross" if ratio >= 0.1 - 1e-8 else "cube"
    if shape not in {"cube", "cross"}:
        raise ValueError("dilation_shape must be one of {'cube','cross','auto'}")

    masks_dir.mkdir(parents=True, exist_ok=True)
    k = 2 * dilate_radius + 1

    def cross_dilate_one(mask_3d: torch.Tensor) -> torch.Tensor:
        out = mask_3d.clone()
        tmp = torch.zeros_like(mask_3d)
        tmp[1:, :, :] |= mask_3d[:-1, :, :]
        tmp[:-1, :, :] |= mask_3d[1:, :, :]
        out |= tmp
        tmp.zero_()
        tmp[:, 1:, :] |= mask_3d[:, :-1, :]
        tmp[:, :-1, :] |= mask_3d[:, 1:, :]
        out |= tmp
        tmp.zero_()
        tmp[:, :, 1:] |= mask_3d[:, :, :-1]
        tmp[:, :, :-1] |= mask_3d[:, :, 1:]
        out |= tmp
        return out

    def cross_dilate(mask_3d: torch.Tensor, r: int) -> torch.Tensor:
        out = mask_3d.clone()
        frontier = mask_3d.clone()
        for _ in range(r):
            expanded = cross_dilate_one(frontier)
            frontier = expanded & (~out)
            out |= frontier
        return out
    for inf, alloc in zip(infos, per_vol):
        case_id = inf["id"]
        X, Y, Z = inf["shape"]
        arr = _load_label_volume(inf["path"])  # (1,X,Y,Z)
        lbl = torch.from_numpy(arr[0])  # (X,Y,Z)
        seed_mask = torch.zeros((X, Y, Z), dtype=torch.bool)
        blocked = torch.zeros((X, Y, Z), dtype=torch.bool) if no_overlap else None

        if sample_mode == "uniform_all":
            # Uniform random sampling across ALL voxels (including background)
            total_voxels = X * Y * Z
            if uniform_exclude6:
                # Eligible = not class 6
                elig = (lbl != 6)
                elig_idx = elig.nonzero(as_tuple=False)
                elig_cnt = int(elig_idx.shape[0])
                seeds_v = max(0, int(np.round(ratio * elig_cnt)))
                if seeds_v >= elig_cnt:
                    seed_mask[:, :, :] = elig
                elif seeds_v > 0 and elig_cnt > 0:
                    perm = torch.randperm(elig_cnt, generator=g)[:seeds_v]
                    sel = elig_idx[perm]
                    seed_mask[sel[:, 0], sel[:, 1], sel[:, 2]] = True
            else:
                seeds_v = max(0, int(np.round(ratio * total_voxels)))
                if seeds_v >= total_voxels:
                    seed_mask[:, :, :] = True
                elif seeds_v > 0:
                    # Sample unique flat indices, then unravel
                    perm = torch.randperm(total_voxels, generator=g)[:seeds_v]
                    yz = Y * Z
                    xs = (perm // yz)
                    rem = perm % yz
                    ys = rem // Z
                    zs = rem % Z
                    seed_mask[xs, ys, zs] = True
        else:
            # Foreground allocation across classes 1..4 + background share
            class_idxs = {}
            for c in [1, 2, 3, 4]:
                idx = (lbl == c).nonzero(as_tuple=False)
                class_idxs[c] = idx
            bg_idx = (lbl == 0).nonzero(as_tuple=False)

            fg_total = alloc["fg"]
            bg_total = alloc["bg"]

            if fg_total > 0:
                if balance == "uniform":
                    per_cls = np.full(4, fg_total // 4, dtype=int)
                    per_cls[: fg_total % 4] += 1
                    cls_order = [1, 2, 3, 4]
                else:
                    counts = np.array([
                        class_idxs[1].shape[0], class_idxs[2].shape[0], class_idxs[3].shape[0], class_idxs[4].shape[0]
                    ], dtype=float)
                    s = counts.sum()
                    if s > 0:
                        per_cls = np.floor(fg_total * (counts / s)).astype(int)
                        rem = fg_total - per_cls.sum()
                        order = np.argsort(-counts)
                        for i in range(rem):
                            per_cls[order[i % len(order)]] += 1
                    else:
                        per_cls = np.zeros(4, dtype=int)
                    cls_order = [1, 2, 3, 4]

                for i, c in enumerate(cls_order):
                    n = int(per_cls[i])
                    if n <= 0:
                        continue
                    idx = class_idxs[c]
                    if idx.shape[0] == 0:
                        continue
                    if no_overlap and blocked is not None and idx.shape[0] > 0:
                        ok = ~blocked[idx[:, 0], idx[:, 1], idx[:, 2]]
                        idx = idx[ok]
                    if idx.shape[0] == 0:
                        continue
                    perm = torch.randperm(idx.shape[0], generator=g)[: min(n, idx.shape[0])]
                    sel = idx[perm]
                    seed_mask[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                    if no_overlap and blocked is not None:
                        added = torch.zeros_like(seed_mask)
                        added[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                        if shape == "cube":
                            dil = torch.nn.functional.max_pool3d(
                                added[None, None].float(), kernel_size=k, stride=1, padding=dilate_radius
                            ) > 0.5
                            blocked |= dil[0, 0]
                        else:
                            blocked |= cross_dilate(added, dilate_radius)

            # Background
            if bg_total > 0 and bg_idx.shape[0] > 0:
                idx = bg_idx
                if no_overlap and blocked is not None and idx.shape[0] > 0:
                    ok = ~blocked[idx[:, 0], idx[:, 1], idx[:, 2]]
                    idx = idx[ok]
                if idx.shape[0] > 0:
                    perm = torch.randperm(idx.shape[0], generator=g)[: min(bg_total, idx.shape[0])]
                    sel = idx[perm]
                    seed_mask[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                    if no_overlap and blocked is not None:
                        added = torch.zeros_like(seed_mask)
                        added[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                        if shape == "cube":
                            dil = torch.nn.functional.max_pool3d(
                                added[None, None].float(), kernel_size=k, stride=1, padding=dilate_radius
                            ) > 0.5
                            blocked |= dil[0, 0]
                        else:
                            blocked |= cross_dilate(added, dilate_radius)

        # Build dilated supervision mask and propagated pseudo labels
        if shape == "cube":
            seed_t = seed_mask[None, None].float()
            sup = torch.nn.functional.max_pool3d(seed_t, kernel_size=k, stride=1, padding=dilate_radius) > 0.5
            sup = sup[0, 0]
            pseudo = torch.full_like(lbl, fill_value=0, dtype=torch.int64)
            # Prioritize FG over BG in overlaps deterministically
            for c in [1, 2, 3, 4, 0]:
                scls = (lbl == c) & seed_mask
                if scls.any():
                    scls_t = scls[None, None].float()
                    dil_c = torch.nn.functional.max_pool3d(
                        scls_t, kernel_size=k, stride=1, padding=dilate_radius
                    ) > 0.5
                    vox = dil_c[0, 0]
                    pseudo[vox] = int(c)
        else:
            # cross: Manhattan BFS with tie-break: FG classes before BG
            assigned = seed_mask.clone()
            pseudo = torch.full_like(lbl, fill_value=-1, dtype=torch.int16)
            # seed assignments
            for c in [1, 2, 3, 4, 0]:
                m = seed_mask & (lbl == c)
                pseudo[m] = int(c)
            for _ in range(dilate_radius):
                assigned_any = pseudo != -1
                for c in [1, 2, 3, 4, 0]:
                    src = pseudo == c
                    if not src.any():
                        continue
                    neigh = cross_dilate_one(src)
                    cand = neigh & (~assigned_any)
                    if cand.any():
                        pseudo[cand] = int(c)
                        assigned_any |= cand
            sup = (pseudo != -1)
            pseudo = pseudo.to(torch.int64)

        # Save masks and stats
        import numpy as _np
        safe_id = str(case_id).replace('/', '_')
        _np.save(masks_dir / f"{safe_id}_seedmask.npy", seed_mask.cpu().numpy()[None, ...])
        _np.save(masks_dir / f"{safe_id}_supmask.npy", sup.cpu().numpy()[None, ...])
        _np.save(masks_dir / f"{safe_id}_pseudolabel.npy", pseudo.cpu().numpy()[None, ...])
        # Stats with clarity about denominators
        total_vox = X * Y * Z
        if sample_mode == "uniform_all" and uniform_exclude6:
            elig_cnt = int(((lbl != 6).sum()).item())
            elig_frac = float(elig_cnt) / float(total_vox)
            denom_mode = "eligible(lbl!=6)"
        else:
            elig_cnt = total_vox
            elig_frac = 1.0
            denom_mode = "all"
        stats = {
            "id": case_id,
            "shape": [1, X, Y, Z],
            "seed_fraction": float(seed_mask.float().mean().item()),
            "sup_fraction": float(sup.float().mean().item()),
            "eligible_fraction": float(elig_frac),
            "eligible_count": int(elig_cnt),
            "ratio": float(ratio),
            "denominator": denom_mode,
            "dilation_shape": shape,
            "radius": int(dilate_radius),
        }
        (masks_dir / f"{safe_id}_supmask_stats.json").write_text(json.dumps(stats, indent=2))


class ClipZScoreNormalizeD(MapTransform):
    """Per-sample robust normalization for images: clip to [p1, p99] then z-score.

    Applies to keys provided (typically ["image"]). Assumes channel-first arrays after EnsureChannelFirstd.
    """

    def __init__(self, keys: List[str]):
        super().__init__(keys)
        self.eps = 1e-8

    def __call__(self, data):
        d = dict(data)
        import numpy as np

        for key in self.keys:
            arr = d.get(key)
            if arr is None:
                continue
            # Compute percentiles over spatial dims (all except channel dim 0)
            # Expect shape (C, X, Y, Z). Handle 3D or 4D generically.
            if arr.ndim >= 3:
                # Flatten spatial dims for percentile calc
                if arr.ndim == 3:
                    flat = arr.reshape(-1)
                else:
                    flat = arr.reshape(arr.shape[0], -1)
                    flat = flat.reshape(-1)
            else:
                flat = arr.reshape(-1)

            p1 = np.percentile(flat, 1)
            p99 = np.percentile(flat, 99)
            clipped = np.clip(arr, p1, p99)
            mean = clipped.mean()
            std = clipped.std()
            d[key] = ((clipped - mean) / (std + self.eps)).astype(np.float32)

        return d


class FGBiasedCropD(MapTransform):
    """Foreground-biased random crop for 3D volumes.

    With probability `prob`, picks a random foreground voxel (label in {1..4}) as center and crops ROI.
    Falls back to uniform random crop if no foreground is present or by probability.
    Assumes channel-first arrays after EnsureChannelFirstd and expects keys ['image','label'].
    """

    def __init__(self, keys: List[str], roi_size: Tuple[int, int, int], prob: float = 0.0, margin: int = 8):
        super().__init__(keys)
        self.roi = roi_size
        self.prob = float(prob)
        self.margin = int(margin)

    def __call__(self, data):
        import numpy as np
        d = dict(data)

        img = d.get("image")
        lbl = d.get("label")
        # Require image and label with channel-first 3D shapes
        if img is None or lbl is None or lbl.ndim != 4 or img.ndim != 4:
            return d

        # Expect shapes (C, X, Y, Z)
        _, X, Y, Z = lbl.shape
        rx, ry, rz = self.roi

        # Valid region for crop corner so that ROI fits inside
        minx = 0
        miny = 0
        minz = 0
        maxx = max(0, X - rx)
        maxy = max(0, Y - ry)
        maxz = max(0, Z - rz)

        def apply_crop(dct, sx, sy, sz):
            for k in self.keys:
                arr = dct.get(k)
                if arr is None or arr.ndim != 4:
                    continue
                dct[k] = arr[:, sx : sx + rx, sy : sy + ry, sz : sz + rz]

        # Default: fallback to uniform random crop within valid region
        sx = int(np.random.randint(minx, maxx + 1)) if maxx > minx else 0
        sy = int(np.random.randint(miny, maxy + 1)) if maxy > miny else 0
        sz = int(np.random.randint(minz, maxz + 1)) if maxz > minz else 0

        # Try foreground-biased selection with probability self.prob
        use_fg = (self.prob > 0.0) and (np.random.rand() <= self.prob)
        if use_fg:
            # Find foreground voxels ignoring class 6
            fg_mask = (lbl[0] != 0) & (lbl[0] != 6)
            fg_indices = np.argwhere(fg_mask)
            if fg_indices.size > 0:
                # Pick a random foreground center and center ROI around it
                cx, cy, cz = fg_indices[np.random.randint(0, fg_indices.shape[0])]
                sx = int(np.clip(cx - rx // 2, minx, maxx))
                sy = int(np.clip(cy - ry // 2, miny, maxy))
                sz = int(np.clip(cz - rz // 2, minz, maxz))
            # else: keep uniform random sx,sy,sz computed above

        # Always return a cropped ROI to enforce fixed-size batches
        apply_crop(d, sx, sy, sz)
        return d


class BuildStaticPointsMaskD(MapTransform):
    """Build and cache a fixed per-volume supervision mask for few_points.

    - Computes sparse point seeds + dilation from the full label volume once per case (by `id`).
    - Stores the boolean mask under `out_key` (default 'sup_mask').
    - Subsequent spatial transforms (pad/flip/crop) must include this key so it aligns with the label.
    """

    def __init__(
        self,
        keys: list[str],
        out_key: str = "sup_mask",
        id_key: str = "id",
        ratio: float = 0.01,
        dilate_radius: int = 1,
        balance: str = "proportional",
        max_seeds: int = -1,
        bg_frac: float = 0.25,
        seed_strategy: str = "random",
        no_overlap_after_dilation: bool = False,
        save_dir: str | None = None,
    ):
        super().__init__(keys)
        self.out_key = out_key
        self.id_key = id_key
        self.ratio = float(ratio)
        self.dilate_radius = int(dilate_radius)
        self.balance = str(balance)
        self.max_seeds = int(max_seeds)
        self.bg_frac = float(bg_frac)
        self.seed_strategy = str(seed_strategy)
        self._cache = {}
        self.save_dir = save_dir
        self.no_overlap_after_dilation = bool(no_overlap_after_dilation)

    def __call__(self, data):
        import numpy as np
        d = dict(data)
        case_id = d.get(self.id_key)
        # Handle list/meta id structure
        if isinstance(case_id, list) and case_id:
            case_id = case_id[0]
        if case_id is None:
            case_id = f"case_{id(d)}"

        if case_id in self._cache:
            sup = self._cache[case_id]
            # Ensure dtype and shape
            d[self.out_key] = sup.copy()
            return d

        # Expect labels under first key
        lbl = d.get(self.keys[0])
        if lbl is None:
            return d
        # Ensure channel-first (C,X,Y,Z) with C==1
        arr = lbl
        if isinstance(arr, np.ndarray):
            # Convert to torch for reuse of the existing builder
            import torch
            t = torch.from_numpy(arr).long().unsqueeze(0) if arr.ndim == 3 else torch.from_numpy(arr).long()
        else:
            import torch
            t = arr if isinstance(arr, torch.Tensor) else torch.as_tensor(arr).long()

        if t.ndim == 4:
            t = t.unsqueeze(0)  # (1,1,X,Y,Z)

        # Auto max_seeds logic consistent with train loop
        auto_max = self.max_seeds
        if auto_max <= 0:
            auto_max = 5000 if self.ratio >= 0.1 - 1e-8 else 500

        sup_mask, seed_mask = build_points_supervision_mask(
            labels=t, ratio=self.ratio, dilate_radius=self.dilate_radius,
            balance=self.balance, max_seeds=auto_max, bg_frac=self.bg_frac,
            seed_strategy=self.seed_strategy, ensure_min_coverage=True, max_iter=6,
            no_overlap_after_dilation=self.no_overlap_after_dilation, return_seeds=True,
        )  # (1,1,X,Y,Z) bool, (1,1,X,Y,Z) bool

        # Convert back to numpy bool
        sup_np = sup_mask[0].cpu().numpy().astype(bool)
        seed_np = seed_mask[0].cpu().numpy().astype(bool)
        d[self.out_key] = sup_np
        d["seed_mask"] = seed_np
        self._cache[case_id] = sup_np
        # Optional: persist mask and simple stats for reproducibility
        if self.save_dir:
            from pathlib import Path
            import json
            sdir = Path(self.save_dir)
            sdir.mkdir(parents=True, exist_ok=True)
            safe_id = str(case_id).replace('/', '_')
            npy_path = sdir / f"{safe_id}_supmask.npy"
            npy_seed_path = sdir / f"{safe_id}_seedmask.npy"
            stats_path = sdir / f"{safe_id}_supmask_stats.json"
            try:
                # Save mask
                import numpy as _np
                _np.save(npy_path, sup_np)
                _np.save(npy_seed_path, seed_np)
                # Compute simple counts per class on original label arr
                if isinstance(arr, _np.ndarray):
                    lbl_np = arr
                else:
                    lbl_np = _np.asarray(arr)
                counts = {}
                seed_counts = {}
                for c in [0, 1, 2, 3, 4, 6]:
                    counts[str(c)] = int((_np.logical_and(sup_np[0], lbl_np[0] == c)).sum())
                    seed_counts[str(c)] = int((_np.logical_and(seed_np[0], lbl_np[0] == c)).sum())
                frac = float(sup_np.mean())
                seed_frac = float(seed_np.mean())
                stats = {
                    "id": case_id,
                    "shape": list(sup_np.shape),
                    "sup_fraction": frac,
                    "seed_fraction": seed_frac,
                    "counts_per_class": counts,
                    "seed_counts_per_class": seed_counts,
                }
                stats_path.write_text(json.dumps(stats, indent=2))
            except Exception:
                pass
        return d


class LoadSavedMasksD(MapTransform):
    """Load precomputed few-shot masks (seed_mask, sup_mask, pseudo_label) by case id.

    Expects files saved as <dir>/<id>_seedmask.npy, <id>_supmask.npy, <id>_pseudolabel.npy.
    """

    def __init__(self, keys: list[str], id_key: str, dir_path: str):
        super().__init__(keys)
        self.id_key = id_key
        self.dir = Path(dir_path)

    def __call__(self, data):
        import numpy as np
        d = dict(data)
        if not self.dir:
            return d
        case_id = d.get(self.id_key)
        if isinstance(case_id, list) and case_id:
            case_id = case_id[0]
        if case_id is None:
            return d
        safe_id = str(case_id).replace('/', '_')
        seedp = self.dir / f"{safe_id}_seedmask.npy"
        supp = self.dir / f"{safe_id}_supmask.npy"
        plp = self.dir / f"{safe_id}_pseudolabel.npy"
        if seedp.exists():
            d["seed_mask"] = np.load(seedp).astype(bool)
        if supp.exists():
            d["sup_mask"] = np.load(supp).astype(bool)
        if plp.exists():
            # store as int64 for CE targets later
            d["pseudo_label"] = np.load(plp).astype(np.int64)
        # Build a valid_mask (1 within original FOV, 0 in padded region after SpatialPadd)
        # Prefer the sup_mask shape if available, else infer from label
        if "sup_mask" in d and isinstance(d["sup_mask"], np.ndarray):
            shp = d["sup_mask"].shape
            d["valid_mask"] = np.ones(shp, dtype=bool)
        else:
            label = d.get("label")
            if isinstance(label, np.ndarray):
                # assume channel-first
                d["valid_mask"] = np.ones_like(label, dtype=bool)
        return d


def get_transforms(
    roi=(112, 112, 80),
    norm: str = "clip_zscore",
    aug_intensity: bool = False,
    aug_prob: float = 0.2,
    aug_noise_std: float = 0.01,
    aug_shift: float = 0.1,
    aug_scale: float = 0.1,
    fg_crop_prob: float = 0.0,
    fg_crop_margin: int = 8,
    fewshot_mode: str | None = None,
    fewshot_static: bool = False,
    fewshot_ratio: float = 0.0,
    fp_dilate_radius: int = 1,
    fp_balance: str = "proportional",
    fp_bg_frac: float = 0.25,
    fp_max_seeds: int = -1,
    fp_seed_strategy: str = "random",
    fp_no_overlap: bool = False,
    save_sup_masks_dir: str | None = None,
):
    norm_transform = None
    if norm == "clip_zscore":
        norm_transform = ClipZScoreNormalizeD(keys=["image"])
    elif norm == "fixed_wp5":
        norm_transform = ScaleIntensityRanged(keys=["image"], a_min=-3, a_max=8.5, b_min=0.0, b_max=1.0, clip=True)
    elif norm == "none":
        norm_transform = None
    else:
        raise ValueError(f"Unknown normalization option: {norm}")

    def build_seq(include_crop: bool, include_aug: bool, training: bool):
        seq = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ]
        # If using static supervision masks (few_points or few_slices), load them before spatial transforms
        static_sup = training and fewshot_static and (save_sup_masks_dir is not None)
        if static_sup:
            seq.append(LoadSavedMasksD(keys=["label"], id_key="id", dir_path=save_sup_masks_dir))
        if norm_transform is not None:
            seq.append(norm_transform)
        # Only pad during training to support fixed-size crops.
        # For validation, avoid padding labels so evaluation is computed on true FOV only;
        # sliding_window_inference pads internally as needed and returns the original spatial size.
        if training:
            pad_keys = ["image", "label", "sup_mask", "seed_mask", "pseudo_label", "valid_mask"] if static_sup else ["image", "label"]
            # Allow optional keys to be absent (e.g., pseudo_label may not exist for sparse points)
            seq.append(SpatialPadd(keys=pad_keys, spatial_size=roi, allow_missing_keys=True))
        # Optional intensity augmentations (training only)
        if training and aug_intensity:
            seq.extend(
                [
                    RandGaussianNoised(keys=["image"], prob=aug_prob, std=aug_noise_std),
                    RandShiftIntensityd(keys=["image"], offsets=aug_shift, prob=aug_prob),
                    RandScaleIntensityd(keys=["image"], factors=aug_scale, prob=aug_prob),
                ]
            )
        if include_aug:
            flip_keys = ["image", "label", "sup_mask", "seed_mask", "pseudo_label", "valid_mask"] if static_sup else ["image", "label"]
            seq.extend([
                RandFlipd(keys=flip_keys, spatial_axis=0, prob=0.5, allow_missing_keys=True),
                RandFlipd(keys=flip_keys, spatial_axis=1, prob=0.5, allow_missing_keys=True),
                RandFlipd(keys=flip_keys, spatial_axis=2, prob=0.5, allow_missing_keys=True),
            ])
        if include_crop:
            if fg_crop_prob > 0.0 and training:
                crop_keys = ["image", "label", "sup_mask", "seed_mask", "pseudo_label", "valid_mask"] if static_sup else ["image", "label"]
                # Custom crop transform already tolerates missing keys
                seq.append(FGBiasedCropD(keys=crop_keys, roi_size=roi, prob=fg_crop_prob, margin=fg_crop_margin))
            else:
                crop_keys = ["image", "label", "sup_mask", "seed_mask", "pseudo_label", "valid_mask"] if static_sup else ["image", "label"]
                seq.append(RandSpatialCropd(keys=crop_keys, roi_size=roi, random_size=False, allow_missing_keys=True))
        return seq

    train = Compose(build_seq(include_crop=True, include_aug=True, training=True))
    val = Compose(build_seq(include_crop=False, include_aug=False, training=False))
    return train, val


def build_model(arch: str = "basicunet") -> torch.nn.Module:
    """Build segmentation network.
    Default: BasicUNet (3D) trained from scratch, which is the preferred/baseline model.
    """
    if arch == "basicunet":
        # BasicUNet default features are (32, 64, 128, 256, 512, 32)
        return BasicUNet(spatial_dims=3, in_channels=1, out_channels=5)
    elif arch == "unet":
        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=5,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


class WP5HeadWrapper(nn.Module):
    def __init__(self, backbone: nn.Module, in_channels: int, out_channels: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def build_model_from_bundle(bundle_dir: Path, out_channels: int = 5) -> torch.nn.Module:
    """Build network exactly as defined in a MONAI bundle's configs/inference.json, then
    replace the final Conv3d layer to have the desired out_channels.
    """
    config_file = bundle_dir / "configs" / "inference.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Bundle inference.json not found: {config_file}")
    parser = ConfigParser()
    parser.read_config(str(config_file))
    # pass bundle_root into parser so any @id references resolve
    parser.update(pairs={"bundle_root": str(bundle_dir)})
    # Some bundles define 'network', others 'network_def'
    try:
        net = parser.get_parsed_content("network")
    except Exception:
        net = parser.get_parsed_content("network_def")

    # Determine backbone output channels from last Conv3d by convention
    last_conv_out = None
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv3d):
            last_conv_out = module.out_channels
    if last_conv_out is None:
        raise RuntimeError("Could not determine backbone output channels (no Conv3d found).")
    print(f"Wrapping bundle network with 1x1 head: in_channels={last_conv_out} -> out_channels={out_channels}")
    return WP5HeadWrapper(net, in_channels=last_conv_out, out_channels=out_channels)


def load_pretrained_non_strict(net: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    """Load checkpoint non-strict, with helpful logging on matched/missing keys.
    Tries common key containers and strips common prefixes.
    """
    sd_raw = torch.load(ckpt_path, map_location=device)

    # Extract state dict from common containers
    candidate_keys = [
        "state_dict",
        "model_state_dict",
        "network",
        "net",
    ]
    if isinstance(sd_raw, dict):
        for k in candidate_keys:
            if k in sd_raw and isinstance(sd_raw[k], dict):
                sd = sd_raw[k]
                break
        else:
            sd = sd_raw
    else:
        sd = sd_raw

    # Try to improve key matching by adjusting common prefixes
    def strip_prefix(d: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        return {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in d.items()}

    sd_try = sd
    for prefix in ("module.", "model."):
        if any(k.startswith(prefix) for k in sd_try.keys()):
            sd_try = strip_prefix(sd_try, prefix)

    # If wrapped (e.g., WP5HeadWrapper), model keys may start with 'backbone.' or 'backbone.model.'
    net_keys = list(net.state_dict().keys())
    if net_keys:
        sample = net_keys[0]
        # Derive composite textual prefix until a numeric token appears
        tokens = sample.split('.')
        prefix_tokens = []
        for t in tokens:
            if t.isdigit():
                break
            # stop at first known param leaf
            if t in {"weight", "bias", "running_mean", "running_var"}:
                break
            prefix_tokens.append(t)
        composite_prefix = '.'.join(prefix_tokens)
        if composite_prefix and not all(k.startswith(composite_prefix + '.') for k in sd_try.keys()):
            # if sd_try already has a leading 'backbone.' but missing '.model.', try to build to composite
            sd_try = {f"{composite_prefix}." + k if not k.startswith(composite_prefix + '.') else k: v for k, v in sd_try.items()}

    # Filter out keys with mismatched shape (e.g., classification head with different out_channels)
    model_sd = net.state_dict()
    sd_filtered = {}
    dropped_shape = []
    for k, v in sd_try.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            sd_filtered[k] = v
        else:
            dropped_shape.append(k)

    if dropped_shape:
        print(f"Skipping {len(dropped_shape)} keys due to shape mismatch (first 10): {dropped_shape[:10]}")

    # Attempt load with filtered dict
    load_res = net.load_state_dict(sd_filtered, strict=False)
    missing = set(load_res.missing_keys)
    unexpected = set(load_res.unexpected_keys)
    matched_keys = set(sd_filtered.keys()) & set(model_sd.keys())
    print(
        f"Pretrained load summary: matched={len(matched_keys)}, missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if missing:
        print(f"Missing keys (first 10): {list(sorted(missing))[:10]}")
    all_unexpected = set(dropped_shape) | unexpected
    if all_unexpected:
        print(f"Unexpected/mismatched keys (first 10): {list(sorted(all_unexpected))[:10]}")


def reinitialize_weights(model: torch.nn.Module) -> None:
    """Reset parameters of all modules that define reset_parameters."""
    for m in model.modules():
        reset_fn = getattr(m, "reset_parameters", None)
        if callable(reset_fn):
            try:
                reset_fn()
            except Exception:
                pass


def dice_loss_masked(logits: torch.Tensor, target: torch.Tensor, ignore_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute soft Dice loss over classes 0..4 on voxels where ignore_mask==True (i.e., label != 6).
    logits: (B,C,X,Y,Z) with C=5, target: (B,1,X,Y,Z) int labels in [0..6], ignore_mask: (B,1,X,Y,Z) bool.
    """
    probs = F.softmax(logits, dim=1)  # (B,5,X,Y,Z)
    target_clamped = torch.clamp(target, 0, 4).long()  # (B,1,X,Y,Z)
    # build one-hot using torch.nn.functional.one_hot for explicit control
    gt_oh = F.one_hot(target_clamped.squeeze(1), num_classes=5)  # (B,X,Y,Z,5)
    gt_onehot = gt_oh.permute(0, 4, 1, 2, 3).to(probs.dtype)  # (B,5,X,Y,Z)
    mask = ignore_mask.float()  # (B,1,X,Y,Z)
    mask = mask.expand(-1, 5, -1, -1, -1)  # (B,5,X,Y,Z)
    inter = torch.sum(probs * gt_onehot * mask, dim=(0, 2, 3, 4))
    denom = torch.sum(probs * mask + gt_onehot * mask, dim=(0, 2, 3, 4))
    dice_per_class = (2 * inter + eps) / (denom + eps)
    loss = 1.0 - dice_per_class.mean()
    return loss


def compute_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    heavy: bool = True,
    hd_percentile: float = 100.0,
) -> Dict[int, Dict[str, float]]:
    """Compute per-class Dice, IoU, and optionally HD/ASD for classes 0..4; ignore class 6.

    Parameters
    - pred, gt: shaped (B,1,X,Y,Z)
    - heavy: include HD/ASD if True
    - Policy: when a class is absent in both prediction and GT for a class/case, score 1.0 (both-empty=1.0)
    """
    B = pred.shape[0]
    ignore_mask = (gt != 6)
    classes = [0, 1, 2, 3, 4]
    if heavy:
        hd_metric = HausdorffDistanceMetric(percentile=float(hd_percentile), reduction="none")
        asd_metric = SurfaceDistanceMetric(symmetric=True, reduction="none")

    out: Dict[int, Dict[str, float]] = {}
    for cls in classes:
        if cls == 0:
            pred_mask = (pred == 0)
            gt_mask = (gt == 0)
        else:
            # Correct per-class masking: compare equality to the class id
            pred_mask = (pred == cls)
            gt_mask = (gt == cls)
        pm = (pred_mask & ignore_mask).squeeze(1).cpu().numpy().astype(np.uint8)
        gm = (gt_mask & ignore_mask).squeeze(1).cpu().numpy().astype(np.uint8)
        inter = (pm & gm).sum(axis=(1, 2, 3))
        psum = pm.sum(axis=(1, 2, 3))
        gsum = gm.sum(axis=(1, 2, 3))
        uni = (pm | gm).sum(axis=(1, 2, 3))

        both_empty = (psum + gsum) == 0
        valid = ~both_empty
        dice = np.full(pred.shape[0], np.nan, dtype=np.float32)
        iou = np.full(pred.shape[0], np.nan, dtype=np.float32)
        # computed where valid
        dice[valid] = (2.0 * inter[valid]) / (psum[valid] + gsum[valid] + 1e-8)
        iou_valid = uni[valid] > 0
        iou_vals = np.zeros_like(inter[valid], dtype=np.float32)
        iou_vals[iou_valid] = inter[valid][iou_valid] / (uni[valid][iou_valid] + 1e-8)
        iou[valid] = iou_vals
        # Official policy: both-empty contribute 1.0
        dice[both_empty] = 1.0
        iou[both_empty] = 1.0

        if heavy:
            # HD & ASD handling for empties
            hd_vals = np.full(B, np.nan, dtype=np.float32)
            asd_vals = np.full(B, np.nan, dtype=np.float32)
            for b in range(B):
                # Skip both-empty and single-empty for HD/ASD aggregation
                if psum[b] == 0 and gsum[b] == 0:
                    continue
                if psum[b] == 0 or gsum[b] == 0:
                    continue
                else:
                    pt = torch.from_numpy(pm[b:b+1][None, ...].astype(np.float32))  # (1,1,...)
                    gt_t = torch.from_numpy(gm[b:b+1][None, ...].astype(np.float32))
                    try:
                        hd_vals[b] = float(np.array(hd_metric(pt, gt_t)).reshape(-1)[0])
                    except Exception:
                        hd_vals[b] = np.nan
                    try:
                        asd_vals[b] = float(np.array(asd_metric(pt, gt_t)).reshape(-1)[0])
                    except Exception:
                        asd_vals[b] = np.nan
            hd_mean = float(np.nanmean(hd_vals)) if np.any(~np.isnan(hd_vals)) else None
            asd_mean = float(np.nanmean(asd_vals)) if np.any(~np.isnan(asd_vals)) else None
        else:
            hd_mean = None
            asd_mean = None

        out[cls] = {
            "dice": float(np.nanmean(dice)) if np.any(~np.isnan(dice)) else 0.0,
            "iou": float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0,
            "hd": hd_mean,
            "asd": asd_mean,
        }

    return out


def _select_slices_mask_per_sample(lbl: torch.Tensor, axis: int, k: int) -> torch.Tensor:
    """Return mask for selected K slices along given axis for a single sample.

    Expects lbl shape (1,1,X,Y,Z). Builds a boolean mask of the same shape with True on selected slices.
    Strategy: pick top-K slices by foreground voxel count over classes 1..4 (ignoring 6).
    axis: 0->X, 1->Y, 2->Z (spatial axes)
    """
    # lbl: (1,1,X,Y,Z)
    if lbl.ndim != 5:
        raise ValueError(f"Expected 5D lbl (B=1,C=1,X,Y,Z), got shape={tuple(lbl.shape)}")

    device = lbl.device
    fg = (lbl != 0) & (lbl != 6)  # (1,1,X,Y,Z)

    # Compute per-slice counts along the requested axis
    if axis == 2:  # Z axis (dim index 4 in 5D tensor)
        counts = fg.sum(dim=(0, 1, 2, 3))  # (Z)
        axis_len = fg.shape[4]
    elif axis == 1:  # Y axis (dim index 3)
        counts = fg.sum(dim=(0, 1, 2, 4))  # (Y)
        axis_len = fg.shape[3]
    elif axis == 0:  # X axis (dim index 2)
        counts = fg.sum(dim=(0, 1, 3, 4))  # (X)
        axis_len = fg.shape[2]
    else:
        raise ValueError("axis must be 0,1,2 for X,Y,Z")

    # Determine K and indices
    k_eff = max(1, min(int(k), int(counts.numel())))
    topk = torch.topk(counts, k=k_eff, largest=True).indices.to(device)

    # Build mask
    mask = torch.zeros_like(fg, dtype=torch.bool)
    if axis == 2:  # Z
        mask[:, :, :, :, topk] = True
    elif axis == 1:  # Y
        mask[:, :, :, topk, :] = True
    else:  # X
        mask[:, :, topk, :, :] = True

    return mask


def build_slice_supervision_mask(labels: torch.Tensor, roi: Tuple[int, int, int], axis_mode: str, ratio: float, k_override: int | None) -> torch.Tensor:
    """Build a batch supervision mask for few-slices mode.
    labels: (B,1,X,Y,Z) on device. axis_mode in {'z','y','x','multi'}.
    ratio: fraction along axis; k_override if provided takes precedence.
    Returns bool mask (B,1,X,Y,Z) with True where supervised.
    """
    B, _, X, Y, Z = labels.shape
    # Determine K per axis
    def k_from_ratio(length: int) -> int:
        k = int(np.ceil(max(1e-6, ratio) * length))
        return max(1, k)

    if axis_mode == "z":
        kz = k_override if (k_override and k_override > 0) else k_from_ratio(Z)
        ks = {2: kz}
    elif axis_mode == "y":
        ky = k_override if (k_override and k_override > 0) else k_from_ratio(Y)
        ks = {1: ky}
    elif axis_mode == "x":
        kx = k_override if (k_override and k_override > 0) else k_from_ratio(X)
        ks = {0: kx}
    elif axis_mode == "multi":
        # Determine K independently per axis based on its length
        kx = k_override if (k_override and k_override > 0) else k_from_ratio(X)
        ky = k_override if (k_override and k_override > 0) else k_from_ratio(Y)
        kz = k_override if (k_override and k_override > 0) else k_from_ratio(Z)
        ks = {0: kx, 1: ky, 2: kz}
    else:
        raise ValueError("axis_mode must be one of {'x','y','z','multi'}")

    masks = []
    for b in range(B):
        m = torch.zeros_like(labels[b:b+1], dtype=torch.bool)
        for ax, kval in ks.items():
            if kval <= 0:
                continue
            m_ax = _select_slices_mask_per_sample(labels[b:b+1], axis=ax, k=kval)
            m |= m_ax
        masks.append(m)
    return torch.cat(masks, dim=0)


def build_points_supervision_mask(
    labels: torch.Tensor,
    ratio: float,
    dilate_radius: int,
    balance: str,
    max_seeds: int,
    bg_frac: float,
    seed_strategy: str,
    ensure_min_coverage: bool = False,
    max_iter: int = 5,
    no_overlap_after_dilation: bool = False,
    return_seeds: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Build supervision mask via sparse seed points and dilation.
    labels: (B,1,X,Y,Z)
    ratio: desired supervised voxel fraction per patch
    dilate_radius: 1=>3x3x3, 2=>5x5x5
    balance: 'proportional' or 'uniform' across classes 1..4
    max_seeds: cap number of sampled seeds per patch
    bg_frac: fraction of seeds allocated to background
    seed_strategy: 'random' (boundary not yet implemented -> falls back to random)
    Returns bool mask (B,1,X,Y,Z)
    """
    B, _, X, Y, Z = labels.shape
    device = labels.device
    total_voxels = X * Y * Z
    target_voxels = int(np.ceil(max(1e-6, ratio) * total_voxels))
    # Approximate seeds needed based on dilation kernel volume
    kv = (2 * dilate_radius + 1) ** 3
    seeds_target = max(1, min(int(np.ceil(target_voxels / kv)), max_seeds if max_seeds > 0 else target_voxels))

    out_masks = []
    out_seed_masks = [] if return_seeds else None
    classes = [1, 2, 3, 4]
    for b in range(B):
        lbl = labels[b, 0]  # (X,Y,Z)
        # Build per-class voxel indices
        cls_counts = []
        cls_indices = []
        for c in classes:
            idx = (lbl == c).nonzero(as_tuple=False)
            cls_indices.append(idx)
            cls_counts.append(idx.shape[0])
        bg_idx = (lbl == 0).nonzero(as_tuple=False)

        # Seeds allocation across classes
        n_bg = int(np.floor(seeds_target * np.clip(bg_frac, 0.0, 0.9)))
        n_fg_total = max(0, seeds_target - n_bg)
        per_cls = np.zeros(len(classes), dtype=int)
        if n_fg_total > 0:
            if balance == "uniform":
                base = n_fg_total // len(classes)
                rem = n_fg_total % len(classes)
                per_cls[:] = base
                per_cls[:rem] += 1
            else:  # proportional
                counts = np.array(cls_counts, dtype=float)
                s = counts.sum()
                if s > 0:
                    per_cls = np.floor(n_fg_total * (counts / s)).astype(int)
                    # Distribute remainder
                    rem = n_fg_total - per_cls.sum()
                    order = np.argsort(-counts)
                    for i in range(rem):
                        per_cls[order[i % len(order)]] += 1
                else:
                    # No FG present; push all to background
                    n_bg = seeds_target
                    per_cls[:] = 0

        # Sample seeds (we may add more later if ensure_min_coverage)
        seed_mask = torch.zeros_like(lbl, dtype=torch.bool)
        blocked = torch.zeros_like(lbl, dtype=torch.bool) if no_overlap_after_dilation else None
        g = torch.Generator(device=device)
        g.manual_seed(torch.seed())

        # Foreground classes
        for c_i, idx in enumerate(cls_indices):
            n = int(per_cls[c_i])
            if n <= 0 or idx.shape[0] == 0:
                continue
            if no_overlap_after_dilation:
                # filter indices to those not within blocked
                mask_ok = ~blocked[idx[:, 0], idx[:, 1], idx[:, 2]]
                idx = idx[mask_ok]
                if idx.shape[0] == 0:
                    continue
            # random sampling of rows from idx
            perm = torch.randperm(idx.shape[0], generator=g, device=device)[: min(n, idx.shape[0])]
            sel = idx[perm]
            seed_mask[sel[:, 0], sel[:, 1], sel[:, 2]] = True
            if no_overlap_after_dilation:
                # Update blocked region by dilating just-added seeds
                added = torch.zeros_like(lbl, dtype=torch.bool)
                added[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                added_t = added.unsqueeze(0).unsqueeze(0).float()
                k = 2 * dilate_radius + 1
                dil = F.max_pool3d(added_t, kernel_size=k, stride=1, padding=dilate_radius) > 0.5
                blocked |= dil[0, 0]

        # Background
        if n_bg > 0 and bg_idx.shape[0] > 0:
            idx = bg_idx
            if no_overlap_after_dilation:
                mask_ok = ~blocked[idx[:, 0], idx[:, 1], idx[:, 2]]
                idx = idx[mask_ok]
            if idx.shape[0] > 0:
                perm = torch.randperm(idx.shape[0], generator=g, device=device)[: min(n_bg, idx.shape[0])]
                sel = idx[perm]
                seed_mask[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                if no_overlap_after_dilation:
                    added = torch.zeros_like(lbl, dtype=torch.bool)
                    added[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                    added_t = added.unsqueeze(0).unsqueeze(0).float()
                    k = 2 * dilate_radius + 1
                    dil = F.max_pool3d(added_t, kernel_size=k, stride=1, padding=dilate_radius) > 0.5
                    blocked |= dil[0, 0]

        # Dilate via max_pool3d trick
        seed_mask_t = seed_mask.unsqueeze(0).unsqueeze(0).float()  # (1,1,X,Y,Z)
        k = 2 * dilate_radius + 1
        dilated = F.max_pool3d(seed_mask_t, kernel_size=k, stride=1, padding=dilate_radius)
        sup = dilated > 0.5

        if ensure_min_coverage:
            # Iteratively add more seeds in uncovered regions to approach target coverage
            target_cov = float(max(1e-6, ratio))
            it = 0
            while it < max_iter:
                cur_cov = float(sup.float().mean().item())
                if cur_cov >= target_cov:
                    break
                need_voxels = int(np.ceil(target_cov * total_voxels - float(sup.sum().item())))
                if need_voxels <= 0:
                    break
                add_seeds = max(1, int(np.ceil(need_voxels / kv)))
                # Allocate additional seeds across classes using remaining eligible voxels
                add_bg = int(np.floor(add_seeds * np.clip(bg_frac, 0.0, 0.9)))
                add_fg_total = max(0, add_seeds - add_bg)
                add_per_cls = np.zeros(len(classes), dtype=int)
                if add_fg_total > 0:
                    # Eligible counts per class (centers not already supervised)
                    elig_counts = []
                    elig_indices = []
                    sup3 = sup[0, 0]
                    for idx in cls_indices:
                        if idx.shape[0] == 0:
                            elig_indices.append(idx)
                            elig_counts.append(0)
                            continue
                        mask = ~sup3[idx[:, 0], idx[:, 1], idx[:, 2]]
                        if no_overlap_after_dilation and blocked is not None:
                            mask &= ~blocked[idx[:, 0], idx[:, 1], idx[:, 2]]
                        elig = idx[mask]
                        elig_indices.append(elig)
                        elig_counts.append(elig.shape[0])
                    counts = np.array(elig_counts, dtype=float)
                    s = counts.sum()
                    if s > 0:
                        add_per_cls = np.floor(add_fg_total * (counts / s)).astype(int)
                        rem = add_fg_total - add_per_cls.sum()
                        order = np.argsort(-counts)
                        for i in range(rem):
                            add_per_cls[order[i % len(order)]] += 1
                    else:
                        add_bg = add_seeds
                        add_per_cls[:] = 0
                # Sample additional FG seeds
                for c_i, elig in enumerate(elig_indices if add_fg_total > 0 else []):
                    n = int(add_per_cls[c_i])
                    if n <= 0 or elig.shape[0] == 0:
                        continue
                    perm = torch.randperm(elig.shape[0], generator=g, device=device)[: min(n, elig.shape[0])]
                    sel = elig[perm]
                    seed_mask[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                    if no_overlap_after_dilation:
                        added = torch.zeros_like(lbl, dtype=torch.bool)
                        added[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                        added_t = added.unsqueeze(0).unsqueeze(0).float()
                        k = 2 * dilate_radius + 1
                        dil = F.max_pool3d(added_t, kernel_size=k, stride=1, padding=dilate_radius) > 0.5
                        blocked |= dil[0, 0]
                # Additional BG seeds from uncovered background
                if add_bg > 0 and bg_idx.shape[0] > 0:
                    sup3 = sup[0, 0]
                    mask = ~sup3[bg_idx[:, 0], bg_idx[:, 1], bg_idx[:, 2]]
                    if no_overlap_after_dilation and blocked is not None:
                        mask &= ~blocked[bg_idx[:, 0], bg_idx[:, 1], bg_idx[:, 2]]
                    elig_bg = bg_idx[mask]
                    if elig_bg.shape[0] > 0:
                        perm = torch.randperm(elig_bg.shape[0], generator=g, device=device)[: min(add_bg, elig_bg.shape[0])]
                        sel = elig_bg[perm]
                        seed_mask[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                        if no_overlap_after_dilation:
                            added = torch.zeros_like(lbl, dtype=torch.bool)
                            added[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                            added_t = added.unsqueeze(0).unsqueeze(0).float()
                            k = 2 * dilate_radius + 1
                            dil = F.max_pool3d(added_t, kernel_size=k, stride=1, padding=dilate_radius) > 0.5
                            blocked |= dil[0, 0]
                # Recompute sup after adding seeds
                seed_mask_t = seed_mask.unsqueeze(0).unsqueeze(0).float()
                dilated = F.max_pool3d(seed_mask_t, kernel_size=k, stride=1, padding=dilate_radius)
                sup = dilated > 0.5
                it += 1
        out_masks.append(sup.bool())
        if return_seeds:
            out_seed_masks.append(seed_mask.unsqueeze(0).unsqueeze(0))

    sup_out = torch.cat(out_masks, dim=0)
    if return_seeds:
        seed_out = torch.cat(out_seed_masks, dim=0)
        return sup_out, seed_out
    return sup_out


def precompute_sup_masks_from_selected_points(
    train_list: List[Dict],
    selected_points_dir: Path,
    pseudo_label_dir: Path | None,
    out_dir: Path,
    dilate_radius: int = 1,
) -> None:
    """Precompute static few_points masks from preselected points and optional dense pseudo labels.

    Expects per-case files:
      - <selected_points_dir>/<id>/mask_selected_points.nii
      - optional pseudo label at <pseudo_label_dir>/<basename_label_file>

    Saves per-case:
      - <out_dir>/<id>_seedmask.npy (1,X,Y,Z) bool
      - <out_dir>/<id>_supmask.npy (1,X,Y,Z) bool
      - <out_dir>/<id>_pseudolabel.npy (1,X,Y,Z) int64 if provided
      - <out_dir>/<id>_supmask_stats.json summary
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("nibabel is required to load NIfTI selected points/pseudo labels") from e

    import torch
    import numpy as _np
    k = 2 * int(dilate_radius) + 1

    for rec in train_list:
        cid = rec.get("id") or Path(rec["label"]).stem.replace("_label", "")
        pts_path = selected_points_dir / str(cid) / "mask_selected_points.nii"
        if not pts_path.exists():
            print(f"WARNING: selected points mask not found for {cid}: {pts_path}")
            continue
        pts_img = nib.load(str(pts_path))
        pts = pts_img.get_fdata().astype(_np.int64)
        if pts.ndim != 3:
            raise RuntimeError(f"Selected points mask must be 3D: {pts_path}")
        seed = (pts > 0)
        X, Y, Z = seed.shape
        seed_t = torch.from_numpy(seed[None, None].astype(_np.float32))
        sup = torch.nn.functional.max_pool3d(seed_t, kernel_size=k, stride=1, padding=int(dilate_radius)) > 0.5
        # Save masks
        safe_id = str(cid).replace('/', '_')
        _np.save(out_dir / f"{safe_id}_seedmask.npy", seed[None, ...])
        _np.save(out_dir / f"{safe_id}_supmask.npy", sup[0, 0].cpu().numpy()[None, ...])
        stats = {
            "id": cid,
            "shape": [1, X, Y, Z],
            "seed_fraction": float(seed.mean()),
            "sup_fraction": float(sup[0, 0].float().mean().item()),
            "radius": int(dilate_radius),
            "source_points": str(pts_path),
        }
        (out_dir / f"{safe_id}_supmask_stats.json").write_text(json.dumps(stats, indent=2))
        # Optional dense pseudo label
        if pseudo_label_dir is not None:
            lbl_base = Path(rec["label"]).name
            pl_path = pseudo_label_dir / lbl_base
            if pl_path.exists():
                pl_img = nib.load(str(pl_path))
                pl = pl_img.get_fdata().astype(_np.int64)
                if pl.ndim != 3:
                    raise RuntimeError(f"Pseudo label must be 3D: {pl_path}")
                _np.save(out_dir / f"{safe_id}_pseudolabel.npy", pl[None, ...])
            else:
                print(f"INFO: pseudo label not found for {cid} at {pl_path}; skipping pseudolabel save")


def precompute_sup_masks_from_selected_slices(
    train_list: List[Dict],
    slice_sel_json: Path,
    out_dir: Path,
) -> None:
    """Precompute static sup masks from a selected_slices.json for few_slices mode.

    Each entry in JSON must have: {"id": str, "axis": "x"|"y"|"z", "index": int}.
    Saves <out_dir>/<id>_supmask.npy with shape (1,X,Y,Z) bool in RAS orientation.
    """
    entries = json.loads(slice_sel_json.read_text())
    by_id: Dict[str, Dict[str, set]] = {}
    for e in entries:
        cid = e["id"]
        ax = str(e["axis"]).lower()
        idx = int(e["index"])
        by_id.setdefault(cid, {}).setdefault(ax, set()).add(idx)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in train_list:
        cid = rec["id"]
        # Load label to get oriented shape
        arr = _load_label_volume(rec["label"])  # (1,X,Y,Z)
        _, X, Y, Z = arr.shape
        sup = np.zeros_like(arr, dtype=bool)
        axes_sel = by_id.get(cid, {})
        xs = axes_sel.get('x', set())
        ys = axes_sel.get('y', set())
        zs = axes_sel.get('z', set())
        for i in xs:
            if 0 <= i < X:
                sup[0, i, :, :] = True
        for i in ys:
            if 0 <= i < Y:
                sup[0, :, i, :] = True
        for i in zs:
            if 0 <= i < Z:
                sup[0, :, :, i] = True
        np.save(out_dir / f"{cid.replace('/', '_')}_supmask.npy", sup)


def evaluate(
    net: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    out_dir: Path,
    save_preds: bool = False,
    max_cases: int | None = None,
    heavy: bool = True,
    hd_percentile: float = 100.0,
) -> Dict[str, Dict]:
    net.eval()
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "preds"
    if save_preds:
        pred_dir.mkdir(parents=True, exist_ok=True)

    # init writer if possible
    try:
        import nibabel as nib  # noqa: F401
        saver = SaveImage(
            output_dir=str(pred_dir),
            output_postfix="pred",
            output_ext=".nii.gz",
            output_dtype=np.uint8,
            resample=False,
            mode="nearest",
            separate_folder=False,
            print_log=False,
        ) if save_preds else None
    except Exception:
        saver = None

    roi = (112, 112, 80)
    classes = [0, 1, 2, 3, 4]
    sums = {c: {"dice": 0.0, "iou": 0.0, "hd": 0.0, "asd": 0.0, "n": 0} for c in classes}

    with torch.no_grad():
        for i, batch in enumerate(dl):
            if max_cases is not None and i >= max_cases:
                break
            img = batch["image"].to(device)
            gt = batch["label"].to(device)
            logits = sliding_window_inference(
                img,
                roi_size=roi,
                sw_batch_size=1,
                predictor=net,
                sw_device=device,
                device=device,
            )
            pred = torch.argmax(logits, dim=1, keepdim=True)

            if save_preds and saver is not None:
                meta_list = batch.get("image_meta_dict", None)
                if meta_list is not None:
                    for b in range(pred.shape[0]):
                        saver(pred[b].cpu(), meta_list[b] if isinstance(meta_list, list) else meta_list)
                else:
                    # fallback
                    id_field = batch.get("id", None)
                    for b in range(pred.shape[0]):
                        base = id_field[b] if isinstance(id_field, list) else id_field
                        if base is None:
                            base = f"case_{i}_{b}"
                        np.save(pred_dir / f"{base}_pred.npy", pred[b].cpu().numpy())

            per_class = compute_metrics(
                pred.cpu(),
                gt.cpu(),
                heavy=heavy,
                hd_percentile=hd_percentile,
            )
            for c in classes:
                sums[c]["dice"] += per_class[c]["dice"]
                sums[c]["iou"] += per_class[c]["iou"]
                if heavy and per_class[c]["hd"] is not None:
                    sums[c]["hd"] += per_class[c]["hd"]
                if heavy and per_class[c]["asd"] is not None:
                    sums[c]["asd"] += per_class[c]["asd"]
                sums[c]["n"] += 1

    # finalize
    summary = {}
    for c in classes:
        entry = {
            "dice": sums[c]["dice"] / max(sums[c]["n"], 1),
            "iou": sums[c]["iou"] / max(sums[c]["n"], 1),
        }
        if heavy:
            entry["hd"] = sums[c]["hd"] / max(sums[c]["n"], 1)
            entry["asd"] = sums[c]["asd"] / max(sums[c]["n"], 1)
        else:
            entry["hd"] = None
            entry["asd"] = None
        summary[str(c)] = entry

    avg = {
        "dice": float(np.mean([summary[str(c)]["dice"] for c in classes])),
        "iou": float(np.mean([summary[str(c)]["iou"] for c in classes])),
        "hd": float(np.mean([summary[str(c)]["hd"] for c in classes if summary[str(c)]["hd"] is not None])) if heavy else None,
        "asd": float(np.mean([summary[str(c)]["asd"] for c in classes if summary[str(c)]["asd"] is not None])) if heavy else None,
    }
    meta = {
        "empty_pair_policy": "count_as_one",
        "heavy": bool(heavy),
        "hd_percentile": float(hd_percentile),
        "classes": [0, 1, 2, 3, 4],
        "ignore_label": 6,
    }
    payload = {"per_class": summary, "average": avg, "meta": meta}
    # save epoch metrics to JSON
    (metrics_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    return payload


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    data_root = Path(args.data_root)
    split_cfg = Path(args.split_cfg)
    out_dir = Path(args.output_dir)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    # initialize run logging once outputs are resolved
    _init_run_logging(out_dir=str(out_dir), enable=bool(getattr(args, "log_to_file", True)), filename=str(getattr(args, "log_file_name", "train.log")))
    # persist args for reproducibility
    try:
        (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))
    except Exception:
        pass

    # Build datalists and optionally subset train
    train_list, test_list = build_datalists(data_root / "data", split_cfg)
    train_list = subset_datalist(train_list, args.subset_ratio, args.seed)

    # Transforms and datasets
    # Where to save/load static supervision masks
    if getattr(args, "sup_masks_dir", ""):
        save_masks_dir = args.sup_masks_dir
        print(f"Using precomputed supervision masks from: {save_masks_dir}")
    else:
        save_masks_dir = str(out_dir / "sup_masks") if (getattr(args, "save_sup_masks", False) or args.fewshot_static) else None

    # Precompute static masks for few_points or few_slices mode as requested
    if args.fewshot_mode == "few_points" and args.fewshot_static and not getattr(args, "sup_masks_dir", ""):
        Path(save_masks_dir).mkdir(parents=True, exist_ok=True)
        if getattr(args, "selected_points_dir", ""):
            print("Precomputing static few-shot masks from selected points (raw or dilated seeds)...")
            pl_dir = Path(args.pseudo_label_dir) if getattr(args, "pseudo_label_dir", "") else None
            precompute_sup_masks_from_selected_points(
                train_list=train_list,
                selected_points_dir=Path(args.selected_points_dir),
                pseudo_label_dir=pl_dir,
                out_dir=Path(save_masks_dir),
                dilate_radius=args.fp_dilate_radius,
            )
        elif getattr(args, "global_budget", True):
            # Avoid accidental overwrite and invalid ratios
            target_dir = Path(save_masks_dir)
            existing = target_dir.exists() and any(target_dir.glob("*_seedmask.npy"))
            if args.fewshot_ratio <= 0.0:
                print("WARNING: --fewshot_ratio is 0.0; skipping precompute of static masks.")
            elif existing and not getattr(args, "force_recompute_sup", False):
                print(f"Found existing supervision masks under {target_dir}; skipping recompute (use --force_recompute_sup to overwrite).")
            else:
                print(
                    "Precomputing static few-shot masks with global seed budget across train set... "
                    f"ratio={args.fewshot_ratio}, seed_bg_frac={args.seed_bg_frac}, radius={args.fp_dilate_radius}, balance={args.fp_balance}"
                )
                precompute_static_global_seed_masks(
                    train_list=train_list,
                    masks_dir=target_dir,
                    ratio=args.fewshot_ratio,
                    seed_bg_frac=args.seed_bg_frac,
                    dilate_radius=args.fp_dilate_radius,
                    balance=args.fp_balance,
                    no_overlap=args.fp_no_overlap,
                    dilation_shape=args.dilation_shape,
                    seed=args.seed,
                    sample_mode=getattr(args, "fp_sample_mode", "stratified"),
                    uniform_exclude6=getattr(args, "fp_uniform_exclude6", False),
                )
    elif args.fewshot_mode == "few_slices" and args.fewshot_static and getattr(args, "slice_sel_json", "") and not getattr(args, "sup_masks_dir", ""):
        # If a selection JSON is provided and no external sup_masks_dir is set, precompute masks into <out_dir>/sup_masks
        Path(save_masks_dir).mkdir(parents=True, exist_ok=True)
        print("Precomputing static sup masks for few_slices from selection JSON...")
        precompute_sup_masks_from_selected_slices(
            train_list=train_list,
            slice_sel_json=Path(args.slice_sel_json),
            out_dir=Path(save_masks_dir),
        )

    t_train, t_val = get_transforms(
        roi=(args.roi_x, args.roi_y, args.roi_z),
        norm=args.norm,
        aug_intensity=args.aug_intensity,
        aug_prob=args.aug_prob,
        aug_noise_std=args.aug_noise_std,
        aug_shift=args.aug_shift,
        aug_scale=args.aug_scale,
        fg_crop_prob=args.fg_crop_prob,
        fg_crop_margin=args.fg_crop_margin,
        fewshot_mode=args.fewshot_mode,
        fewshot_static=args.fewshot_static,
        fewshot_ratio=args.fewshot_ratio,
        fp_dilate_radius=args.fp_dilate_radius,
        fp_balance=args.fp_balance,
        fp_bg_frac=args.fp_bg_frac,
        fp_max_seeds=args.fp_max_seeds,
        fp_seed_strategy=args.fp_seed_strategy,
        fp_no_overlap=args.fp_no_overlap,
        save_sup_masks_dir=save_masks_dir,
    )
    ds_train = Dataset(train_list, transform=t_train)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    ds_test = Dataset(test_list, transform=t_val)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Model (require bundle architecture if provided)
    if args.bundle_dir:
        bundle_path = Path(args.bundle_dir)
        if not bundle_path.exists():
            raise FileNotFoundError(f"--bundle_dir provided but not found: {bundle_path}")
        net = build_model_from_bundle(bundle_path, out_channels=5).to(device)
        print(f"Built network from bundle: {args.bundle_dir}")
    else:
        net = build_model(args.net).to(device)
    # Initialization policy (clear and unambiguous)
    if args.init == "pretrained":
        if args.pretrained_ckpt and Path(args.pretrained_ckpt).exists():
            print(f"Initializing from pretrained checkpoint: {args.pretrained_ckpt}")
            load_pretrained_non_strict(net, Path(args.pretrained_ckpt), device)
        else:
            print(
                f"WARNING: --init pretrained requested but --pretrained_ckpt not found: {args.pretrained_ckpt}. Proceeding with random initialization."
            )
            reinitialize_weights(net)
    else:
        if args.pretrained_ckpt:
            print("INFO: --init scratch set; ignoring --pretrained_ckpt.")
        print("Initializing model from scratch.")
        reinitialize_weights(net)

    # Use bundle-aligned LRs by default (higher than before)
    base_lr = args.lr_ft if args.init == "pretrained" else args.lr
    if Novograd is not None:
        optimizer = Novograd(net.parameters(), lr=base_lr)
        opt_name = "Novograd"
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)
        opt_name = "Adam"
    # Epoch-level MultiStepLR similar to bundle behavior (decays at 60% and 85%)
    milestones = [max(1, int(0.6 * args.epochs)), max(2, int(0.85 * args.epochs))]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    best_dice = -1.0
    roi = (args.roi_x, args.roi_y, args.roi_z)

    # Log basic run config once
    # Report parameter count and config
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"Run config: init={args.init}, net={'bundle' if args.bundle_dir else args.net}, params={n_params/1e6:.2f}M, "
        f"optimizer={opt_name}, base_lr={base_lr:.2e}, epochs={args.epochs}, milestones={milestones}"
    )

    for epoch in range(1, args.epochs + 1):
        net.train()
        epoch_loss = 0.0
        n_batches = 0
        cov_sum = 0.0
        cov_sum_seeds = 0.0
        cov_sum_sup = 0.0
        pl_cov_sum = 0.0
        t0 = time.time()
        # global counter of debug grad checks across training
        if not hasattr(train, "_grad_checks_done"):
            train._grad_checks_done = 0  # type: ignore[attr-defined]
        for batch in dl_train:
            img = batch["image"].to(device)
            lbl = batch["label"].long().to(device)  # (B,1,...)
            ignore_mask = (lbl != 6)

            # Supervision masks
            if args.fewshot_mode == "few_points":
                if ("seed_mask" not in batch) or ("sup_mask" not in batch):
                    raise RuntimeError("few_points static mode requires precomputed 'seed_mask' and 'sup_mask' in batch.")
                seed_mask = batch["seed_mask"].to(device).bool()
                sup_mask = batch["sup_mask"].to(device).bool()
                pseudo_label = batch.get("pseudo_label", None)
                if pseudo_label is not None:
                    pseudo_label = pseudo_label.to(device).squeeze(1)
            elif args.fewshot_mode == "few_slices":
                # If static sup masks are provided via transforms, prefer them
                if args.fewshot_static and ("sup_mask" in batch):
                    sup_mask = batch["sup_mask"].to(device).bool()
                    seed_mask = sup_mask
                    pseudo_label = None
                else:
                    k_override = args.fs_k_slices if args.fs_k_slices > 0 else None
                    sup_mask = build_slice_supervision_mask(
                        labels=lbl, roi=(args.roi_x, args.roi_y, args.roi_z), axis_mode=args.fs_axis_mode,
                        ratio=args.fewshot_ratio, k_override=k_override,
                    )
                    seed_mask = sup_mask
                    pseudo_label = None
            else:
                sup_mask = torch.ones_like(lbl, dtype=torch.bool, device=device)
                seed_mask = sup_mask
                pseudo_label = None

            if getattr(args, "force_no_supervision", False):
                seed_mask = torch.zeros_like(lbl, dtype=torch.bool, device=device)
                sup_mask = torch.zeros_like(lbl, dtype=torch.bool, device=device)

            # Coverage logging
            with torch.no_grad():
                if args.fewshot_mode == "few_points":
                    vm = batch.get("valid_mask", None)
                    if vm is not None:
                        vm_t = vm.to(device).bool()
                        # Optionally exclude class 6 from denominator for uniform_all+exclude6 semantics
                        if getattr(args, "fp_sample_mode", "stratified") == "uniform_all" and getattr(args, "fp_uniform_exclude6", False):
                            vm_t = vm_t & (lbl != 6)
                        denom = float(vm_t.sum().item()) if vm_t.numel() > 0 else 0.0
                        if denom > 0:
                            cov_seeds = float(((seed_mask & vm_t).float().sum().item()) / denom)
                            cov_sup = float(((sup_mask & vm_t).float().sum().item()) / denom)
                        else:
                            cov_seeds = float(seed_mask.float().mean().item())
                            cov_sup = float(sup_mask.float().mean().item())
                    else:
                        cov_seeds = float(seed_mask.float().mean().item())
                        cov_sup = float(sup_mask.float().mean().item())
                    cov_sum_seeds += cov_seeds
                    cov_sum_sup += cov_sup
                    cov = cov_seeds if getattr(args, "coverage_mode", "sup") == "seeds" else cov_sup
                else:
                    cov = float(sup_mask.float().mean().item())
                    cov_sum_seeds += cov
                    cov_sum_sup += cov
                cov_sum += cov

            logits = net(img)
            # retain gradient on logits if we plan to inspect it
            if getattr(args, "debug_grad_check", False):
                logits.retain_grad()
            # Seed-only GT supervision
            ce_target_seed = lbl.squeeze(1).clone()
            seed_mask_s = seed_mask.squeeze(1)
            ce_target_seed[(~seed_mask_s) | (ce_target_seed == 6)] = 255
            supervised_count = int((ce_target_seed != 255).sum().item())
            ce_sup = torch.tensor(0.0, device=device)
            if supervised_count > 0:
                ce_sup = F.cross_entropy(logits, ce_target_seed, ignore_index=255)

            # Pseudo supervision on dilated region using propagated labels (lower weight)
            ce_pseudo = torch.tensor(0.0, device=device)
            pl_count = 0
            if args.fewshot_mode == "few_points" and pseudo_label is not None:
                dil_only = (sup_mask.squeeze(1) & (~seed_mask_s))
                if dil_only.any():
                    pl_target = pseudo_label.clone()
                    pl_target[~dil_only] = 255
                    pl_count = int((pl_target != 255).sum().item())
                    if pl_count > 0:
                        ce_pseudo = F.cross_entropy(logits, pl_target, ignore_index=255)

            # Optional model-based PL on remaining unlabeled voxels
            ce_pl = torch.tensor(0.0, device=device)
            if args.pl_enable and epoch > args.pl_warmup_epochs:
                with torch.no_grad():
                    probs = F.softmax(logits, dim=1)
                    pmax, pcls = probs.max(dim=1)
                    unlabeled = (~sup_mask.squeeze(1)) & ignore_mask.squeeze(1)
                    pmask = unlabeled & (pmax >= args.pl_threshold)
                    pl_cov = float(pmask.float().mean().item())
                    pl_cov_sum += pl_cov
                if pmask.any():
                    ce_pl_map = F.cross_entropy(logits, pcls, reduction='none')
                    ce_pl = (ce_pl_map * pmask.float()).sum() / pmask.float().sum()

            # Combine losses: seed GT + pseudo on dilation + optional PL
            ce = ce_sup + args.pseudo_weight * ce_pseudo + args.pl_weight * ce_pl
            if supervised_count == 0:
                dice = torch.tensor(0.0, device=device)
            else:
                dice = dice_loss_masked(logits, lbl, ignore_mask & seed_mask)
            loss = 0.5 * ce + 0.5 * dice
            if supervised_count == 0 and pl_count == 0:
                loss = loss + 0.0 * logits.sum()

            optimizer.zero_grad()
            loss.backward()

            # Optional gradient inspection: verify zero grads outside active supervision
            if getattr(args, "debug_grad_check", False):
                # Limit how many batches to print across the whole training run
                if train._grad_checks_done < max(1, int(getattr(args, "debug_grad_batches", 1))):  # type: ignore[attr-defined]
                    g = logits.grad.detach()
                    B, C = g.shape[:2]
                    # Active supervision mask over voxels (B,1,X,Y,Z), excluding label==6
                    active = torch.zeros_like(lbl, dtype=torch.bool)
                    # seed CE and Dice supervise seed_mask where label!=6
                    active |= (seed_mask & ignore_mask)
                    # dilated-only region for pseudo supervision
                    try:
                        dil_only = (sup_mask.squeeze(1) & (~seed_mask.squeeze(1)))
                    except Exception:
                        dil_only = torch.zeros_like(seed_mask.squeeze(1), dtype=torch.bool)
                    if pseudo_label is not None:
                        active |= dil_only.unsqueeze(1)
                    # model-based PL mask if enabled
                    pmask = None
                    if args.pl_enable and epoch > args.pl_warmup_epochs:
                        with torch.no_grad():
                            probs_dbg = torch.softmax(logits, dim=1)
                            pmax_dbg, _ = probs_dbg.max(dim=1)
                            pmask = ((~sup_mask.squeeze(1)) & ignore_mask.squeeze(1) & (pmax_dbg >= args.pl_threshold))
                    if pmask is not None:
                        active |= pmask.unsqueeze(1)
                    # Exclude class 6 voxels from consideration entirely
                    valid = ignore_mask
                    # Expand masks to channel dimension to compare with g
                    active_c = active.expand(-1, C, -1, -1, -1)
                    valid_c = valid.expand(-1, C, -1, -1, -1)
                    unsup_c = valid_c & (~active_c)
                    sup_c = active_c
                    eps = float(getattr(args, "debug_grad_eps", 1e-10))
                    unsup_abs = g[unsup_c].abs()
                    sup_abs = g[sup_c].abs()
                    nz_unsup = int((unsup_abs > eps).sum().item())
                    max_unsup = float(unsup_abs.max().item()) if unsup_abs.numel() else 0.0
                    mean_unsup = float(unsup_abs.mean().item()) if unsup_abs.numel() else 0.0
                    mean_sup = float(sup_abs.mean().item()) if sup_abs.numel() else 0.0
                    total_unsup = int(unsup_abs.numel())
                    total_sup = int(sup_abs.numel())
                    status = "OK" if nz_unsup == 0 else "WARN"
                    print(
                        f"[grad-check] epoch {epoch} batch {n_batches+1}: {status} || "
                        f"unsup_nz={nz_unsup}/{total_unsup}, max_unsup={max_unsup:.3e}, "
                        f"mean_unsup={mean_unsup:.3e}, mean_sup={mean_sup:.3e}"
                    )
                    # Optional: print top-K offending unsupervised locations
                    topk = int(getattr(args, "debug_grad_topk", 0))
                    if topk > 0 and nz_unsup > 0:
                        k = min(topk, unsup_abs.numel())
                        vals, idxs = torch.topk(unsup_abs, k=k, largest=True)
                        # Reconstruct indices into (B,C,X,Y,Z) among unsupervised entries
                        # For readability, we only print first few
                        print("[grad-check] top unsup grads:")
                        # Build flat mapping from unsup mask to indices
                        flat_idx = idxs.cpu().numpy().tolist()
                        vals_np = vals.cpu().numpy().tolist()
                        # Traverse to find coordinates (costly if huge; keep k small)
                        # We iterate over tensor positions to find matching linear order within unsup mask
                        # For performance, we compute direct unravel on a compacted view
                        # Create a view where unsup entries are compacted contiguously
                        # This step keeps it simple without heavy coordinate reporting
                        for i, v in enumerate(vals_np):
                            if i >= k:
                                break
                            print(f"    {i+1}: |grad|={v:.3e}")
                    train._grad_checks_done += 1  # type: ignore[attr-defined]
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        dur = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        cov_epoch = cov_sum / max(n_batches, 1)
        cov_seeds_epoch = cov_sum_seeds / max(n_batches, 1)
        cov_sup_epoch = cov_sum_sup / max(n_batches, 1)
        plcov_epoch = pl_cov_sum / max(n_batches, 1) if args.pl_enable else 0.0
        # Always print both seed and sup coverage means if using few_points
        extra_cov = ""
        if args.fewshot_mode == "few_points":
            # Show higher precision (scientific if needed) to avoid 0.0000 confusion
            extra_cov = f" - cov_seeds_mean {cov_seeds_epoch:.6g} - cov_sup_mean {cov_sup_epoch:.6g}"
        print(
            f"Epoch {epoch}/{args.epochs} - loss {avg_loss:.4f} - {dur:.1f}s - lr {optimizer.param_groups[0]['lr']:.2e}"
            + f" - avg_cov({getattr(args, 'coverage_mode', 'sup')}) {cov_epoch:.6g}" + extra_cov
            + (f" - pl_coverage {plcov_epoch:.6g}" if args.pl_enable else "")
        )

        # Evaluate on test set (fast: skip HD/ASD to keep GPU utilization higher)
        metrics = evaluate(
            net,
            dl_test,
            device,
            out_dir,
            save_preds=False,
            max_cases=None,
            heavy=False,
            hd_percentile=getattr(args, "hd_percentile", 100.0),
        )
        avg_dice = metrics["average"]["dice"]
        # Save epoch metrics
        epoch_metrics_path = out_dir / "metrics" / f"epoch_{epoch:03d}.json"
        epoch_metrics_path.write_text(json.dumps(metrics, indent=2))
        # Pretty per-class logging
        pc = metrics["per_class"]
        dice_parts = [f"overall {metrics['average']['dice']:.6f}"] + [
            f"cls {c}: {pc[str(c)]['dice']:.6f}" for c in [0, 1, 2, 3, 4]
        ]
        iou_parts = [f"overall {metrics['average']['iou']:.6f}"] + [
            f"cls {c}: {pc[str(c)]['iou']:.6f}" for c in [0, 1, 2, 3, 4]
        ]
        print(
            f"Epoch {epoch} {args.init} test avg (0..4): "
            + "{"
            + f"'dice': {', '.join(dice_parts)}, "
            + f"'iou': {', '.join(iou_parts)}, "
            + f"'hd': {metrics['average']['hd']}, 'asd': {metrics['average']['asd']}"
            + "}"
        )

        # Save best checkpoint
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(net.state_dict(), out_dir / "best.ckpt")
        # Save last checkpoint
        torch.save(net.state_dict(), out_dir / "last.ckpt")
        # Step the scheduler
        scheduler.step()

    print(f"Training complete. Best avg Dice (0..4): {best_dice:.4f}")


def _init_run_logging(out_dir: str, enable: bool = True, filename: str = "train.log") -> None:
    """Set up a simple tee for stdout/stderr to a log file.

    This catches print() output without requiring shell `tee`.
    """
    if not enable:
        return
    path = Path(out_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a", buffering=1, encoding="utf-8")

    class Tee:
        def __init__(self, stream, mirror):
            self.stream = stream
            self.mirror = mirror

        def write(self, s):
            self.stream.write(s)
            self.mirror.write(s)
            return len(s)

        def flush(self):
            self.stream.flush()
            self.mirror.flush()

    import io as _io
    sys.stdout = Tee(sys.__stdout__, fh)  # type: ignore[assignment]
    sys.stderr = Tee(sys.__stderr__, fh)  # type: ignore[assignment]


def parse_args(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="WP5 Fine-tuning (use scripts/eval_wp5.py for evaluation)")
    p.add_argument("--mode", choices=["train"], default="train", help="Run mode (only 'train' supported)")
    p.add_argument("--data_root", type=str, default="/data3/wp5/wp5-code/dataloaders/wp5-dataset", help="WP5 dataset root (contains data/)")
    p.add_argument("--split_cfg", type=str, default="/data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json", help="Predefined split config JSON")
    p.add_argument("--output_dir", type=str, default="runs/wp5_finetune", help="Output directory (base path)")
    p.add_argument(
        "--train_label_override_dir",
        type=str,
        default="",
        help="Override training label paths with files from this directory (mirror basenames)",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate for scratch training (Novograd/Adam)")
    p.add_argument("--lr_ft", type=float, default=3e-4, help="Learning rate for finetuning when --init pretrained (Novograd/Adam)")
    p.add_argument("--subset_ratio", type=float, default=1.0, help="Proportion of train data to use (e.g., 1.0, 0.1, 0.01)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--roi_x", type=int, default=112)
    p.add_argument("--roi_y", type=int, default=112)
    p.add_argument("--roi_z", type=int, default=80)
    p.add_argument(
        "--norm",
        type=str,
        default="clip_zscore",
        choices=["clip_zscore", "fixed_wp5", "none"],
        help="Image normalization: 'clip_zscore' (p1/p99 clip + z-score), 'fixed_wp5' ([-3,8.5] -> [0,1]), or 'none'",
    )
    p.add_argument("--net", choices=["basicunet"], default="basicunet", help="Backbone architecture (WP5: basicunet recommended; UNet deprecated)")
    p.add_argument("--init", choices=["scratch", "pretrained"], default="scratch", help="Initialize randomly or load a pretrained checkpoint")
    p.add_argument("--pretrained_ckpt", type=str, default="", help="Checkpoint to initialize weights when --init pretrained")
    p.add_argument("--bundle_dir", type=str, default="", help="Path to MONAI bundle directory (with configs/ and models/)")
    p.add_argument("--ckpt", type=str, default="", help="Checkpoint to load for inference")
    p.add_argument("--save_preds", action="store_true", help="Save predictions during inference")
    # Official evaluation semantics are fixed (both-empty=1.0). Use scripts/eval_wp5.py to evaluate.
    p.add_argument(
        "--hd_percentile",
        type=float,
        default=100.0,
        help="Hausdorff percentile: 100.0 for HD, 95.0 for HD95 (matches /data3/wp5/wp5-code).",
    )
    # Few-shot configuration
    p.add_argument(
        "--fewshot_mode",
        type=str,
        default="few_samples",
        choices=["few_samples", "few_slices", "few_points"],
        help=(
            "Few-shot regime: 'few_samples' (default; use subset of volumes via --subset_ratio), "
            "'few_slices' (supervise K slices per patch), 'few_points' (sparse points + dilation)"
        ),
    )
    p.add_argument("--fewshot_ratio", type=float, default=0.0, help="Supervised fraction per patch (e.g., 0.1 or 0.01)")
    # Few-slices options
    p.add_argument("--fs_axis_mode", type=str, default="z", choices=["x", "y", "z", "multi"], help="Axis selection for few_slices mode")
    p.add_argument("--fs_class_aware", action="store_true", help="Class-aware slice selection (reserved; current selection uses FG counts)")
    p.add_argument("--fs_k_slices", type=int, default=-1, help="Explicit number of supervised slices; overrides ratio if >0")
    # Few-points options
    p.add_argument("--fp_dilate_radius", type=int, default=1, help="Dilation radius for seed points (1=>3x3x3, 2=>5x5x5)")
    p.add_argument("--fp_balance", type=str, default="proportional", choices=["proportional", "uniform"], help="Class balancing strategy for foreground seeds")
    p.add_argument("--fp_max_seeds", type=int, default=-1, help="Cap on number of point seeds per patch (auto if <=0)")
    p.add_argument("--fp_bg_frac", type=float, default=0.25, help="[Dynamic mode legacy] Fraction of seeds sampled as background per patch")
    p.add_argument("--fp_seed_strategy", type=str, default="random", choices=["random", "boundary"], help="Seed sampling strategy (boundary reserved; currently random)")
    p.add_argument("--fp_no_overlap", action="store_true", help="When sampling seeds, avoid any overlap of their dilation neighborhoods (spreads seeds out)")
    p.add_argument("--dilation_shape", type=str, default="auto", choices=["auto", "cube", "cross"], help="Dilation neighborhood: 'cube' (Chebyshev) or 'cross' (Manhattan); auto uses cross for ratio>=0.1 else cube")
    p.add_argument("--fp_sample_mode", type=str, default="stratified", choices=["stratified", "uniform_all"], help="Static few_points sampling: 'stratified' (FG/BG split + class balance) or 'uniform_all' (uniform over all voxels)")
    p.add_argument("--fp_uniform_exclude6", action="store_true", help="When --fp_sample_mode=uniform_all, sample only from voxels with label != 6 (background still included)")
    # Static global budget options
    p.add_argument("--global_budget", action="store_true", default=True, help="Enforce fewshot_ratio globally across the entire train set (seed budget)")
    p.add_argument("--seed_bg_frac", type=float, default=0.10, help="Global seed budget fraction reserved for background (class 0)")
    p.add_argument("--pseudo_weight", type=float, default=0.3, help="Loss weight for propagated labels in dilated regions (pseudo supervision)")
    # Few-shot fixed-budget control
    p.add_argument("--fewshot_static", action="store_true", help="Use a fixed per-volume supervision mask (precomputed once) for few_points mode")
    # Debug/testing flags
    p.add_argument("--force_no_supervision", action="store_true", help="Force supervision mask to zero (no labeled voxels) for debugging")
    # Augmentations
    p.add_argument("--aug_intensity", action="store_true", help="Enable intensity augmentations (noise, shift, scale)")
    p.add_argument("--aug_prob", type=float, default=0.2, help="Per-augmentation probability for intensity augs")
    p.add_argument("--aug_noise_std", type=float, default=0.01, help="Std for RandGaussianNoised when --aug_intensity")
    p.add_argument("--aug_shift", type=float, default=0.1, help="Offset for RandShiftIntensityd when --aug_intensity")
    p.add_argument("--aug_scale", type=float, default=0.1, help="Factor for RandScaleIntensityd when --aug_intensity")
    # Foreground-biased cropping
    p.add_argument("--fg_crop_prob", type=float, default=0.0, help="Probability to bias crop around foreground voxels (training only)")
    p.add_argument("--fg_crop_margin", type=int, default=8, help="Margin for foreground crop to keep ROI inside boundaries")
    # Pseudo-labels
    p.add_argument("--pl_enable", action="store_true", help="Enable confidence-thresholded pseudo-labels on unlabeled voxels")
    p.add_argument("--pl_threshold", type=float, default=0.9, help="Confidence threshold for pseudo-labeling (pmax >= threshold)")
    p.add_argument("--pl_weight", type=float, default=0.2, help="Weight for pseudo-label CE term")
    p.add_argument("--pl_warmup_epochs", type=int, default=5, help="Number of epochs to skip pseudo-labeling for warm-up")
    # Safety for mask precompute
    p.add_argument("--force_recompute_sup", action="store_true", help="Force overwriting existing static supervision masks when precomputing")
    # Debug / verification
    p.add_argument("--debug_grad_check", action="store_true", help="Log gradient stats and verify zeros outside supervision regions")
    p.add_argument("--debug_grad_batches", type=int, default=1, help="Number of batches to run grad-check logging across the whole training run")
    p.add_argument("--debug_grad_eps", type=float, default=1e-10, help="Tolerance for considering a gradient as non-zero in grad-check")
    p.add_argument("--debug_grad_topk", type=int, default=0, help="If >0, print top-K absolute gradient magnitudes outside supervision")
    # Sup mask persistence
    p.add_argument("--save_sup_masks", action="store_true", help="Save per-volume static supervision masks (few_points + --fewshot_static) to <output_dir>/sup_masks")
    p.add_argument("--sup_masks_dir", type=str, default="", help="Use precomputed supervision masks from this directory (overrides default <output_dir>/sup_masks)")
    # Static selection for few_slices
    p.add_argument("--slice_sel_json", type=str, default="", help="Optional selected_slices.json to precompute static sup masks for few_slices")
    # Static masks from selected points (1% seeds) and dense pseudo labels
    p.add_argument(
        "--selected_points_dir",
        type=str,
        default="",
        help="Directory containing per-case selected points (mask_selected_points.nii under <id>/)",
    )
    p.add_argument(
        "--pseudo_label_dir",
        type=str,
        default="",
        help="Directory containing dense pseudo labels mirroring original label filenames",
    )
    # Coverage logging mode
    p.add_argument("--coverage_mode", type=str, default="sup", choices=["sup", "seeds"], help="Report coverage using dilated supervised mask ('sup') or seed points only ('seeds')")
    # Output directory timestamp control
    p.add_argument(
        "--no_timestamp",
        action="store_true",
        help="Do not append timestamp to --output_dir (default behavior appends _YYYYmmdd-HHMMSS)",
    )
    # Logging
    p.add_argument("--log_to_file", action="store_true", default=True, help="Tee stdout/stderr to <output_dir>/train.log")
    p.add_argument("--log_file_name", type=str, default="train.log", help="Training log filename")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    # Append timestamp suffix to avoid overwriting runs unless explicitly disabled
    if not getattr(args, "no_timestamp", False):
        ts = time.strftime("%Y%m%d-%H%M%S")
        base = Path(args.output_dir)
        args.output_dir = str(base.parent / f"{base.name}_{ts}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"Using output directory: {args.output_dir}")
    # Only train mode is supported; evaluation is handled by scripts/eval_wp5.py
    train(args)
