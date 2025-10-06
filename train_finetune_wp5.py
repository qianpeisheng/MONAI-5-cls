#!/usr/bin/env python3
"""
WP5 fine-tuning script (full or partial data) with reproducible training, per-epoch evaluation, and checkpoint/inference utilities.

Usage examples:

Train on full data and evaluate each epoch:
  python train_finetune_wp5.py --mode train \
    --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
    --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json \
    --output_dir runs/wp5_finetune_full --subset_ratio 1.0 --epochs 50 --batch_size 2 --lr 1e-4

Train on 10% (deterministic subset) and evaluate:
  python train_finetune_wp5.py --mode train \
    --subset_ratio 0.1 --seed 42 --output_dir runs/wp5_finetune_10pct

Train from a pretrained checkpoint (non-strict load for head):
  python train_finetune_wp5.py --mode train \
    --init pretrained --pretrained_ckpt path/to/pretrained.ckpt \
    --output_dir runs/wp5_finetune_pretrained

Train from scratch explicitly:
  python train_finetune_wp5.py --mode train --init scratch --output_dir runs/wp5_finetune_scratch

Evaluate a saved checkpoint on test set and save predictions:
  python train_finetune_wp5.py --mode infer \
    --ckpt runs/wp5_finetune_full/best.ckpt \
    --save_preds --output_dir runs/wp5_finetune_full_infer

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


def subset_datalist(datalist: List[Dict], ratio: float, seed: int) -> List[Dict]:
    if ratio >= 0.999:
        return list(datalist)
    n = max(1, int(len(datalist) * ratio))
    rng = random.Random(seed)
    idxs = list(range(len(datalist)))
    rng.shuffle(idxs)
    idxs = sorted(idxs[:n])
    return [datalist[i] for i in idxs]


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
        if self.prob <= 0.0:
            return d
        if np.random.rand() > self.prob:
            return d
        img = d.get("image")
        lbl = d.get("label")
        if img is None or lbl is None:
            return d
        # Expect shapes (C, X, Y, Z)
        if lbl.ndim != 4:
            return d
        C, X, Y, Z = lbl.shape
        rx, ry, rz = self.roi
        # Valid region for crop corner
        minx = 0
        miny = 0
        minz = 0
        maxx = max(0, X - rx)
        maxy = max(0, Y - ry)
        maxz = max(0, Z - rz)
        # Find foreground voxels ignoring class 6
        fg_mask = (lbl[0] != 0) & (lbl[0] != 6)
        fg_indices = np.argwhere(fg_mask)
        if fg_indices.size == 0:
            return d  # fallback: keep as-is, next crop will handle
        # Pick a random foreground center
        cx, cy, cz = fg_indices[np.random.randint(0, fg_indices.shape[0])]
        # Compute crop start so that center is roughly centered, with margins
        sx = int(np.clip(cx - rx // 2, minx, maxx))
        sy = int(np.clip(cy - ry // 2, miny, maxy))
        sz = int(np.clip(cz - rz // 2, minz, maxz))
        # Apply crop for all keys present
        for k in self.keys:
            arr = d.get(k)
            if arr is None or arr.ndim != 4:
                continue
            d[k] = arr[:, sx : sx + rx, sy : sy + ry, sz : sz + rz]
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
        if norm_transform is not None:
            seq.append(norm_transform)
        seq.append(SpatialPadd(keys=["image", "label"], spatial_size=roi))
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
            seq.extend(
                [
                    RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
                    RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
                    RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
                ]
            )
        if include_crop:
            if fg_crop_prob > 0.0 and training:
                seq.append(FGBiasedCropD(keys=["image", "label"], roi_size=roi, prob=fg_crop_prob, margin=fg_crop_margin))
            else:
                seq.append(RandSpatialCropd(keys=["image", "label"], roi_size=roi, random_size=False))
        return seq

    train = Compose(build_seq(include_crop=True, include_aug=True, training=True))
    val = Compose(build_seq(include_crop=False, include_aug=False, training=False))
    return train, val


def build_model(arch: str = "basicunet") -> torch.nn.Module:
    """Build segmentation network.
    Default to BasicUNet to match MONAI spleen_ct_segmentation bundle family.
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


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor, heavy: bool = True) -> Dict[int, Dict[str, float]]:
    """Compute per-class Dice, IoU, and optionally HD/ASD for classes 0..4; ignore class 6.
    pred: (B,1,X,Y,Z), gt: (B,1,X,Y,Z)
    Returns mapping class->metrics.
    """
    B = pred.shape[0]
    ignore_mask = (gt != 6)
    classes = [0, 1, 2, 3, 4]
    if heavy:
        hd_metric = HausdorffDistanceMetric(percentile=100.0, reduction="none")
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
        dice = np.where(psum + gsum > 0, (2 * inter) / (psum + gsum + 1e-8), 1.0)
        iou = np.where(uni > 0, inter / (uni + 1e-8), 1.0)

        if heavy:
            # HD & ASD handling for empties
            hd_vals = np.zeros(B, dtype=np.float32)
            asd_vals = np.zeros(B, dtype=np.float32)
            for b in range(B):
                if psum[b] == 0 and gsum[b] == 0:
                    hd_vals[b] = 0.0
                    asd_vals[b] = 0.0
                elif psum[b] == 0 or gsum[b] == 0:
                    hd_vals[b] = 0.0
                    asd_vals[b] = 0.0
                else:
                    pt = torch.from_numpy(pm[b:b+1][None, ...].astype(np.float32))  # (1,1,...)
                    gt_t = torch.from_numpy(gm[b:b+1][None, ...].astype(np.float32))
                    try:
                        hd_vals[b] = float(np.array(hd_metric(pt, gt_t)).reshape(-1)[0])
                    except Exception:
                        hd_vals[b] = 0.0
                    try:
                        asd_vals[b] = float(np.array(asd_metric(pt, gt_t)).reshape(-1)[0])
                    except Exception:
                        asd_vals[b] = 0.0
            hd_mean = float(np.mean(hd_vals))
            asd_mean = float(np.mean(asd_vals))
        else:
            hd_mean = None
            asd_mean = None

        out[cls] = {"dice": float(np.mean(dice)), "iou": float(np.mean(iou)), "hd": hd_mean, "asd": asd_mean}

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


def build_points_supervision_mask(labels: torch.Tensor, ratio: float, dilate_radius: int, balance: str,
                                  max_seeds: int, bg_frac: float, seed_strategy: str) -> torch.Tensor:
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

        # Sample seeds
        seed_mask = torch.zeros_like(lbl, dtype=torch.bool)
        g = torch.Generator(device=device)
        g.manual_seed(torch.seed())

        # Foreground classes
        for c_i, idx in enumerate(cls_indices):
            n = int(per_cls[c_i])
            if n <= 0 or idx.shape[0] == 0:
                continue
            # random sampling of rows from idx
            perm = torch.randperm(idx.shape[0], generator=g, device=device)[: min(n, idx.shape[0])]
            sel = idx[perm]
            seed_mask[sel[:, 0], sel[:, 1], sel[:, 2]] = True

        # Background
        if n_bg > 0 and bg_idx.shape[0] > 0:
            perm = torch.randperm(bg_idx.shape[0], generator=g, device=device)[: min(n_bg, bg_idx.shape[0])]
            sel = bg_idx[perm]
            seed_mask[sel[:, 0], sel[:, 1], sel[:, 2]] = True

        # Dilate via max_pool3d trick
        seed_mask_t = seed_mask.unsqueeze(0).unsqueeze(0).float()  # (1,1,X,Y,Z)
        k = 2 * dilate_radius + 1
        dilated = F.max_pool3d(seed_mask_t, kernel_size=k, stride=1, padding=dilate_radius)
        sup = dilated > 0.5
        out_masks.append(sup.bool())

    return torch.cat(out_masks, dim=0)


def evaluate(
    net: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    out_dir: Path,
    save_preds: bool = False,
    max_cases: int | None = None,
    heavy: bool = True,
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

            per_class = compute_metrics(pred.cpu(), gt.cpu(), heavy=heavy)
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
    # save epoch metrics to JSON
    (metrics_dir / "summary.json").write_text(json.dumps({"per_class": summary, "average": avg}, indent=2))
    return {"per_class": summary, "average": avg}


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    data_root = Path(args.data_root)
    split_cfg = Path(args.split_cfg)
    out_dir = Path(args.output_dir)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    # Build datalists and optionally subset train
    train_list, test_list = build_datalists(data_root / "data", split_cfg)
    train_list = subset_datalist(train_list, args.subset_ratio, args.seed)

    # Transforms and datasets
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
    # Initialization policy
    if args.init == "pretrained":
        if args.pretrained_ckpt and Path(args.pretrained_ckpt).exists():
            print(f"Initializing from pretrained checkpoint: {args.pretrained_ckpt}")
            load_pretrained_non_strict(net, Path(args.pretrained_ckpt), device)
        else:
            print(
                f"WARNING: --init pretrained requested but --pretrained_ckpt not found: {args.pretrained_ckpt}. Proceeding without pretrained weights."
            )
    else:
        print("Initializing model from scratch (reinitialized weights for fair comparison)")
        # For fair comparison, follow the same code path as pretrained: 
        # if a checkpoint is available, load then reinitialize; otherwise just reinitialize
        if args.pretrained_ckpt and Path(args.pretrained_ckpt).exists():
            print(f"(scratch) Loading then reinitializing from: {args.pretrained_ckpt}")
            load_pretrained_non_strict(net, Path(args.pretrained_ckpt), device)
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
    print(
        f"Run config: init={args.init}, net={'bundle' if args.bundle_dir else args.net}, optimizer={opt_name}, base_lr={base_lr:.2e}, epochs={args.epochs}, milestones={milestones}"
    )

    for epoch in range(1, args.epochs + 1):
        net.train()
        epoch_loss = 0.0
        n_batches = 0
        cov_sum = 0.0
        pl_cov_sum = 0.0
        t0 = time.time()
        for batch in dl_train:
            img = batch["image"].to(device)
            lbl = batch["label"].long().to(device)  # (B,1,...)
            ignore_mask = (lbl != 6)

            # Build supervision mask if few-shot is enabled
            sup_mask = torch.ones_like(lbl, dtype=torch.bool, device=device)
            # Only build supervision mask for slice/point modes; 'few_samples' uses full supervision
            if args.fewshot_mode in ("few_slices", "few_points"):
                if args.fewshot_mode == "few_slices":
                    k_override = args.fs_k_slices if args.fs_k_slices > 0 else None
                    sup_mask = build_slice_supervision_mask(
                        labels=lbl, roi=(args.roi_x, args.roi_y, args.roi_z), axis_mode=args.fs_axis_mode,
                        ratio=args.fewshot_ratio, k_override=k_override,
                    )
                elif args.fewshot_mode == "few_points":
                    auto_max_seeds = args.fp_max_seeds
                    # Provide sensible defaults if not set
                    if auto_max_seeds <= 0:
                        auto_max_seeds = 5000 if args.fewshot_ratio >= 0.1 - 1e-8 else 500
                    sup_mask = build_points_supervision_mask(
                        labels=lbl, ratio=args.fewshot_ratio, dilate_radius=args.fp_dilate_radius,
                        balance=args.fp_balance, max_seeds=auto_max_seeds, bg_frac=args.fp_bg_frac,
                        seed_strategy=args.fp_seed_strategy,
                    )
                else:
                    raise ValueError(f"Unknown fewshot_mode: {args.fewshot_mode}")

            # Coverage logging (mean across batch)
            with torch.no_grad():
                cov = float(sup_mask.float().mean().item())
                cov_sum += cov

            logits = net(img)
            # Prepare CE targets with 255 as ignore index for unsupervised voxels or label==6
            ce_target = lbl.squeeze(1).clone()
            ce_target[(~sup_mask.squeeze(1)) | (ce_target == 6)] = 255
            ce_sup = F.cross_entropy(logits, ce_target, ignore_index=255)
            ce_pl = torch.tensor(0.0, device=device)
            # Optional pseudo-labels: only apply after warmup epochs
            if args.pl_enable and epoch > args.pl_warmup_epochs:
                with torch.no_grad():
                    probs = F.softmax(logits, dim=1)
                    pmax, pcls = probs.max(dim=1)  # (B,X,Y,Z)
                    unlabeled = (~sup_mask.squeeze(1)) & ignore_mask.squeeze(1)
                    pmask = unlabeled & (pmax >= args.pl_threshold)
                    pl_cov = float(pmask.float().mean().item())
                    pl_cov_sum += pl_cov
                if pmask.any():
                    pl_target = pcls
                    ce_pl_map = F.cross_entropy(logits, pl_target, reduction='none')  # (B,X,Y,Z)
                    ce_pl = (ce_pl_map * pmask.float()).sum() / pmask.float().sum()
                else:
                    ce_pl = torch.tensor(0.0, device=device)
            ce = ce_sup + args.pl_weight * ce_pl
            # Dice loss with combined mask
            dice = dice_loss_masked(logits, lbl, ignore_mask & sup_mask)
            loss = 0.5 * ce + 0.5 * dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        dur = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        cov_epoch = cov_sum / max(n_batches, 1)
        plcov_epoch = pl_cov_sum / max(n_batches, 1) if args.pl_enable else 0.0
        print(
            f"Epoch {epoch}/{args.epochs} - loss {avg_loss:.4f} - {dur:.1f}s - lr {optimizer.param_groups[0]['lr']:.2e}"
            + f" - sup_coverage {cov_epoch:.4f}"
            + (f" - pl_coverage {plcov_epoch:.4f}" if args.pl_enable else "")
        )

        # Evaluate on test set (fast: skip HD/ASD to keep GPU utilization higher)
        metrics = evaluate(net, dl_test, device, out_dir, save_preds=False, max_cases=None, heavy=False)
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


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    data_root = Path(args.data_root)
    split_cfg = Path(args.split_cfg)
    out_dir = Path(args.output_dir)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    # Build test list
    _, test_list = build_datalists(data_root / "data", split_cfg)
    _, t_val = get_transforms(roi=(args.roi_x, args.roi_y, args.roi_z), norm=args.norm)
    ds_test = Dataset(test_list, transform=t_val)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=args.num_workers)

    net = build_model().to(device)
    if args.ckpt and Path(args.ckpt).exists():
        sd = torch.load(args.ckpt, map_location=device)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        net.load_state_dict(sd, strict=False)
        print(f"Loaded checkpoint: {args.ckpt}")
    else:
        print("No valid checkpoint provided; running with randomly initialized weights.")

    # Full metrics (including HD/ASD) during inference
    metrics = evaluate(net, dl_test, device, out_dir, save_preds=args.save_preds, max_cases=None, heavy=True)
    (out_dir / "metrics" / "summary.json").write_text(json.dumps(metrics, indent=2))
    print("Inference complete. Metrics saved.")


def parse_args():
    p = argparse.ArgumentParser(description="WP5 Fine-tuning & Inference (MONAI)")
    p.add_argument("--mode", choices=["train", "infer"], default="train", help="Run mode")
    p.add_argument("--data_root", type=str, default="/data3/wp5/wp5-code/dataloaders/wp5-dataset", help="WP5 dataset root (contains data/)")
    p.add_argument("--split_cfg", type=str, default="/data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json", help="Predefined split config JSON")
    p.add_argument("--output_dir", type=str, default="runs/wp5_finetune", help="Output directory")
    p.add_argument("--epochs", type=int, default=100)
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
    p.add_argument("--net", choices=["basicunet", "unet"], default="basicunet", help="Backbone architecture")
    p.add_argument("--init", choices=["scratch", "pretrained"], default="scratch", help="Initialize randomly or load a pretrained checkpoint")
    p.add_argument("--pretrained_ckpt", type=str, default="", help="Checkpoint to initialize weights when --init pretrained")
    p.add_argument("--bundle_dir", type=str, default="", help="Path to MONAI bundle directory (with configs/ and models/)")
    p.add_argument("--ckpt", type=str, default="", help="Checkpoint to load for inference")
    p.add_argument("--save_preds", action="store_true", help="Save predictions during inference")
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
    p.add_argument("--fp_bg_frac", type=float, default=0.25, help="Fraction of seeds allocated to background class")
    p.add_argument("--fp_seed_strategy", type=str, default="random", choices=["random", "boundary"], help="Seed sampling strategy (boundary reserved; currently random)")
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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        train(args)
    else:
        infer(args)
