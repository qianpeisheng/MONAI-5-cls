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
    ScaleIntensityRanged,
    SpatialPadd,
    SaveImage,
)
from monai.utils import set_determinism


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


def get_transforms(roi=(112, 112, 80)):
    train = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=-3, a_max=8.5, b_min=0.0, b_max=1.0, clip=True),
            # pad to at least ROI to avoid variable shapes when images are smaller than ROI
            SpatialPadd(keys=["image", "label"], spatial_size=roi),
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
            RandSpatialCropd(keys=["image", "label"], roi_size=roi, random_size=False),
        ]
    )
    val = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(keys=["image"], a_min=-3, a_max=8.5, b_min=0.0, b_max=1.0, clip=True),
            SpatialPadd(keys=["image", "label"], spatial_size=roi),
        ]
    )
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

    # If net has a top-level prefix like 'model.' and ckpt keys don't, prepend it
    net_keys = list(net.state_dict().keys())
    net_prefix = None
    if net_keys:
        first = net_keys[0]
        if "." in first:
            net_prefix = first.split(".", 1)[0]
    if net_prefix and not any(k.startswith(net_prefix + ".") for k in sd_try.keys()):
        sd_try = {f"{net_prefix}." + k: v for k, v in sd_try.items()}

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
            pred_mask = (pred > 0)
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
    t_train, t_val = get_transforms(roi=(args.roi_x, args.roi_y, args.roi_z))
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

    # Use smaller LR for finetuning by default
    base_lr = args.lr_ft if args.init == "pretrained" else args.lr
    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)
    # Add a cosine annealing scheduler for gradual LR decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=base_lr * 0.1)

    best_dice = -1.0
    roi = (args.roi_x, args.roi_y, args.roi_z)

    for epoch in range(1, args.epochs + 1):
        net.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for batch in dl_train:
            img = batch["image"].to(device)
            lbl = batch["label"].long().to(device)  # (B,1,...)
            ignore_mask = (lbl != 6)

            logits = net(img)
            ce = F.cross_entropy(logits, lbl.squeeze(1), ignore_index=6)
            dice = dice_loss_masked(logits, lbl, ignore_mask)
            loss = 0.5 * ce + 0.5 * dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        dur = time.time() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch}/{args.epochs} - loss {avg_loss:.4f} - {dur:.1f}s - lr {optimizer.param_groups[0]['lr']:.2e}")

        # Evaluate on test set (fast: skip HD/ASD to keep GPU utilization higher)
        metrics = evaluate(net, dl_test, device, out_dir, save_preds=False, max_cases=None, heavy=False)
        avg_dice = metrics["average"]["dice"]
        # Save epoch metrics
        epoch_metrics_path = out_dir / "metrics" / f"epoch_{epoch:03d}.json"
        epoch_metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"Epoch {epoch} test avg (0..4): {metrics['average']}")

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
    _, t_val = get_transforms(roi=(args.roi_x, args.roi_y, args.roi_z))
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
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate for scratch training")
    p.add_argument("--lr_ft", type=float, default=3e-5, help="Learning rate for finetuning when --init pretrained")
    p.add_argument("--subset_ratio", type=float, default=1.0, help="Proportion of train data to use (e.g., 1.0, 0.1, 0.01)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--roi_x", type=int, default=112)
    p.add_argument("--roi_y", type=int, default=112)
    p.add_argument("--roi_z", type=int, default=80)
    p.add_argument("--net", choices=["basicunet", "unet"], default="basicunet", help="Backbone architecture")
    p.add_argument("--init", choices=["scratch", "pretrained"], default="scratch", help="Initialize randomly or load a pretrained checkpoint")
    p.add_argument("--pretrained_ckpt", type=str, default="", help="Checkpoint to initialize weights when --init pretrained")
    p.add_argument("--bundle_dir", type=str, default="", help="Path to MONAI bundle directory (with configs/ and models/)")
    p.add_argument("--ckpt", type=str, default="", help="Checkpoint to load for inference")
    p.add_argument("--save_preds", action="store_true", help="Save predictions during inference")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.mode == "train":
        train(args)
    else:
        infer(args)
