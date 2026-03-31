#!/usr/bin/env python3
"""
Export 5-panel slice figures for WP5 volumes.

For each case in a datalist (image/label/id), and for each slice along
the specified axes (x, y, z), this script saves a high-resolution image
with 5 columns: Data (grayscale), Ground Truth (overlay), Fully supervised
(overlay), Sparse supervision (overlay), and Ours (overlay).

Usage:
  python3 scripts/export_wp5_slice_quints.py \
    --out runs/exports_slice_quints_v2 --axes z --max_cases 1 --dpi 300

Requirements: numpy, nibabel, matplotlib
"""

from __future__ import annotations

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence

import numpy as np

try:
    import nibabel as nib  # type: ignore
except Exception as e:  # pragma: no cover
    print("ERROR: nibabel is required. Try: pip install nibabel", file=sys.stderr)
    raise

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    print("ERROR: matplotlib is required. Try: pip install matplotlib", file=sys.stderr)
    raise


def robust_minmax(vol2d: np.ndarray, pmin: float = 1.0, pmax: float = 99.0) -> Tuple[float, float]:
    flat = vol2d.reshape(-1)
    lo = float(np.percentile(flat, pmin))
    hi = float(np.percentile(flat, pmax))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(vol2d))
        hi = float(np.nanmax(vol2d))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = 0.0, 1.0
    return lo, hi


def colorize_seg2d(seg2d: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    h, w = seg2d.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    cmap = {
        1: (1.0, 0.0, 0.0),   # red
        2: (0.0, 1.0, 0.0),   # green
        3: (0.0, 0.3, 1.0),   # blue-ish
        4: (1.0, 1.0, 0.0),   # yellow
        6: (0.5, 0.5, 0.5),   # gray (ignored class)
    }
    for cls, rgb in cmap.items():
        m = (seg2d == cls)
        if not np.any(m):
            continue
        out[m, 0] = rgb[0]
        out[m, 1] = rgb[1]
        out[m, 2] = rgb[2]
        out[m, 3] = alpha
    return out


def load_nii(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    img = nib.load(str(path))
    try:
        img = nib.as_closest_canonical(img)
    except Exception:
        pass
    arr = img.get_fdata()
    return arr, getattr(img, "affine", None)


def load_volume_any(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    s = path.suffix.lower()
    if s == ".npy" or path.name.endswith(".npy"):
        arr = np.load(str(path))
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        return arr, None
    return load_nii(path)


def center_pad_or_crop(vol: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    import numpy as _np

    def _crop_to(v, ts):
        out = v
        for axis in range(3):
            diff = out.shape[axis] - ts[axis]
            if diff > 0:
                start = diff // 2
                end = start + ts[axis]
                slicer = [slice(None)] * out.ndim
                slicer[axis] = slice(start, end)
                out = out[tuple(slicer)]
        return out

    def _pad_to(v, ts):
        pad_width = []
        for axis in range(3):
            diff = ts[axis] - v.shape[axis]
            if diff > 0:
                left = diff // 2
                right = diff - left
                pad_width.append((left, right))
            else:
                pad_width.append((0, 0))
        return _np.pad(v, pad_width, mode='constant', constant_values=0)

    out = _crop_to(vol, target_shape)
    out = _pad_to(out, target_shape)
    return out


def find_pred_path(pred_dir: Path, image_path: Path, case_id: str) -> Optional[Path]:
    base_no_ext = image_path.name
    if base_no_ext.endswith('.nii.gz'):
        base_no_ext = base_no_ext[:-7]
    elif base_no_ext.endswith('.nii'):
        base_no_ext = base_no_ext[:-4]

    cand_nii = pred_dir / f"{base_no_ext}_pred.nii.gz"
    cand_npy = pred_dir / f"{base_no_ext}_pred.npy"
    if cand_nii.exists():
        return cand_nii
    if cand_npy.exists():
        return cand_npy

    for pattern in ("*pred*.nii*", "*pred*.npy"):
        for p in pred_dir.glob(pattern):
            if case_id and case_id in p.name:
                return p

    any_pred = sorted(list(pred_dir.glob("*pred*.nii*")) + list(pred_dir.glob("*pred*.npy")))
    return any_pred[0] if any_pred else None


def draw_and_save_quint(
    out_png: Path,
    img_vol: np.ndarray,
    lbl_vol: Optional[np.ndarray],
    pred_full: Optional[np.ndarray],
    pred_sparse: Optional[np.ndarray],
    pred_ours: Optional[np.ndarray],
    axis: str,
    index: int,
    alpha: float,
    dpi: int,
    case_id: str,
    nslices_axis: int,
) -> None:
    # Extract slice
    if axis == "x":
        img2d = img_vol[index, :, :]
        lbl2d = lbl_vol[index, :, :] if lbl_vol is not None else None
        pf2d = pred_full[index, :, :] if pred_full is not None else None
        ps2d = pred_sparse[index, :, :] if pred_sparse is not None else None
        po2d = pred_ours[index, :, :] if pred_ours is not None else None
    elif axis == "y":
        img2d = img_vol[:, index, :]
        lbl2d = lbl_vol[:, index, :] if lbl_vol is not None else None
        pf2d = pred_full[:, index, :] if pred_full is not None else None
        ps2d = pred_sparse[:, index, :] if pred_sparse is not None else None
        po2d = pred_ours[:, index, :] if pred_ours is not None else None
    else:  # z
        img2d = img_vol[:, :, index]
        lbl2d = lbl_vol[:, :, index] if lbl_vol is not None else None
        pf2d = pred_full[:, :, index] if pred_full is not None else None
        ps2d = pred_sparse[:, :, index] if pred_sparse is not None else None
        po2d = pred_ours[:, :, index] if pred_ours is not None else None

    import numpy as _np
    img2d = _np.flipud(_np.rot90(img2d, k=1))
    if lbl2d is not None:
        lbl2d = _np.flipud(_np.rot90(lbl2d, k=1))
    if pf2d is not None:
        pf2d = _np.flipud(_np.rot90(pf2d, k=1))
    if ps2d is not None:
        ps2d = _np.flipud(_np.rot90(ps2d, k=1))
    if po2d is not None:
        po2d = _np.flipud(_np.rot90(po2d, k=1))

    vmin, vmax = robust_minmax(img2d)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.8), dpi=dpi)

    # Panel 1: Data
    axes[0].imshow(img2d.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    axes[0].set_title("Data", fontsize=14, fontweight='bold')
    axes[0].axis("off")

    # Panel 2: GT overlay
    axes[1].imshow(img2d.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    if lbl2d is not None:
        axes[1].imshow(colorize_seg2d(lbl2d.T, alpha=alpha), origin="lower")
    axes[1].set_title("Ground Truth", fontsize=14, fontweight='bold')
    axes[1].axis("off")

    # Panel 3: Fully supervised overlay
    axes[2].imshow(img2d.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    if pf2d is not None:
        axes[2].imshow(colorize_seg2d(pf2d.T, alpha=alpha), origin="lower")
    axes[2].set_title("Fully supervised", fontsize=14, fontweight='bold')
    axes[2].axis("off")

    # Panel 4: Sparse supervision overlay
    axes[3].imshow(img2d.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    if ps2d is not None:
        axes[3].imshow(colorize_seg2d(ps2d.T, alpha=alpha), origin="lower")
    axes[3].set_title("Sparse supervision", fontsize=14, fontweight='bold')
    axes[3].axis("off")

    # Panel 5: Ours overlay
    axes[4].imshow(img2d.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    if po2d is not None:
        axes[4].imshow(colorize_seg2d(po2d.T, alpha=alpha), origin="lower")
    axes[4].set_title("Ours", fontsize=14, fontweight='bold')
    axes[4].axis("off")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), bbox_inches="tight")
    plt.close(fig)


def _process_one_case(task):
    """Worker function: load volumes for one case, render all slices."""
    (rec_idx, rec, pred_full_dir, pred_sparse_dir, pred_ours_dir,
     out_dir, axes_list, alpha, dpi, max_slices, skip_if_missing) = task

    cid = rec.get("id", f"case_{rec_idx}")
    img_path = Path(rec.get("image", ""))
    lbl_path = Path(rec.get("label", ""))
    if not img_path.exists() or not lbl_path.exists():
        return f"[Skip] Missing image/label for case {cid}"

    p_full = find_pred_path(pred_full_dir, img_path, cid)
    p_sparse = find_pred_path(pred_sparse_dir, img_path, cid)
    p_ours = find_pred_path(pred_ours_dir, img_path, cid)
    if skip_if_missing and (p_full is None or p_sparse is None or p_ours is None):
        return f"[Skip] Missing preds for case {cid}"

    img_vol, _ = load_nii(img_path)
    lbl_vol, _ = load_nii(lbl_path)
    pf_vol = ps_vol = po_vol = None
    if p_full is not None:
        pf_vol, _ = load_volume_any(p_full)
        if pf_vol.ndim == 4 and pf_vol.shape[0] == 1:
            pf_vol = pf_vol[0]
    if p_sparse is not None:
        ps_vol, _ = load_volume_any(p_sparse)
        if ps_vol.ndim == 4 and ps_vol.shape[0] == 1:
            ps_vol = ps_vol[0]
    if p_ours is not None:
        po_vol, _ = load_volume_any(p_ours)
        if po_vol.ndim == 4 and po_vol.shape[0] == 1:
            po_vol = po_vol[0]

    if lbl_vol.shape != img_vol.shape:
        lbl_vol = center_pad_or_crop(lbl_vol, img_vol.shape)
    if pf_vol is not None and pf_vol.shape != img_vol.shape:
        pf_vol = center_pad_or_crop(pf_vol, img_vol.shape)
    if ps_vol is not None and ps_vol.shape != img_vol.shape:
        ps_vol = center_pad_or_crop(ps_vol, img_vol.shape)
    if po_vol is not None and po_vol.shape != img_vol.shape:
        po_vol = center_pad_or_crop(po_vol, img_vol.shape)

    n_slices = 0
    for ax in axes_list:
        dim = {"x": 0, "y": 1, "z": 2}[ax]
        nslices = int(img_vol.shape[dim])
        limit = nslices if max_slices <= 0 else min(nslices, max_slices)
        for sidx in range(limit):
            out_png = out_dir / cid / ax / f"{cid}_{ax}{sidx:03d}.png"
            draw_and_save_quint(
                out_png=out_png, img_vol=img_vol, lbl_vol=lbl_vol,
                pred_full=pf_vol, pred_sparse=ps_vol, pred_ours=po_vol,
                axis=ax, index=sidx, alpha=alpha, dpi=dpi,
                case_id=cid, nslices_axis=nslices,
            )
            n_slices += 1
    return f"{cid} done ({n_slices} slices)"


def main():
    import multiprocessing as mp

    ap = argparse.ArgumentParser()
    ap.add_argument("--datalist", type=str, default="/home/peisheng/MONAI/datalist_test_new.json")
    ap.add_argument("--pred_fully", type=str, default="/home/peisheng/MONAI/runs/wp5_full_supervised_20251210-164804/eval/preds")
    ap.add_argument("--pred_sparse", type=str, default="/home/peisheng/MONAI/runs/wp5_fewpoints_0_1pct_global_20251210-191641/eval/preds")
    ap.add_argument("--pred_ours", type=str, default="/data3/MONAI_experiments/sweep_graphlp_conf_ens_lossw_sweep40/COMQ_dec_bs_g4_0p20_g3_0p15_g2_0p00_20260130-005424/eval/preds")
    ap.add_argument("--out", type=str, default="runs/exports_slice_quints_v2")
    ap.add_argument("--axes", type=str, nargs="+", default=["z", "y", "x"], choices=["x", "y", "z"])
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--max_cases", type=int, default=0, help="0 means all")
    ap.add_argument("--max_slices", type=int, default=0, help="0 means all per axis")
    ap.add_argument("--skip_if_missing_pred", action="store_true", help="Skip case if any pred missing")
    ap.add_argument("--case_filter", type=str, default="", help="Substring to filter case ids")
    ap.add_argument("--workers", type=int, default=0, help="0 = number of CPUs")
    args = ap.parse_args()

    datalist_p = Path(args.datalist)
    pred_full_dir = Path(args.pred_fully)
    pred_sparse_dir = Path(args.pred_sparse)
    pred_ours_dir = Path(args.pred_ours)
    out_dir = Path(args.out)

    if not datalist_p.exists():
        print(f"ERROR: datalist not found: {datalist_p}", file=sys.stderr)
        sys.exit(1)
    for label, d in [("pred_fully", pred_full_dir), ("pred_sparse", pred_sparse_dir), ("pred_ours", pred_ours_dir)]:
        if not d.exists():
            print(f"ERROR: {label} dir not found: {d}", file=sys.stderr)
            sys.exit(1)

    records = json.loads(datalist_p.read_text())
    if not isinstance(records, list) or not records:
        print("ERROR: datalist is empty or not a list", file=sys.stderr)
        sys.exit(1)

    # Build task list
    tasks = []
    for i, rec in enumerate(records):
        cid = rec.get("id", f"case_{i}")
        if args.case_filter and args.case_filter not in cid:
            continue
        tasks.append((
            i, rec, pred_full_dir, pred_sparse_dir, pred_ours_dir,
            out_dir, args.axes, args.alpha, args.dpi,
            args.max_slices, args.skip_if_missing_pred,
        ))
        if args.max_cases > 0 and len(tasks) >= args.max_cases:
            break

    n_workers = args.workers if args.workers > 0 else min(mp.cpu_count(), len(tasks))
    print(f"Processing {len(tasks)} cases with {n_workers} workers ...")

    with mp.Pool(n_workers) as pool:
        for idx, result in enumerate(pool.imap_unordered(_process_one_case, tasks)):
            print(f"[{idx+1}/{len(tasks)}] {result}")

    print(f"Done. Saved outputs under: {out_dir}")


if __name__ == "__main__":
    main()
