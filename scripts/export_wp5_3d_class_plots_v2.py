#!/usr/bin/env python3
"""
Export 4-panel 3D class-wise figures for WP5 volumes.

For each case and for each foreground class (default: 1..4), saves a
single 4-panel 3D figure:
  [GT] [Fully supervised] [Sparse supervision] [Ours]

Usage:
  python3 scripts/export_wp5_3d_class_plots_v2.py \
    --out runs/exports_3d_class_plots_v2 --max_cases 1 --dpi 300

Requirements: numpy, nibabel, matplotlib, scikit-image
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import nibabel as nib  # type: ignore
except Exception as e:  # pragma: no cover
    print("ERROR: nibabel is required. Try: pip install nibabel", file=sys.stderr)
    raise

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    print("ERROR: matplotlib is required. Try: pip install matplotlib", file=sys.stderr)
    raise

try:
    from skimage import measure as skmeasure  # type: ignore
except Exception as e:  # pragma: no cover
    print("ERROR: scikit-image is required. Try: pip install scikit-image", file=sys.stderr)
    raise


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


def extract_mesh_from_mask(mask3d: np.ndarray, step: int = 2) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    m = (mask3d.astype(np.uint8) > 0)
    voxels = int(m.sum())
    if voxels == 0:
        return None
    m = np.pad(m, 1, mode='constant', constant_values=0)
    vol = m.astype(np.float32)
    step_sz = max(1, int(step))
    try:
        verts, faces, _, _ = skmeasure.marching_cubes(
            vol, level=0.5, step_size=step_sz, allow_degenerate=True, method='lewiner'
        )
        return verts, faces
    except TypeError:
        try:
            verts, faces, _, _ = skmeasure.marching_cubes(vol, level=0.5, step_size=step_sz)
            return verts, faces
        except Exception:
            pass
    except Exception:
        try:
            mc_legacy = getattr(skmeasure, 'marching_cubes_lewiner', None)
            if mc_legacy is not None:
                verts, faces, _, _ = mc_legacy(vol, level=0.5, step_size=step_sz)
                return verts, faces
        except Exception:
            pass
    return None


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


def plot_class_surfaces_four_panel(
    out_png: Path,
    case_id: str,
    gt: Optional[np.ndarray],
    pred_full: Optional[np.ndarray],
    pred_sparse: Optional[np.ndarray],
    pred_ours: Optional[np.ndarray],
    cls: int,
    step: int = 2,
    opacity: float = 0.55,
    dpi: int = 300,
) -> None:
    fig = plt.figure(figsize=(16, 4.5), dpi=dpi)

    def add_panel(ax, mask: Optional[np.ndarray], title: str, color: str):
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_axis_off()
        if mask is None:
            ax.text2D(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center', va='center', fontsize=13, fontweight='bold')
            return
        m = (mask == cls).astype(np.uint8)
        if not m.any():
            ax.text2D(0.5, 0.5, 'No voxels', transform=ax.transAxes, ha='center', va='center', fontsize=13, fontweight='bold')
            return
        mesh = extract_mesh_from_mask(m, step=step)
        if mesh is None:
            ax.text2D(0.5, 0.5, 'Mesh failed', transform=ax.transAxes, ha='center', va='center', fontsize=13, fontweight='bold')
            return
        verts, faces = mesh
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color=color, linewidth=0.1, antialiased=True, alpha=opacity)

    ax1 = fig.add_subplot(1, 4, 1, projection='3d')
    add_panel(ax1, gt, 'GT', 'green')

    ax2 = fig.add_subplot(1, 4, 2, projection='3d')
    add_panel(ax2, pred_full, 'Fully supervised', 'red')

    ax3 = fig.add_subplot(1, 4, 3, projection='3d')
    add_panel(ax3, pred_sparse, 'Sparse supervision', 'blue')

    ax4 = fig.add_subplot(1, 4, 4, projection='3d')
    add_panel(ax4, pred_ours, 'Ours', 'orange')

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(str(out_png), bbox_inches='tight')
    plt.close(fig)


def _process_one_case(task):
    """Worker function: load volumes for one case, render all classes."""
    (rec_idx, rec, pred_full_dir, pred_sparse_dir, pred_ours_dir,
     out_dir, classes, step, opacity, dpi) = task

    cid = rec.get("id", f"case_{rec_idx}")
    img_path = Path(rec.get("image", ""))
    lbl_path = Path(rec.get("label", ""))
    if not img_path.exists() or not lbl_path.exists():
        return f"[Skip] Missing image/label for case {cid}"

    p_full = find_pred_path(pred_full_dir, img_path, cid)
    p_sparse = find_pred_path(pred_sparse_dir, img_path, cid)
    p_ours = find_pred_path(pred_ours_dir, img_path, cid)

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

    if pf_vol is not None and pf_vol.shape != lbl_vol.shape:
        pf_vol = center_pad_or_crop(pf_vol, lbl_vol.shape)
    if ps_vol is not None and ps_vol.shape != lbl_vol.shape:
        ps_vol = center_pad_or_crop(ps_vol, lbl_vol.shape)
    if po_vol is not None and po_vol.shape != lbl_vol.shape:
        po_vol = center_pad_or_crop(po_vol, lbl_vol.shape)

    for cls in classes:
        out_png = out_dir / cid / f"class_{cls}.png"
        plot_class_surfaces_four_panel(
            out_png=out_png, case_id=cid, gt=lbl_vol,
            pred_full=pf_vol, pred_sparse=ps_vol, pred_ours=po_vol,
            cls=cls, step=step, opacity=opacity, dpi=dpi,
        )
    return f"{cid} done ({len(classes)} classes)"


def main():
    import multiprocessing as mp

    ap = argparse.ArgumentParser()
    ap.add_argument("--datalist", type=str, default="/home/peisheng/MONAI/datalist_test_new.json")
    ap.add_argument("--pred_fully", type=str, default="/home/peisheng/MONAI/runs/wp5_full_supervised_20251210-164804/eval/preds")
    ap.add_argument("--pred_sparse", type=str, default="/home/peisheng/MONAI/runs/wp5_fewpoints_0_1pct_global_20251210-191641/eval/preds")
    ap.add_argument("--pred_ours", type=str, default="/data3/MONAI_experiments/sweep_graphlp_conf_ens_lossw_sweep40/COMQ_dec_bs_g4_0p20_g3_0p15_g2_0p00_20260130-005424/eval/preds")
    ap.add_argument("--out", type=str, default="runs/exports_3d_class_plots_v2")
    ap.add_argument("--classes", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--step", type=int, default=2, help="marching cubes step size (>=1)")
    ap.add_argument("--opacity", type=float, default=0.55)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--max_cases", type=int, default=0, help="0 means all")
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

    tasks = []
    for i, rec in enumerate(records):
        cid = rec.get("id", f"case_{i}")
        if args.case_filter and args.case_filter not in cid:
            continue
        tasks.append((
            i, rec, pred_full_dir, pred_sparse_dir, pred_ours_dir,
            out_dir, args.classes, args.step, args.opacity, args.dpi,
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
