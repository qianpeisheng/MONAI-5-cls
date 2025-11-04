#!/usr/bin/env python3
"""
Export 3D class-wise figures (static) for WP5 volumes.

For each case and for each foreground class (default: 1..4), saves a
single 3-panel 3D figure:
  [GT class surface] [Fully Pred class surface] [1% Voxel class surface]

This reuses the Matplotlib + scikit-image marching cubes approach from the
Streamlit app's fallback, but runs offline to produce publication-ready images.

Usage (single line):
  python3 scripts/export_wp5_3d_class_plots.py \
    --datalist /home/peisheng/MONAI/datalist_test.json \
    --pred_fully /home/peisheng/MONAI/runs/grid_clip_zscore/scratch_subset_100/eval_20251021-120429/preds \
    --pred_voxel1 /home/peisheng/MONAI/runs/fp_1pct_global_d0_20251021-153502_eval/preds \
    --out runs/exports_3d_class_plots --classes 1 2 3 4 --step 2 --opacity 0.55 --dpi 300

Requirements: numpy, nibabel, matplotlib, scikit-image
  pip install numpy nibabel matplotlib scikit-image
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List

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
    pred_vox1: Optional[np.ndarray],
    cls: int,
    step: int = 2,
    opacity: float = 0.55,
    dpi: int = 300,
) -> None:
    fig = plt.figure(figsize=(12, 4.5), dpi=dpi)
    fig.suptitle(f"{case_id} — class {cls}", fontsize=18, fontweight='bold')

    def add_panel(ax, mask: Optional[np.ndarray], title: str, color: str):
        ax.set_title(f"{title} (class {cls})", fontsize=14, fontweight='bold')
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

    # Panels: GT, Fully, 1% Voxel
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    add_panel(ax1, gt, 'GT', 'green')

    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    add_panel(ax2, pred_full, 'Fully', 'red')

    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    add_panel(ax3, pred_vox1, '1% Voxel', 'orange')

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(str(out_png), bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalist", type=str, default="/home/peisheng/MONAI/datalist_test.json")
    ap.add_argument("--pred_fully", type=str, default="/home/peisheng/MONAI/runs/grid_clip_zscore/scratch_subset_100/eval_20251021-120429/preds")
    ap.add_argument("--pred_voxel1", type=str, default="/home/peisheng/MONAI/runs/fp_1pct_global_d0_20251021-153502_eval/preds")
    ap.add_argument("--out", type=str, default="runs/exports_3d_class_plots")
    ap.add_argument("--classes", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--step", type=int, default=2, help="marching cubes step size (>=1)")
    ap.add_argument("--opacity", type=float, default=0.55)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--max_cases", type=int, default=0, help="0 means all")
    ap.add_argument("--case_filter", type=str, default="", help="Substring to filter case ids")
    args = ap.parse_args()

    datalist_p = Path(args.datalist)
    pred_full_dir = Path(args.pred_fully)
    pred_voxel_dir = Path(args.pred_voxel1)
    out_dir = Path(args.out)

    if not datalist_p.exists():
        print(f"ERROR: datalist not found: {datalist_p}", file=sys.stderr)
        sys.exit(1)
    if not pred_full_dir.exists():
        print(f"ERROR: pred_fully dir not found: {pred_full_dir}", file=sys.stderr)
        sys.exit(1)
    if not pred_voxel_dir.exists():
        print(f"ERROR: pred_voxel1 dir not found: {pred_voxel_dir}", file=sys.stderr)
        sys.exit(1)

    records = json.loads(datalist_p.read_text())
    if not isinstance(records, list) or not records:
        print("ERROR: datalist is empty or not a list", file=sys.stderr)
        sys.exit(1)

    done = 0
    for i, rec in enumerate(records):
        cid = rec.get("id", f"case_{i}")
        if args.case_filter and args.case_filter not in cid:
            continue
        img_path = Path(rec.get("image", ""))
        lbl_path = Path(rec.get("label", ""))
        if not img_path.exists() or not lbl_path.exists():
            print(f"[Skip] Missing image/label for case {cid}")
            continue

        p_full = find_pred_path(pred_full_dir, img_path, cid)
        p_vox1 = find_pred_path(pred_voxel_dir, img_path, cid)

        # Load volumes
        lbl_vol, _ = load_nii(lbl_path)
        pf_vol = None
        pv_vol = None
        if p_full is not None:
            pf_vol, _ = load_volume_any(p_full)
            if pf_vol.ndim == 4 and pf_vol.shape[0] == 1:
                pf_vol = pf_vol[0]
        if p_vox1 is not None:
            pv_vol, _ = load_volume_any(p_vox1)
            if pv_vol.ndim == 4 and pv_vol.shape[0] == 1:
                pv_vol = pv_vol[0]

        # Align shapes to label volume (labels usually define space)
        if pf_vol is not None and pf_vol.shape != lbl_vol.shape:
            pf_vol = center_pad_or_crop(pf_vol, lbl_vol.shape)
        if pv_vol is not None and pv_vol.shape != lbl_vol.shape:
            pv_vol = center_pad_or_crop(pv_vol, lbl_vol.shape)

        for cls in args.classes:
            out_png = out_dir / cid / f"class_{cls}.png"
            plot_class_surfaces_four_panel(
                out_png=out_png,
                case_id=cid,
                gt=lbl_vol,
                pred_full=pf_vol,
                pred_vox1=pv_vol,
                cls=cls,
                step=args.step,
                opacity=args.opacity,
                dpi=args.dpi,
            )

        done += 1
        if args.max_cases > 0 and done >= args.max_cases:
            break

    print(f"Done. Saved outputs under: {out_dir}")


if __name__ == "__main__":
    main()
