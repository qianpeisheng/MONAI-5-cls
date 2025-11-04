#!/usr/bin/env python3
"""
Streamlit visualization for WP5 3D segmentations.

Features:
- Browse cases from a datalist (JSON of {image,label,id}).
- Show 2D slices with overlays for ground-truth labels and predictions.
- Optional 3D view using Plotly or Matplotlib (isosurface per selected class).
- Compares four fixed runs: Fully, 1% Volume, 1% Slice, 1% Voxel.

Launch:
  streamlit run scripts/vis_wp5_streamlit.py

Requirements: streamlit, numpy, nibabel, matplotlib, plotly
  pip install streamlit numpy nibabel matplotlib plotly
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st


# Optional dependencies
try:
    import nibabel as nib  # type: ignore
except Exception:
    nib = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

try:
    import plotly.graph_objects as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore
except Exception:
    go = None

# Optional: scikit-image for mesh extraction
try:
    from skimage import measure as skmeasure  # type: ignore
except Exception:
    skmeasure = None

# Fixed configuration for this app (no user-controlled folders)
DATALIST_PATH = Path("/home/peisheng/MONAI/datalist_test.json")
RUNS = {
    "Fully Supervised": Path("/home/peisheng/MONAI/runs/grid_clip_zscore/scratch_subset_100/eval_20251021-120429/preds"),
    "1% Volume": Path("/home/peisheng/MONAI/runs/grid_clip_zscore/scratch_subset_1/eval_20251021-133307/preds"),
    "1% Slice": Path("/home/peisheng/MONAI/runs/few_slices_allcls_gpu0_20251021-110741/eval_20251021-122416/preds"),
    "1% Voxel": Path("/home/peisheng/MONAI/runs/fp_1pct_global_d0_20251021-153502_eval/preds"),
}


def parse_cli_args():
    # Legacy parser kept for backward-compat; not used.
    class _Args:
        pass
    return _Args()


def load_json(path: Path):
    return [] if not path.exists() else __import__("json").loads(path.read_text())


def robust_minmax(vol: np.ndarray, pmin=1, pmax=99) -> Tuple[float, float]:
    flat = vol.reshape(-1)
    lo = float(np.percentile(flat, pmin))
    hi = float(np.percentile(flat, pmax))
    if hi <= lo:
        hi = float(vol.max())
        lo = float(vol.min())
    return lo, hi


def dice_per_class(pred: np.ndarray, gt: np.ndarray, classes=(0, 1, 2, 3, 4), ignore_label: Optional[int] = 6) -> Dict[int, float]:
    out: Dict[int, float] = {}
    # Optionally ignore label 6 by masking those voxels out in both arrays
    if ignore_label is not None:
        mask = gt != ignore_label
        gt = gt[mask]
        pred = pred[mask]
    for c in classes:
        p = pred == c
        g = gt == c
        inter = float(np.logical_and(p, g).sum())
        denom = float(p.sum() + g.sum())
        out[c] = (2.0 * inter / denom) if denom > 0 else (1.0 if p.sum() == g.sum() else 0.0)
    return out


def counts_per_class(vol: np.ndarray, classes=(0, 1, 2, 3, 4, 6)) -> Dict[int, int]:
    return {int(c): int((vol == c).sum()) for c in classes}


def find_pseudo_label_path(pseudo_dir: Path, label_path: Path) -> Path:
    # Mirror the label filename under pseudo_dir
    return pseudo_dir / label_path.name


def find_pred_path(pred_dir: Path, image_path: Path, case_id: str) -> Optional[Path]:
    # Normalize base name without .nii/.nii.gz
    base = image_path.name
    base_no_ext = base
    if base_no_ext.endswith('.nii.gz'):
        base_no_ext = base_no_ext[:-7]
    elif base_no_ext.endswith('.nii'):
        base_no_ext = base_no_ext[:-4]

    # Heuristic 1: exact base + _pred with nii.gz or npy
    cand_nii = pred_dir / f"{base_no_ext}_pred.nii.gz"
    cand_npy = pred_dir / f"{base_no_ext}_pred.npy"
    if cand_nii.exists():
        return cand_nii
    if cand_npy.exists():
        return cand_npy

    # Heuristic 2: files that contain the case_id and 'pred' (nii or npy)
    for pattern in ("*pred*.nii*", "*pred*.npy"):
        for p in pred_dir.glob(pattern):
            if case_id and case_id in p.name:
                return p

    # Heuristic 3: fallback to first pred* file
    any_pred = sorted(list(pred_dir.glob("*pred*.nii*")) + list(pred_dir.glob("*pred*.npy")))
    return any_pred[0] if any_pred else None


def load_nii(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if nib is None:
        st.error("nibabel not installed. pip install nibabel")
        st.stop()
    img = nib.load(str(path))
    # Canonicalize to RAS to match training/eval Orientationd(axcodes='RAS')
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
        # squeeze leading channel dim if present
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        return arr, None
    else:
        return load_nii(path)


def center_pad_or_crop(vol: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Center pad or crop a 3D volume to the target shape using nearest-neighbor semantics.
    Keeps integer labels intact when used on segmentations.
    """
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


def colorize_seg2d(seg2d: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    # Map classes to colors: 0 background transparent; 1 red, 2 green, 3 blue, 4 yellow, 6 ignored gray
    h, w = seg2d.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    cmap = {
        1: (1.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0),
        3: (0.0, 0.3, 1.0),
        4: (1.0, 1.0, 0.0),
        6: (0.5, 0.5, 0.5),
    }
    for cls, rgb in cmap.items():
        m = (seg2d == cls)
        out[m, 0] = rgb[0]
        out[m, 1] = rgb[1]
        out[m, 2] = rgb[2]
        out[m, 3] = alpha
    return out


def _plot_points_crosses(ax, mask2d: np.ndarray, size: float = 60.0, lw: float = 2.0, alpha: float = 1.0):
    """Plot cross markers for selected points mask on a 2D axes.

    - mask2d contains class ids at selected voxels (0 means none).
    - Uses class colors matching overlays for consistency.
    """
    if mask2d is None:
        return
    m = mask2d.astype(np.int32)
    # Class color map consistent with colorize_seg2d
    cmap = {
        1: (1.0, 0.0, 0.0),
        2: (0.0, 1.0, 0.0),
        3: (0.0, 0.3, 1.0),
        4: (1.0, 1.0, 0.0),
    }
    classes = [c for c in np.unique(m) if c in cmap]
    for c in classes:
        ys, xs = np.nonzero(m == c)
        if ys.size == 0:
            continue
        color = cmap[c]
        # Note: image displayed with .T and origin='lower', so use (x=xs, y=ys)
        ax.scatter(
            xs,
            ys,
            marker="x",
            s=size,
            linewidths=lw,
            c=[color],
            alpha=alpha,
            zorder=5,
        )


def draw_slice(image: np.ndarray, label: Optional[np.ndarray], pred: Optional[np.ndarray], axis: str, index: int, alpha: float):
    if plt is None:
        st.warning("matplotlib not installed; 2D view disabled. pip install matplotlib")
        return
    # Extract 2D slice
    if axis == "x":
        img2d = image[index, :, :]
        lbl2d = label[index, :, :] if label is not None else None
        prd2d = pred[index, :, :] if pred is not None else None
    elif axis == "y":
        img2d = image[:, index, :]
        lbl2d = label[:, index, :] if label is not None else None
        prd2d = pred[:, index, :] if pred is not None else None
    else:  # z
        img2d = image[:, :, index]
        lbl2d = label[:, :, index] if label is not None else None
        prd2d = pred[:, :, index] if pred is not None else None

    # Rotate image and colored overlays 90 degrees CCW and flip vertically to align with point overlays
    import numpy as _np
    img2d = _np.flipud(_np.rot90(img2d, k=1))
    if lbl2d is not None:
        lbl2d = _np.flipud(_np.rot90(lbl2d, k=1))
    if prd2d is not None:
        prd2d = _np.flipud(_np.rot90(prd2d, k=1))

    vmin, vmax = robust_minmax(img2d)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img2d.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    if lbl2d is not None:
        ax.imshow(colorize_seg2d(lbl2d.T, alpha=alpha), origin="lower")
    if prd2d is not None:
        ax.imshow(colorize_seg2d(prd2d.T, alpha=alpha), origin="lower")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)


def draw_slice_pair(
    image: np.ndarray,
    label: Optional[np.ndarray],
    pred: Optional[np.ndarray],
    axis: str,
    index: int,
    alpha: float,
    title_left: str = "Ground Truth",
    title_right: str = "Prediction",
    overlay_left: Optional[np.ndarray] = None,
    overlay_right: Optional[np.ndarray] = None,
):
    if plt is None:
        st.warning("matplotlib not installed; 2D view disabled. pip install matplotlib")
        return
    # Extract 2D slice
    if axis == "x":
        img2d = image[index, :, :]
        lbl2d = label[index, :, :] if label is not None else None
        prd2d = pred[index, :, :] if pred is not None else None
    elif axis == "y":
        img2d = image[:, index, :]
        lbl2d = label[:, index, :] if label is not None else None
        prd2d = pred[:, index, :] if pred is not None else None
    else:  # z
        img2d = image[:, :, index]
        lbl2d = label[:, :, index] if label is not None else None
        prd2d = pred[:, :, index] if pred is not None else None

    # Rotate image and colored overlays 90 degrees CCW and flip vertically to align with point overlays
    import numpy as _np
    img2d = _np.flipud(_np.rot90(img2d, k=1))
    if lbl2d is not None:
        lbl2d = _np.flipud(_np.rot90(lbl2d, k=1))
    if prd2d is not None:
        prd2d = _np.flipud(_np.rot90(prd2d, k=1))

    vmin, vmax = robust_minmax(img2d)
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        ax1.set_title(title_left)
        ax1.imshow(img2d.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        if lbl2d is not None:
            ax1.imshow(colorize_seg2d(lbl2d.T, alpha=alpha), origin="lower")
        if overlay_left is not None:
            ov = overlay_left
            if axis == "x":
                ov2d = ov[index, :, :]
            elif axis == "y":
                ov2d = ov[:, index, :]
            else:
                ov2d = ov[:, :, index]
            # Plot crosses for selected points for better visibility
            _plot_points_crosses(ax1, ov2d, size=80.0, lw=2.5, alpha=1.0)
        ax1.axis("off")
        st.pyplot(fig1, clear_figure=True)
    with col2:
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
        ax2.set_title(title_right)
        ax2.imshow(img2d.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        if prd2d is not None:
            ax2.imshow(colorize_seg2d(prd2d.T, alpha=alpha), origin="lower")
        if overlay_right is not None:
            ov = overlay_right
            if axis == "x":
                ov2d = ov[index, :, :]
            elif axis == "y":
                ov2d = ov[:, index, :]
            else:
                ov2d = ov[:, :, index]
            _plot_points_crosses(ax2, ov2d, size=80.0, lw=2.5, alpha=1.0)
        ax2.axis("off")
        st.pyplot(fig2, clear_figure=True)


def draw_slice_triplet(
    image: np.ndarray,
    label: Optional[np.ndarray],
    pred_left: Optional[np.ndarray],
    pred_middle: Optional[np.ndarray],
    pred_right: Optional[np.ndarray],
    axis: str,
    index: int,
    alpha: float,
    title_left: str = "GT",
    title_middle: str = "Pred A",
    title_right: str = "Pred B",
):
    if plt is None:
        st.warning("matplotlib not installed; 2D view disabled. pip install matplotlib")
        return
    # Extract 2D slice
    if axis == "x":
        img2d = image[index, :, :]
        lbl2d = label[index, :, :] if label is not None else None
        prd_l = pred_left[index, :, :] if pred_left is not None else None
        prd_m = pred_middle[index, :, :] if pred_middle is not None else None
        prd_r = pred_right[index, :, :] if pred_right is not None else None
    elif axis == "y":
        img2d = image[:, index, :]
        lbl2d = label[:, index, :] if label is not None else None
        prd_l = pred_left[:, index, :] if pred_left is not None else None
        prd_m = pred_middle[:, index, :] if pred_middle is not None else None
        prd_r = pred_right[:, index, :] if pred_right is not None else None
    else:  # z
        img2d = image[:, :, index]
        lbl2d = label[:, :, index] if label is not None else None
        prd_l = pred_left[:, :, index] if pred_left is not None else None
        prd_m = pred_middle[:, :, index] if pred_middle is not None else None
        prd_r = pred_right[:, :, index] if pred_right is not None else None

    import numpy as _np
    img2d = _np.flipud(_np.rot90(img2d, k=1))
    if lbl2d is not None:
        lbl2d = _np.flipud(_np.rot90(lbl2d, k=1))
    if prd_l is not None:
        prd_l = _np.flipud(_np.rot90(prd_l, k=1))
    if prd_m is not None:
        prd_m = _np.flipud(_np.rot90(prd_m, k=1))
    if prd_r is not None:
        prd_r = _np.flipud(_np.rot90(prd_r, k=1))

    vmin, vmax = robust_minmax(img2d)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    # Left: GT only
    axes[0].imshow(img2d.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    if lbl2d is not None:
        axes[0].imshow(colorize_seg2d(lbl2d.T, alpha=alpha), origin="lower")
    if prd_l is not None:
        axes[0].imshow(colorize_seg2d(prd_l.T, alpha=alpha), origin="lower")
    axes[0].set_title(title_left)
    axes[0].axis("off")
    # Middle: Pred A
    axes[1].imshow(img2d.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    if lbl2d is not None:
        axes[1].imshow(colorize_seg2d(lbl2d.T, alpha=alpha), origin="lower")
    if prd_m is not None:
        axes[1].imshow(colorize_seg2d(prd_m.T, alpha=alpha), origin="lower")
    axes[1].set_title(title_middle)
    axes[1].axis("off")
    # Right: Pred B
    axes[2].imshow(img2d.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    if lbl2d is not None:
        axes[2].imshow(colorize_seg2d(lbl2d.T, alpha=alpha), origin="lower")
    if prd_r is not None:
        axes[2].imshow(colorize_seg2d(prd_r.T, alpha=alpha), origin="lower")
    axes[2].set_title(title_right)
    axes[2].axis("off")
    st.pyplot(fig, clear_figure=True)


def normalize_image_for_volume(image: np.ndarray) -> np.ndarray:
    # Robust normalize to [0,1]
    lo, hi = robust_minmax(image)
    im = image.astype(np.float32)
    if hi > lo:
        return (im - lo) / (hi - lo)
    return np.zeros_like(im, dtype=np.float32)


def plot_3d(image: np.ndarray, seg: Optional[np.ndarray], cls: int, downsample: int = 2, opacity: float = 0.15, show_image: bool = False):
    if go is None:
        st.warning("plotly not installed; 3D view disabled. pip install plotly")
        return
    # Downsample for speed
    def ds(v):
        if downsample <= 1:
            return v
        return v[::downsample, ::downsample, ::downsample]

    img_ds = ds(normalize_image_for_volume(image))
    fig = go.Figure()
    if show_image:
        fig.add_trace(
            go.Volume(
                value=img_ds,
                opacity=0.12,
                surface_count=12,
                colorscale="Gray",
                showscale=False,
                isomin=0.0,
                isomax=1.0,
                opacityscale=[[0.0, 0.0], [0.2, 0.0], [1.0, 1.0]],
                caps=dict(x_show=False, y_show=False, z_show=False),
            )
        )
    if seg is not None and cls in (1, 2, 3, 4):
        mask = (seg == cls).astype(np.float32)
        mask_ds = ds(mask)
        fig.add_trace(
            go.Isosurface(
                value=mask_ds,
                isomin=0.5,
                isomax=1.0,
                surface_count=1,
                opacity=opacity,
                colorscale=[[0, "red"], [1, "red"]] if cls == 1 else (
                    [[0, "green"], [1, "green"]] if cls == 2 else (
                        [[0, "blue"], [1, "blue"]] if cls == 3 else [[0, "yellow"], [1, "yellow"]]
                    )
                ),
                showscale=False,
                caps=dict(x_show=False, y_show=False, z_show=False),
            )
        )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)


def plot_3d_pair(
    image: np.ndarray,
    gt: Optional[np.ndarray],
    pred: Optional[np.ndarray],
    cls: int,
    downsample: int = 2,
    opacity: float = 0.25,
    show_image: bool = False,
    right_title: str = "Prediction",
):
    if go is None:
        st.warning("plotly not installed; 3D view disabled. pip install plotly")
        return
    # Downsample helper
    def ds(v):
        if downsample <= 1:
            return v
        return v[::downsample, ::downsample, ::downsample]

    img_ds = ds(normalize_image_for_volume(image))
    specs = [[{"type": "scene"}, {"type": "scene"}]]
    fig = make_subplots(rows=1, cols=2, specs=specs, subplot_titles=("GT", right_title))

    if show_image:
        for col in (1, 2):
            fig.add_trace(
                go.Volume(
                    value=img_ds,
                    opacity=0.10,
                    surface_count=8,
                    colorscale="Gray",
                    showscale=False,
                    isomin=0.0,
                    isomax=1.0,
                    opacityscale=[[0.0, 0.0], [0.2, 0.0], [1.0, 1.0]],
                    caps=dict(x_show=False, y_show=False, z_show=False),
                ),
                row=1, col=col,
            )

    # GT isosurface
    if gt is not None and cls in (1, 2, 3, 4):
        gmask = (gt == cls).astype(np.float32)
        if gmask.sum() > 0:
            fig.add_trace(
                go.Isosurface(
                    value=ds(gmask),
                    isomin=0.5,
                    isomax=1.0,
                    surface_count=1,
                    opacity=opacity,
                    colorscale=[[0, "green"], [1, "green"]],
                    showscale=False,
                    caps=dict(x_show=False, y_show=False, z_show=False),
                ),
                row=1, col=1,
            )
        else:
            st.info(f"No GT voxels for class {cls} in this case.")

    # Pred isosurface
    if pred is not None and cls in (1, 2, 3, 4):
        pmask = (pred == cls).astype(np.float32)
        if pmask.sum() > 0:
            fig.add_trace(
                go.Isosurface(
                    value=ds(pmask),
                    isomin=0.5,
                    isomax=1.0,
                    surface_count=1,
                    opacity=opacity,
                    colorscale=[[0, "red"], [1, "red"]],
                    showscale=False,
                    caps=dict(x_show=False, y_show=False, z_show=False),
                ),
                row=1, col=2,
            )
        else:
            st.info(f"No Pred voxels for class {cls} in this case.")

    # Layout
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(aspectmode="data"),
        scene2=dict(aspectmode="data"),
    )
    st.plotly_chart(fig, use_container_width=True)


def extract_mesh_from_mask(mask3d: np.ndarray, step: int = 2) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if skmeasure is None:
        return None
    m = (mask3d.astype(np.uint8) > 0)
    # Ensure non-empty and at least some zeros and ones for a surface to exist
    voxels = int(m.sum())
    if voxels == 0:
        return None
    # Add 1-voxel zero padding around to avoid boundary artifacts
    m = np.pad(m, 1, mode='constant', constant_values=0)
    vol = m.astype(np.float32)
    step_sz = max(1, int(step))
    # Try modern API first
    try:
        verts, faces, _, _ = skmeasure.marching_cubes(
            vol, level=0.5, step_size=step_sz, allow_degenerate=True, method='lewiner'
        )
        return verts, faces
    except TypeError:
        # Older scikit-image without allow_degenerate/method
        try:
            verts, faces, _, _ = skmeasure.marching_cubes(vol, level=0.5, step_size=step_sz)
            return verts, faces
        except Exception:
            pass
    except Exception:
        # Try legacy function
        try:
            mc_legacy = getattr(skmeasure, 'marching_cubes_lewiner', None)
            if mc_legacy is not None:
                verts, faces, _, _ = mc_legacy(vol, level=0.5, step_size=step_sz)
                return verts, faces
        except Exception:
            pass
    return None


def plot_3d_pair_matplotlib(
    gt: Optional[np.ndarray],
    pred: Optional[np.ndarray],
    cls: int,
    downsample: int = 2,
    opacity: float = 0.5,
    right_title: str = "Pred",
):
    if plt is None:
        st.warning("matplotlib not installed; 3D fallback disabled.")
        return
    if skmeasure is None:
        st.warning("scikit-image not installed. pip install scikit-image for 3D fallback rendering.")
        return

    # Downsample helper
    def ds(v):
        s = max(1, int(downsample))
        return v[::s, ::s, ::s]

    col1, col2 = st.columns(2)

    # Ground Truth pane
    with col1:
        fig = plt.figure(figsize=(5.5, 5.0))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"GT class {cls}")
        if gt is not None:
            gmask = (gt == cls).astype(np.uint8)
            if gmask.any():
                gmask = ds(gmask)
                mesh = extract_mesh_from_mask(gmask, step=max(1, int(downsample)))
                if mesh is not None:
                    verts, faces = mesh
                    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color='green', linewidth=0.1, antialiased=True, alpha=opacity)
                else:
                    # Fallback: scatter a subset of voxels to indicate shape
                    idx = np.column_stack(np.nonzero(gmask))
                    if idx.shape[0] > 0:
                        sel = idx
                        if idx.shape[0] > 5000:
                            perm = np.random.RandomState(0).permutation(idx.shape[0])[:5000]
                            sel = idx[perm]
                        ax.scatter(sel[:, 0], sel[:, 1], sel[:, 2], s=1, c='green', alpha=min(0.8, opacity+0.2))
                        st.info("GT: using point-cloud fallback (mesh extraction failed)")
            else:
                st.info("No GT voxels for selected class.")
        ax.set_axis_off()
        st.pyplot(fig, clear_figure=True)

    # Prediction pane
    with col2:
        fig = plt.figure(figsize=(5.5, 5.0))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"{right_title} class {cls}")
        if pred is not None:
            pmask = (pred == cls).astype(np.uint8)
            if pmask.any():
                pmask = ds(pmask)
                mesh = extract_mesh_from_mask(pmask, step=max(1, int(downsample)))
                if mesh is not None:
                    verts, faces = mesh
                    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color='red', linewidth=0.1, antialiased=True, alpha=opacity)
                else:
                    # Fallback: scatter subset of pred voxels
                    idx = np.column_stack(np.nonzero(pmask))
                    if idx.shape[0] > 0:
                        sel = idx
                        if idx.shape[0] > 5000:
                            perm = np.random.RandomState(0).permutation(idx.shape[0])[:5000]
                            sel = idx[perm]
                        ax.scatter(sel[:, 0], sel[:, 1], sel[:, 2], s=1, c='red', alpha=min(0.8, opacity+0.2))
                        st.info("Pred: using point-cloud fallback (mesh extraction failed)")
            else:
                st.info("No Pred voxels for selected class.")
        ax.set_axis_off()
        st.pyplot(fig, clear_figure=True)


def main():
    st.title("WP5 Segmentation Viewer — Fixed Runs")
    with st.sidebar:
        st.caption("Browsing fixed evaluation runs")
        st.write("Datalist:")
        st.code(str(DATALIST_PATH))
        for name, p in RUNS.items():
            st.write(f"{name} preds:")
            st.code(str(p))
        alpha = st.slider("Overlay alpha", 0.1, 1.0, 0.4, 0.05)
        st.markdown("---")
        # Disable Plotly option for now; keep Matplotlib fixed with a hover hint.
        st.radio("3D Renderer", options=["Matplotlib (static)"], index=0, horizontal=False, disabled=True, help="Plotly (interactive) will be implemented in the future.")
        renderer = "Matplotlib (static)"
        # Disable show-image toggle (Plotly-only feature) with a hover hint.
        show_img_vol = st.checkbox("3D: show underlying image volume (Plotly)", value=False, disabled=True, help="This Plotly overlay will be implemented in the future.")
        ds = st.slider("3D downsample (higher=faster)", 1, 6, 3)
        op = st.slider("3D surface opacity", 0.05, 0.95, 0.5, 0.05)

    if not DATALIST_PATH.exists():
        st.error(f"Datalist not found: {DATALIST_PATH}")
        st.stop()

    records = load_json(DATALIST_PATH)
    if not records:
        st.warning("Datalist is empty or could not be loaded.")
        st.stop()

    case_ids = sorted([r.get("id", f"case_{i}") for i, r in enumerate(records)])
    case_sel = st.selectbox("Select case", options=case_ids, index=0)
    rec = next(r for r in records if r.get("id") == case_sel)
    img_path = Path(rec["image"]) if rec.get("image") else None
    lbl_path = Path(rec["label"]) if rec.get("label") else None

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if img_path and img_path.exists():
            img_vol, _ = load_nii(img_path)
            st.write(f"Image: {img_path}")
            st.write(f"Shape: {img_vol.shape}")
        else:
            st.error("Image not found; check datalist paths.")
            st.stop()
    with col_b:
        st.write(f"Label: {lbl_path}")
        lbl_vol = None
        if lbl_path and lbl_path.exists():
            lbl_vol, _ = load_nii(lbl_path)

        # Resolve predictions for the four fixed runs
        pred_paths: Dict[str, Optional[Path]] = {}
        for run_name, run_dir in RUNS.items():
            pred_paths[run_name] = find_pred_path(run_dir, img_path, case_sel) if (img_path and run_dir.exists()) else None
            st.write(f"{run_name}: {pred_paths[run_name]}")

        # Load predictions volumes for each run
        pred_vols: Dict[str, Optional[np.ndarray]] = {k: None for k in RUNS.keys()}
        for run_name, pth in pred_paths.items():
            if pth and pth.exists():
                pv, _ = load_volume_any(pth)
                if pv.ndim == 4 and pv.shape[0] == 1:
                    pv = pv[0]
                pred_vols[run_name] = pv

        # Align shapes if needed (center pad/crop)
        if lbl_vol is not None and lbl_vol.shape != img_vol.shape:
            lbl_vol = center_pad_or_crop(lbl_vol, img_vol.shape)
        for k, v in list(pred_vols.items()):
            if v is not None and v.shape != img_vol.shape:
                pred_vols[k] = center_pad_or_crop(v, img_vol.shape)

    # 2D Slice View
    st.subheader("2D Slice View")
    axis = st.radio("Axis", options=["z", "y", "x"], index=0, horizontal=True, key="axis")
    dim = {"x": 0, "y": 1, "z": 2}[axis]
    size = int(img_vol.shape[dim])
    idx = st.slider("Slice index", 0, max(0, size - 1), size // 2, key="idx")

    # Arrange panels: GT | Fully | 1% Volume  /  1% Slice | 1% Voxel
    col_names_row1 = ["Ground Truth", "Fully Supervised", "1% Volume"]
    col_names_row2 = ["1% Slice", "1% Voxel"]

    row1 = st.columns(3)
    with row1[0]:
        st.caption(col_names_row1[0])
        draw_slice(img_vol, lbl_vol, None, axis, idx, alpha)
    with row1[1]:
        st.caption(col_names_row1[1])
        pv = pred_vols.get("Fully Supervised")
        draw_slice(img_vol, lbl_vol, pv, axis, idx, alpha)
    with row1[2]:
        st.caption(col_names_row1[2])
        pv = pred_vols.get("1% Volume")
        draw_slice(img_vol, lbl_vol, pv, axis, idx, alpha)

    row2 = st.columns(2)
    with row2[0]:
        st.caption(col_names_row2[0])
        pv = pred_vols.get("1% Slice")
        draw_slice(img_vol, lbl_vol, pv, axis, idx, alpha)
    with row2[1]:
        st.caption(col_names_row2[1])
        pv = pred_vols.get("1% Voxel")
        draw_slice(img_vol, lbl_vol, pv, axis, idx, alpha)

    # Per-class Dice comparison (selected case)
    if lbl_vol is not None:
        st.subheader("Per-Class Dice vs GT (0..4)")
        cols = st.columns(4)
        names = ["Fully Supervised", "1% Volume", "1% Slice", "1% Voxel"]
        for i, name in enumerate(names):
            with cols[i]:
                pv = pred_vols.get(name)
                st.write(name)
                if pv is not None:
                    d = dice_per_class(pv.astype(np.int64), lbl_vol.astype(np.int64))
                    st.json({str(k): float(v) for k, v in d.items()})
                else:
                    st.info("No prediction for this case.")

    # 3D View (GT vs Prediction)
    st.subheader("3D View (GT vs Prediction)")
    cls = st.selectbox("Class isosurface (1-4)", options=[1, 2, 3, 4], index=1, key="cls")
    tabs = st.tabs(["Fully Supervised", "1% Volume", "1% Slice", "1% Voxel"])
    for tab, name in zip(tabs, ["Fully Supervised", "1% Volume", "1% Slice", "1% Voxel"]):
        with tab:
            pv = pred_vols.get(name)
            if pv is None:
                st.info("No prediction available for this case.")
            else:
                if renderer.startswith("Plotly"):
                    plot_3d_pair(img_vol, lbl_vol, pv, cls=cls, downsample=ds, opacity=op, show_image=show_img_vol, right_title=name)
                else:
                    plot_3d_pair_matplotlib(lbl_vol, pv, cls=cls, downsample=ds, opacity=op, right_title=name)


if __name__ == "__main__":
    # Streamlit entry
    main()
