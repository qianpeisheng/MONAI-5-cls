#!/usr/bin/env python3
"""
Streamlit app: Compare GT labels vs Supervoxel-labeled (voted) voxel labels.

Features
- Case selector from a datalist JSON ({image,label,id})
- 2D slice view: side-by-side GT and SV-labeled overlays
- 3D view: per-class surfaces in GT vs SV-labeled
  - Matplotlib (static) using scikit-image marching cubes
  - PyVista (interactive) with optional volume overlay and decimation
- Per-case Dice scores over classes 0..4 with label 6 ignored

Assumptions
- SV run folder contains per-voxel labels at `<sv_dir>/<id>_labels.npy` (as in
  /home/peisheng/MONAI/runs/sv_fullgt_5k_ras2_voted). These are already
  voxel-wise labels derived via supervoxel voting; no reconstruction needed.
- Image/GT labels are read from the datalist and canonicalized to RAS
  orientation to match MONAI Orientationd(axcodes='RAS').

Usage
  python3 -m streamlit run scripts/vis_gt_vs_sv_streamlit.py -- \
    --sv-dir /home/peisheng/MONAI/runs/sv_fullgt_5k_ras2_voted \
    --datalist datalist_test.json

Optional
  - To show only cases with available SV-labeled volumes, toggle the sidebar option.
  - For interactive 3D, install pyvista and streamlit-pyvista.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

# Optional deps
try:
    import nibabel as nib  # type: ignore
    _HAS_NIB = True
except Exception:
    _HAS_NIB = False

try:
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False

try:
    from skimage import measure as skmeasure  # type: ignore
    _HAS_SKIMG = True
except Exception:
    _HAS_SKIMG = False

try:
    import pyvista as pv  # type: ignore
    _HAS_PV = True
    try:
        pv.start_xvfb()
    except Exception:
        pass
    try:
        pv.global_theme.allow_empty_mesh = True
        pv.global_theme.smooth_shading = False
    except Exception:
        pass
except Exception:
    _HAS_PV = False

_HAS_STPV = False
if _HAS_PV:
    try:
        from stpyvista import stpyvista  # type: ignore
        _HAS_STPV = True
    except Exception:
        _HAS_STPV = False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sv-dir", type=str, default="/home/peisheng/MONAI/runs/sv_fullgt_5k_ras2_voted", help="Directory with <id>_labels.npy from voted supervoxels")
    ap.add_argument("--datalist", type=str, default="datalist_train.json", help="JSON with records of {image,label,id}")
    return ap.parse_args()


@st.cache_data(show_spinner=False)
def load_json_list(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def load_nii_ras(path: str) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    if not _HAS_NIB:
        raise RuntimeError("nibabel is required to read NIfTI. pip install nibabel")
    img = nib.load(path)
    try:
        img = nib.as_closest_canonical(img)
    except Exception:
        pass
    arr = np.asarray(img.get_fdata())
    if arr.ndim == 3:
        arr = arr
    elif arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    # spacing from affine column norms (dx,dy,dz)
    A = getattr(img, "affine", None)
    if A is None:
        spac = (1.0, 1.0, 1.0)
    else:
        spac = tuple(float(np.linalg.norm(A[:3, i])) for i in range(3))
    return arr, (spac[0], spac[1], spac[2])


@st.cache_data(show_spinner=False)
def load_npy(path: str) -> np.ndarray:
    return np.load(path)


def robust_minmax(vol: np.ndarray, pmin=1.0, pmax=99.0) -> Tuple[float, float]:
    flat = vol.reshape(-1)
    lo = float(np.percentile(flat, pmin))
    hi = float(np.percentile(flat, pmax))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float(np.min(vol)), float(np.max(vol))
    return lo, hi


def center_pad_or_crop(vol: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
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
        return np.pad(v, pad_width, mode='constant', constant_values=0)

    out = _crop_to(vol, target_shape)
    out = _pad_to(out, target_shape)
    return out


def colorize_seg2d(seg2d: np.ndarray, alpha: float = 0.4) -> np.ndarray:
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


def dice_per_class(pred: np.ndarray, gt: np.ndarray, classes=(0, 1, 2, 3, 4), ignore_label: Optional[int] = 6) -> Dict[int, float]:
    out: Dict[int, float] = {}
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


def iou_per_class(pred: np.ndarray, gt: np.ndarray, classes=(0, 1, 2, 3, 4), ignore_label: Optional[int] = 6) -> Dict[int, float]:
    out: Dict[int, float] = {}
    if ignore_label is not None:
        mask = gt != ignore_label
        gt = gt[mask]
        pred = pred[mask]
    for c in classes:
        p = pred == c
        g = gt == c
        inter = float(np.logical_and(p, g).sum())
        union = float(np.logical_or(p, g).sum())
        out[c] = (inter / union) if union > 0 else (1.0 if p.sum() == g.sum() else 0.0)
    return out


def draw_slice_pair(
    image: np.ndarray,
    gt: Optional[np.ndarray],
    sv: Optional[np.ndarray],
    axis: str,
    index: int,
    alpha: float,
    show_diff: bool = False,
    ignore_label: Optional[int] = 6,
):
    if not _HAS_PLT:
        st.warning("matplotlib not installed; 2D view disabled. pip install matplotlib")
        return
    if axis == "x":
        img2d = image[index, :, :]
        gt2d = gt[index, :, :] if gt is not None else None
        sv2d = sv[index, :, :] if sv is not None else None
    elif axis == "y":
        img2d = image[:, index, :]
        gt2d = gt[:, index, :] if gt is not None else None
        sv2d = sv[:, index, :] if sv is not None else None
    else:  # z
        img2d = image[:, :, index]
        gt2d = gt[:, :, index] if gt is not None else None
        sv2d = sv[:, :, index] if sv is not None else None

    img2d = np.flipud(np.rot90(img2d, k=1))
    if gt2d is not None:
        gt2d = np.flipud(np.rot90(gt2d, k=1))
    if sv2d is not None:
        sv2d = np.flipud(np.rot90(sv2d, k=1))

    vmin, vmax = robust_minmax(img2d)
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        ax1.set_title("Ground Truth")
        ax1.imshow(img2d.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        if gt2d is not None:
            ax1.imshow(colorize_seg2d(gt2d.T, alpha=alpha), origin="lower")
        ax1.axis("off")
        st.pyplot(fig1, clear_figure=True)
    with col2:
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
        ax2.set_title("SV-Labeled (voted)")
        ax2.imshow(img2d.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        if sv2d is not None:
            ax2.imshow(colorize_seg2d(sv2d.T, alpha=alpha), origin="lower")
            if show_diff and gt2d is not None:
                mask = (sv2d != gt2d)
                if ignore_label is not None:
                    mask &= (gt2d != ignore_label)
                diff = np.zeros((*sv2d.shape, 4), dtype=np.float32)
                diff[mask, 0] = 1.0  # magenta
                diff[mask, 2] = 1.0
                diff[mask, 3] = 0.8
                ax2.imshow(diff.T, origin="lower")
        ax2.axis("off")
        st.pyplot(fig2, clear_figure=True)


def extract_mesh_from_mask(mask3d: np.ndarray, step: int = 2) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not _HAS_SKIMG:
        return None
    m = (mask3d.astype(np.uint8) > 0)
    if int(m.sum()) == 0:
        return None
    m = np.pad(m, 1, mode='constant', constant_values=0)
    vol = m.astype(np.float32)
    step_sz = max(1, int(step))
    try:
        verts, faces, _, _ = skmeasure.marching_cubes(vol, level=0.5, step_size=step_sz)
        return verts, faces
    except Exception:
        return None


def plot_3d_pair_matplotlib(gt: Optional[np.ndarray], sv: Optional[np.ndarray], cls: int, downsample: int = 2, opacity: float = 0.5):
    if not _HAS_PLT:
        st.warning("matplotlib not installed; 3D view disabled. pip install matplotlib")
        return
    if not _HAS_SKIMG:
        st.warning("scikit-image not installed; 3D view disabled. pip install scikit-image")
        return
    def ds(v):
        s = max(1, int(downsample))
        return v[::s, ::s, ::s]
    col1, col2 = st.columns(2)
    with col1:
        fig = plt.figure(figsize=(5.5, 5.0))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"GT class {cls}")
        if gt is not None:
            gmask = (gt == cls)
            if gmask.any():
                mesh = extract_mesh_from_mask(ds(gmask), step=max(1, int(downsample)))
                if mesh is not None:
                    verts, faces = mesh
                    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color='green', linewidth=0.1, antialiased=True, alpha=opacity)
                else:
                    idx = np.column_stack(np.nonzero(ds(gmask)))
                    if idx.shape[0] > 0:
                        sel = idx if idx.shape[0] <= 5000 else idx[np.random.RandomState(0).permutation(idx.shape[0])[:5000]]
                        ax.scatter(sel[:, 0], sel[:, 1], sel[:, 2], s=1, c='green', alpha=min(0.8, opacity+0.2))
        ax.set_axis_off()
        st.pyplot(fig, clear_figure=True)
    with col2:
        fig = plt.figure(figsize=(5.5, 5.0))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"SV class {cls}")
        if sv is not None:
            smask = (sv == cls)
            if smask.any():
                mesh = extract_mesh_from_mask(ds(smask), step=max(1, int(downsample)))
                if mesh is not None:
                    verts, faces = mesh
                    ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], color='red', linewidth=0.1, antialiased=True, alpha=opacity)
                else:
                    idx = np.column_stack(np.nonzero(ds(smask)))
                    if idx.shape[0] > 0:
                        sel = idx if idx.shape[0] <= 5000 else idx[np.random.RandomState(0).permutation(idx.shape[0])[:5000]]
                        ax.scatter(sel[:, 0], sel[:, 1], sel[:, 2], s=1, c='red', alpha=min(0.8, opacity+0.2))
        ax.set_axis_off()
        st.pyplot(fig, clear_figure=True)


def _new_imagedata():
    if not _HAS_PV:
        return None
    if hasattr(pv, "ImageData"):
        return pv.ImageData()
    if hasattr(pv, "UniformGrid"):
        return pv.UniformGrid()
    return None


def _pv_volume_actor(img: np.ndarray, spacing_xyz: Tuple[float, float, float], opacity: float = 0.15):
    grid = _new_imagedata()
    if grid is None:
        return None, {}
    vol = img.astype(np.float32)
    lo, hi = robust_minmax(vol)
    vol = (vol - lo) / (hi - lo + 1e-6)
    X, Y, Z = vol.shape
    dx, dy, dz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
    grid.dimensions = (X + 1, Y + 1, Z + 1)
    grid.spacing = (dx, dy, dz)
    grid.origin = (0.0, 0.0, 0.0)
    grid.cell_data["intensity"] = vol.ravel(order="F")
    mesh = grid.cell_data_to_point_data()
    add_kwargs = dict(scalars="intensity", cmap="gray", opacity=float(opacity))
    return mesh, add_kwargs


def _pv_surface_from_mask(mask: np.ndarray, spacing_xyz: Tuple[float, float, float], decimate: float = 0.2, ds: int = 1):
    grid = _new_imagedata()
    if grid is None:
        return None
    vol = mask.astype(np.uint8)
    if ds > 1:
        vol = vol[::ds, ::ds, ::ds]
    if int(vol.sum()) == 0:
        return None
    X, Y, Z = vol.shape
    dx, dy, dz = float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])
    grid.dimensions = (X + 1, Y + 1, Z + 1)
    grid.spacing = (dx * ds, dy * ds, dz * ds)
    grid.origin = (0.0, 0.0, 0.0)
    grid.cell_data["values"] = vol.astype(np.float32).ravel(order="F")
    try:
        mesh = grid.cell_data_to_point_data()
        surf = mesh.contour(isosurfaces=[0.5], scalars="values")
    except Exception:
        return None
    if surf is None or getattr(surf, "n_points", 0) == 0:
        return None
    if decimate > 0.0:
        try:
            ds_surf = surf.decimate_pro(float(decimate))
            if ds_surf is not None and getattr(ds_surf, "n_points", 0) >= max(100, int(0.05 * surf.n_points)):
                surf = ds_surf
        except Exception:
            pass
    return surf


def plot_3d_pair_pyvista(image: Optional[np.ndarray], spacing_xyz: Tuple[float, float, float], gt: Optional[np.ndarray], sv: Optional[np.ndarray], cls: int, ds: int, decimate: float, opacity: float, show_volume: bool, key_prefix: str = "pv_pair"):
    if not _HAS_PV:
        st.info("PyVista not available. pip install pyvista streamlit-pyvista")
        return
    plot_left = pv.Plotter(off_screen=False, window_size=(768, 576))
    plot_right = pv.Plotter(off_screen=False, window_size=(768, 576))
    plot_left.set_background("white"); plot_right.set_background("white")
    if show_volume and image is not None:
        mesh, add_kwargs = _pv_volume_actor(image, spacing_xyz=spacing_xyz, opacity=0.12)
        if mesh is not None:
            plot_left.add_volume(mesh, **add_kwargs)
            plot_right.add_volume(mesh, **add_kwargs)
    if gt is not None:
        m_gt = (gt == cls)
        surf_gt = _pv_surface_from_mask(m_gt, spacing_xyz=spacing_xyz, decimate=decimate, ds=int(ds))
        if surf_gt is not None:
            plot_left.add_mesh(surf_gt, color="#4daf4a", opacity=float(opacity), name=f"gt_cls{cls}")
    if sv is not None:
        m_sv = (sv == cls)
        surf_sv = _pv_surface_from_mask(m_sv, spacing_xyz=spacing_xyz, decimate=decimate, ds=int(ds))
        if surf_sv is not None:
            plot_right.add_mesh(surf_sv, color="#e41a1c", opacity=float(opacity), name=f"sv_cls{cls}")
    try:
        for p in (plot_left, plot_right):
            p.add_axes(); p.view_isometric(); p.reset_camera(); p.reset_camera_clipping_range()
    except Exception:
        pass
    col1, col2 = st.columns(2)
    with col1:
        if _HAS_STPV:
            stpyvista(plot_left, key=f"{key_prefix}_L|cls={cls}|ds={int(ds)}|dec={decimate:.2f}")
        else:
            img = plot_left.screenshot(return_img=True)
            st.image(img, caption="GT 3D (install streamlit-pyvista for interactivity)")
    with col2:
        if _HAS_STPV:
            stpyvista(plot_right, key=f"{key_prefix}_R|cls={cls}|ds={int(ds)}|dec={decimate:.2f}")
        else:
            img = plot_right.screenshot(return_img=True)
            st.image(img, caption="SV 3D (install streamlit-pyvista for interactivity)")


def main() -> None:
    args = parse_args()
    st.title("GT vs Supervoxel-Labeled (voted) — WP5 Viewer")
    with st.sidebar:
        st.caption("Inputs")
        sv_dir = st.text_input("SV-labeled dir", value=str(args.sv_dir))
        datalist = st.text_input("Datalist JSON", value=str(args.datalist))
        only_avail = st.checkbox("Show only cases with SV-labeled available", value=True)
        alpha = st.slider("Overlay alpha", 0.1, 1.0, 0.4, 0.05)
        st.markdown("---")
        renderer = st.radio("3D Renderer", options=["Matplotlib (static)", "PyVista (interactive)"] if _HAS_PV else ["Matplotlib (static)"], index=1 if _HAS_PV else 0)
        cls3d = st.selectbox("3D class", options=[1, 2, 3, 4], index=1)
        ds3d = st.slider("3D downsample (higher=faster)", 1, 6, 3)
        op3d = st.slider("3D surface opacity", 0.05, 0.95, 0.5, 0.05)
        dec3d = st.slider("PyVista decimate (0-0.9)", 0.0, 0.9, 0.2, 0.05, disabled=(renderer != "PyVista (interactive)"))
        show_vol = st.checkbox("3D: show image volume (PyVista)", value=False, disabled=(renderer != "PyVista (interactive)"))

    records = load_json_list(datalist)
    if not records:
        st.error(f"Datalist not found or empty: {datalist}")
        st.stop()

    # Discover available SV ids
    sv_path = Path(sv_dir)
    sv_ids_available = set()
    if sv_path.exists():
        for p in sv_path.glob("*_labels.npy"):
            name = p.name
            if name.endswith("_labels.npy"):
                sv_ids_available.add(name[: -len("_labels.npy")])

    # Build initial case list
    dl_ids_all = [r.get("id", "") for r in records]
    dl_ids_all = [c for c in dl_ids_all if c]
    case_ids = sorted([c for c in dl_ids_all if (c in sv_ids_available)]) if only_avail else sorted(dl_ids_all)

    # Fallback: if filtered set is empty, try switching to datalist_train.json automatically
    if not case_ids:
        fallback = Path(datalist).with_name("datalist_train.json")
        if fallback.exists():
            records_fb = load_json_list(str(fallback))
            dl_ids_all_fb = [r.get("id", "") for r in records_fb]
            dl_ids_all_fb = [c for c in dl_ids_all_fb if c]
            case_ids_fb = sorted([c for c in dl_ids_all_fb if (c in sv_ids_available)]) if only_avail else sorted(dl_ids_all_fb)
            if case_ids_fb:
                st.info(f"No overlap between SV dir and {datalist}. Using {fallback} instead.")
                records = records_fb
                dl_ids_all = dl_ids_all_fb
                case_ids = case_ids_fb
                datalist = str(fallback)
        # If still empty, disable availability filter implicitly and warn
        if not case_ids and dl_ids_all:
            st.warning("No cases match SV-labeled availability. Showing all datalist cases; SV labels may be missing for some.")
            case_ids = sorted(dl_ids_all)
    if not case_ids:
        st.error("No cases available to display. Check datalist and SV directory paths.")
        st.stop()

    case_sel = st.selectbox("Select case", options=case_ids, index=0)
    rec = next(r for r in records if r.get("id") == case_sel)
    img_path = rec.get("image", "")
    gt_path = rec.get("label", "")

    # Load image and GT
    try:
        img_vol, spacing = load_nii_ras(img_path)
        gt_vol, _ = load_nii_ras(gt_path)
    except Exception as e:
        st.error(f"Failed to load NIfTI: {e}")
        st.stop()

    # Load SV voxel labels
    sv_lab_path = Path(sv_dir) / f"{case_sel}_labels.npy"
    if not sv_lab_path.exists():
        st.error(f"SV-labeled file not found: {sv_lab_path}")
        st.stop()
    try:
        sv_vol = load_npy(str(sv_lab_path))
    except Exception as e:
        st.error(f"Failed to load SV labels: {e}")
        st.stop()
    if sv_vol.ndim == 4 and sv_vol.shape[0] == 1:
        sv_vol = sv_vol[0]

    # Align shapes
    tshape = img_vol.shape
    if gt_vol.shape != tshape:
        gt_vol = center_pad_or_crop(gt_vol, tshape)
    if sv_vol.shape != tshape:
        sv_vol = center_pad_or_crop(sv_vol, tshape)

    # 2D view controls
    st.subheader("2D Slice View")
    axis = st.radio("Axis", options=["z", "y", "x"], index=0, horizontal=True)
    dim = {"x": 0, "y": 1, "z": 2}[axis]
    max_idx = int(img_vol.shape[dim])
    sidx = st.slider("Slice index", 0, max(0, max_idx - 1), max_idx // 2)
    show_diff = st.checkbox("Show mismatch overlay (magenta)", value=False)
    draw_slice_pair(img_vol.astype(np.float32), gt_vol.astype(np.int64), sv_vol.astype(np.int64), axis=axis, index=int(sidx), alpha=alpha, show_diff=show_diff)

    # Per-case metrics
    st.subheader("Per-Class Metrics (ignore label 6)")
    d = dice_per_class(sv_vol.astype(np.int64), gt_vol.astype(np.int64))
    j = iou_per_class(sv_vol.astype(np.int64), gt_vol.astype(np.int64))
    colm = st.columns(2)
    with colm[0]:
        st.caption("Dice (0..4)")
        st.json({str(k): float(v) for k, v in d.items()})
    with colm[1]:
        st.caption("IoU (0..4)")
        st.json({str(k): float(v) for k, v in j.items()})

    # 3D view
    st.subheader("3D View (GT vs SV)")
    if renderer.startswith("PyVista"):
        plot_3d_pair_pyvista(image=img_vol, spacing_xyz=spacing, gt=gt_vol, sv=sv_vol, cls=int(cls3d), ds=int(ds3d), decimate=float(dec3d), opacity=float(op3d), show_volume=bool(show_vol))
    else:
        plot_3d_pair_matplotlib(gt=gt_vol, sv=sv_vol, cls=int(cls3d), downsample=int(ds3d), opacity=float(op3d))


if __name__ == "__main__":
    main()
