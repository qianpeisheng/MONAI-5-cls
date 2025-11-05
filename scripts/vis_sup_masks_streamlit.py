#!/usr/bin/env python3
"""
Streamlit 3D viewer for WP5 supervision masks (separate app).

Features
- Pick a sup_masks directory and a case id to visualize
- Load image/label (RAS) and overlay:
  - label class surfaces (0..4; 6 ignored by default)
  - seed points (from <id>_seedmask.npy)
  - supervised (dilated) region surface (from <id>_supmask.npy)
- Adjustable opacities and class visibility
- Basic stats (coverage, counts) from *_supmask_stats.json

Usage
  streamlit run scripts/vis_sup_masks_streamlit.py -- \
    --sup-dir /home/peisheng/MONAI/runs/sup_masks_0p1pct_global_d0_5_nov \
    --data-root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
    --datalist datalist_train.json

Notes
- Requires: streamlit, pyvista, numpy, nibabel or MONAI (for consistent RAS). Recommended: streamlit-pyvista
- If streamlit-pyvista is not installed, falls back to an off-screen screenshot render.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

# Optional MONAI for consistent RAS orientation
try:
    from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd
    _HAS_MONAI = True
except Exception:
    _HAS_MONAI = False

try:
    import nibabel as nib
    _HAS_NIB = True
except Exception:
    _HAS_NIB = False

import os
import pyvista as pv
# Start a virtual X server if no DISPLAY (mirrors working sample pattern)
try:
    pv.start_xvfb()  # provides DISPLAY for interactive rendering in headless sessions
except Exception:
    pass
# Speed/robustness: tolerate empty meshes and avoid costly shading by default
try:
    pv.global_theme.allow_empty_mesh = True
    pv.global_theme.smooth_shading = False
except Exception:
    pass


def _new_imagedata():
    """Create a VTK image data grid compatible with the installed PyVista version.

    Prefer ImageData; fall back to UniformGrid if needed.
    """
    if hasattr(pv, "ImageData"):
        return pv.ImageData()
    # Fallback for environments exposing UniformGrid only
    if hasattr(pv, "UniformGrid"):
        return pv.UniformGrid()
    raise AttributeError("PyVista installation lacks ImageData/UniformGrid. Please upgrade pyvista + vtk.")

_HAS_STPYVISTA = False
try:
    # pip install streamlit-pyvista
    from stpyvista import stpyvista  # type: ignore
    _HAS_STPYVISTA = True
except Exception:
    _HAS_STPYVISTA = False


# ---------- Caching helpers ----------
@st.cache_data(show_spinner=False)
def _cached_load_volume_ras(image_path: str, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
    return _load_volume_ras(image_path, label_path)


@st.cache_data(show_spinner=False)
def _cached_np_load(path: str) -> np.ndarray:
    return np.load(path)


@st.cache_data(show_spinner=False)
def _cached_json_load(path: str) -> Dict:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def _load_volume_ras(image_path: str, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load image/label into RAS orientation, channel-first arrays.

    Returns
    - image: (1,X,Y,Z) float32
    - label: (1,X,Y,Z) int64
    """
    if _HAS_MONAI:
        t = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ])
        d = t({"image": image_path, "label": label_path})
        img = d["image"].astype(np.float32)
        lbl = d["label"].astype(np.int64)
        return img, lbl
    # Fallback without MONAI: use nibabel if available, else naive load
    if _HAS_NIB:
        img_nii = nib.load(image_path)
        lbl_nii = nib.load(label_path)
        img = np.asarray(img_nii.get_fdata()).astype(np.float32)
        lbl = np.asarray(lbl_nii.get_fdata()).astype(np.int64)
        # Without MONAI Orientationd, assume as-is orientation
        if img.ndim == 3:
            img = img[None]
        if lbl.ndim == 3:
            lbl = lbl[None]
        return img, lbl
    # Last resort: attempt numpy load (not NIfTI)
    raise RuntimeError("Neither MONAI nor nibabel available to load NIfTI. Please install monai[all] or nibabel.")


def _pclip_zscore(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.0, eps: float = 1e-8) -> np.ndarray:
    """Robust per-sample normalization for visualization (clip to [p1,p99] then z-score)."""
    arr = x.astype(np.float32)
    flat = arr.reshape(-1)
    lo = np.percentile(flat, p_low)
    hi = np.percentile(flat, p_high)
    arr = np.clip(arr, lo, hi)
    mu = arr.mean()
    sigma = arr.std()
    return (arr - mu) / (sigma + eps)


def _discover_cases(sup_dir: Path) -> List[str]:
    ids = []
    for p in sorted(sup_dir.glob("*_supmask.npy")):
        b = p.name[:-12]  # strip _supmask.npy
        ids.append(b)
    return ids


def _case_paths_from_datalist(datalist_json: Optional[Path]) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    if not datalist_json or not datalist_json.exists():
        return mapping
    try:
        lst = json.loads(datalist_json.read_text())
        for rec in lst:
            cid = str(rec.get("id"))
            mapping[cid] = {"image": rec.get("image", ""), "label": rec.get("label", "")}
    except Exception:
        pass
    return mapping


def _case_paths_from_root(data_root: Optional[Path], case_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    if not data_root:
        return None, None
    data_dir = data_root / "data"
    ip = data_dir / f"{case_id}_image.nii"
    lp = data_dir / f"{case_id}_label.nii"
    return (ip if ip.exists() else None), (lp if lp.exists() else None)


def _build_seed_points(seed_mask: np.ndarray) -> Optional[pv.PolyData]:
    """Convert a (1,X,Y,Z) boolean seed mask to a PyVista point cloud for glyphing."""
    if seed_mask is None:
        return None
    m = seed_mask.astype(bool)
    if m.ndim == 4:
        m = m[0]
    idx = np.argwhere(m)
    if idx.size == 0:
        return None
    # PyVista expects points as (N, 3) float array
    pts = idx[:, [2, 1, 0]].astype(np.float32)  # order: Z,Y,X -> x,y,z
    cloud = pv.PolyData(pts)
    return cloud


def _surface_from_mask(mask: np.ndarray, iso: float = 0.5, decimate: float = 0.0, downsample: int = 1) -> Optional[pv.PolyData]:
    """Build an iso-surface mesh from a (X,Y,Z) or (1,X,Y,Z) binary mask.

    Uses cell_data then converts to point_data before contouring for compatibility.
    """
    vol = mask
    if vol.ndim == 4:
        vol = vol[0]
    if vol.sum() == 0:
        return None
    # Optional stride downsampling to reduce polygon count
    ds = max(1, int(downsample))
    if ds > 1:
        vol = vol[::ds, ::ds, ::ds]
    # Build ImageData grid using cell-centered scalars (dims = shape + 1)
    grid = _new_imagedata()
    X, Y, Z = vol.shape
    grid.dimensions = (Z + 1, Y + 1, X + 1)  # (nx, ny, nz) in x,y,z
    grid.spacing = (float(ds), float(ds), float(ds))
    grid.origin = (0.0, 0.0, 0.0)
    # Use cell_data for binary mask then convert to point_data
    grid.cell_data["values"] = vol.astype(np.float32).ravel(order="F")
    try:
        mesh = grid.cell_data_to_point_data()
        surf = mesh.contour(isosurfaces=[iso], scalars="values")
    except Exception:
        return None
    # Guard against empty polys
    if surf is None or getattr(surf, "n_points", 0) == 0:
        return None
    if decimate > 0.0 and getattr(surf, "n_points", 0) > 0:
        try:
            surf = surf.decimate_pro(decimate)
        except Exception:
            pass
    return surf


def _volume_actor(img: np.ndarray, opacity: float = 0.15) -> Tuple[pv.DataSet, Dict]:
    vol = img
    if vol.ndim == 4:
        vol = vol[0]
    vol = _pclip_zscore(vol)
    # ImageData with point-centered intensities for volume rendering
    grid = _new_imagedata()
    X, Y, Z = vol.shape
    grid.dimensions = (Z + 1, Y + 1, X + 1)
    grid.spacing = (1.0, 1.0, 1.0)
    grid.origin = (0.0, 0.0, 0.0)
    # store into cell_data then convert to point_data for stable add_volume behavior
    grid.cell_data["intensity"] = vol.ravel(order="F")
    mesh = grid.cell_data_to_point_data()
    add_kwargs = dict(scalars="intensity", cmap="gray", opacity=opacity)
    return mesh, add_kwargs


def _finalize_camera(plotter: pv.Plotter) -> None:
    """Set a reasonable default view and ensure bounds are framed.

    Uses isometric view and resets camera/clipping to fit all actors.
    """
    try:
        plotter.add_axes()
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.reset_camera_clipping_range()
    except Exception:
        pass


def _render_scene(
    image: Optional[np.ndarray],
    label: Optional[np.ndarray],
    seed_mask: Optional[np.ndarray],
    sup_mask: Optional[np.ndarray],
    show_volume: bool,
    show_labels: bool,
    show_sup: bool,
    show_unlabeled: bool,
    unlabeled_max_points: int,
    show_seeds: bool,
    class_vis: List[int],
    vol_opacity: float,
    sup_opacity: float,
    decimate: float,
    ds_surfaces: int,
    scene_key: str,
):
    # With Xvfb started above, we can render interactively (stpyvista) without forcing off-screen
    plotter = pv.Plotter(off_screen=False, window_size=(1024, 768))
    plotter.set_background("white")
    # Consistent per-class colors (0..4); 6 is ignored
    class_colors = {
        0: "#b0b0b0",  # background
        1: "#ff3b30",  # red
        2: "#ff9500",  # orange
        3: "#007aff",  # blue
        4: "#5856d6",  # indigo
    }

    # Track which classes are visible to assemble a legend later
    present_classes: set[int] = set()

    # Volume
    if show_volume and image is not None:
        grid, add_kwargs = _volume_actor(image, opacity=vol_opacity)
        plotter.add_volume(grid, **add_kwargs)

    # Supervised region surface
    if show_sup and sup_mask is not None:
        surf = _surface_from_mask(sup_mask, iso=0.5, decimate=decimate, downsample=ds_surfaces)
        if surf is not None and getattr(surf, "n_points", 0) > 0:
            plotter.add_mesh(surf, color="#34c759", opacity=sup_opacity, name="sup")

    # Label class surfaces
    if show_labels and label is not None:
        lbl = label
        if lbl.ndim == 4:
            lbl = lbl[0]
        for c in class_vis:
            if c not in class_colors:
                continue
            if c == 6:
                continue
            m = (lbl == c)
            if m.sum() == 0:
                continue
            surf = _surface_from_mask(m, iso=0.5, decimate=decimate, downsample=ds_surfaces)
            if surf is not None and getattr(surf, "n_points", 0) > 0:
                plotter.add_mesh(surf, color=class_colors[c], opacity=0.7 if c != 0 else 0.2, name=f"cls{c}")
                present_classes.add(c)

    # Seed points (color by underlying GT class when label is available)
    if show_seeds and seed_mask is not None:
        if label is not None:
            lblv = label[0] if label.ndim == 4 else label
            seeds = seed_mask[0] if seed_mask.ndim == 4 else seed_mask
            scale = float(int(ds_surfaces)) if int(ds_surfaces) > 1 else 1.0
            for c in class_vis:
                if c == 6:
                    continue
                coords = np.argwhere(seeds & (lblv == c))
                if coords.size == 0:
                    continue
                pts = coords[:, [2, 1, 0]].astype(np.float32) * scale
                cloud = pv.PolyData(pts)
                plotter.add_points(
                    cloud,
                    color=class_colors.get(c, "white"),
                    render_points_as_spheres=True,
                    point_size=6,
                    name=f"seeds_c{c}",
                )
                present_classes.add(c)
        else:
            cloud = _build_seed_points(seed_mask)
            if cloud is not None and cloud.n_points > 0:
                try:
                    if int(ds_surfaces) > 1:
                        cloud.points *= float(int(ds_surfaces))
                except Exception:
                    pass
                plotter.add_points(cloud, color="#ff2d55", render_points_as_spheres=True, point_size=6, name="seeds")

    # Unlabeled voxels (sampled point cloud)
    if show_unlabeled and sup_mask is not None:
        sup = sup_mask
        if sup.ndim == 4:
            sup = sup[0]
        valid = np.ones_like(sup, dtype=bool)
        if label is not None:
            lblv = label[0] if label.ndim == 4 else label
            valid &= (lblv != 6)
        unl = valid & (~sup.astype(bool))
        if unl.any():
            coords = np.argwhere(unl)
            if coords.shape[0] > unlabeled_max_points:
                sel = np.random.default_rng(1234).choice(coords.shape[0], size=unlabeled_max_points, replace=False)
                coords = coords[sel]
            if coords.size > 0:
                scale = float(int(ds_surfaces)) if int(ds_surfaces) > 1 else 1.0
                pts = coords[:, [2, 1, 0]].astype(np.float32) * scale
                cloud = pv.PolyData(pts)
                plotter.add_points(cloud, color="#8e8e93", render_points_as_spheres=True, point_size=3, name="unlabeled")

    # Legend for class colors and supervised region
    legend_entries: list[tuple[str, str]] = []
    for c in sorted(present_classes):
        if c in class_colors:
            label_txt = f"Class {c}" if c != 0 else "Class 0 (bg)"
            legend_entries.append((label_txt, class_colors[c]))
    if show_sup:
        legend_entries.append(("Supervised", "#34c759"))
    if legend_entries:
        try:
            plotter.add_legend(legend_entries)
        except Exception:
            pass

    # Set a good default camera after adding actors
    _finalize_camera(plotter)

    if _HAS_STPYVISTA:
        try:
            stpyvista(plotter, key=scene_key)
        except Exception as e:
            st.warning(f"Interactive embed failed ({e}); showing static image.")
            img = plotter.screenshot(return_img=True)
            st.image(img, caption="Static render (interactive backend unavailable)")
    else:
        # Fallback to static screenshot
        img = plotter.screenshot(return_img=True)
        st.image(img, caption="Static render (install streamlit-pyvista for interactivity)")

    return legend_entries


def _load_stats(stats_path: Path) -> Dict:
    try:
        return json.loads(stats_path.read_text())
    except Exception:
        return {}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sup-dir", type=str, default="", help="Directory containing <id>_supmask.npy files")
    ap.add_argument("--data-root", type=str, default="", help="WP5 root containing data/ with NIfTI pairs")
    ap.add_argument("--datalist", type=str, default="", help="Optional datalist JSON mapping ids -> image/label paths")
    return ap.parse_args()


def main():
    args = parse_args()
    st.set_page_config(page_title="WP5 Sup-Masks 3D Viewer", layout="wide")
    st.title("WP5 Supervision Masks — 3D Viewer (PyVista)")

    # Sidebar: config (lock sup_masks directory to a fixed path)
    st.sidebar.header("Configuration")
    SUP_DIR_DEFAULT = Path("/home/peisheng/MONAI/runs/sup_masks_0p1pct_global_d0_5_nov")
    sup_dir = SUP_DIR_DEFAULT if SUP_DIR_DEFAULT.exists() else None
    st.sidebar.info(f"sup_masks directory (locked): {SUP_DIR_DEFAULT}")
    sup_info = {}
    if sup_dir is not None and (sup_dir / "sup_masks_config.json").exists():
        try:
            sup_info = json.loads((sup_dir / "sup_masks_config.json").read_text())
        except Exception:
            sup_info = {}
    if sup_info:
        st.sidebar.caption(f"Config: {sup_info.get('config', {})}")

    # Keep data_root and datalist configurable (these do not leak mask dir control)
    data_root = Path(args.data_root) if args.data_root else None
    data_root_str = st.sidebar.text_input("data_root (for image/label)", value=str(data_root) if data_root else "")
    data_root = Path(data_root_str) if data_root_str else None

    datalist = Path(args.datalist) if args.datalist else None
    datalist_str = st.sidebar.text_input("datalist JSON (optional)", value=str(datalist) if datalist else "")
    datalist = Path(datalist_str) if datalist_str else None

    if sup_dir is None:
        st.warning("Select a valid sup_masks directory to begin.")
        return

    ids = _discover_cases(sup_dir)
    if not ids:
        st.error("No *_supmask.npy files found in the selected directory.")
        return

    # Map ids to image/label paths via datalist if provided
    idmap = _case_paths_from_datalist(datalist)

    st.sidebar.header("Case Selection")
    sel_id = st.sidebar.selectbox("case id", options=ids, index=0)

    # Resolve image/label paths
    img_path = None
    lbl_path = None
    if sel_id in idmap:
        rec = idmap[sel_id]
        ip = Path(rec.get("image", "")) if rec.get("image") else None
        lp = Path(rec.get("label", "")) if rec.get("label") else None
        if ip and ip.exists():
            img_path = ip
        if lp and lp.exists():
            lbl_path = lp
    if img_path is None or lbl_path is None:
        ip2, lp2 = _case_paths_from_root(data_root, sel_id)
        img_path = img_path or ip2
        lbl_path = lbl_path or lp2

    col_l, col_r = st.columns([3, 1])
    with col_r:
        st.subheader("Overlays")
        # Faster defaults: no volume, no label surfaces; show supervised region and seeds
        show_volume = st.checkbox("Show image volume", value=False)
        show_labels = st.checkbox("Show label surfaces", value=False)
        show_sup = st.checkbox("Show supervised region", value=True)
        show_seeds = st.checkbox("Show seed points", value=True)
        show_unlabeled = st.checkbox("Show unlabeled (sampled points)", value=False)
        class_vis = st.multiselect("Classes (0..4)", options=[0, 1, 2, 3, 4], default=[1, 2, 3, 4], key="class_sel")
        vol_opacity = st.slider("Volume opacity", min_value=0.0, max_value=1.0, value=0.10, step=0.05)
        sup_opacity = st.slider("Supervised region opacity", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
        decimate = st.slider("Surface decimation (0..0.9)", min_value=0.0, max_value=0.9, value=0.6, step=0.05)
        ds_surfaces = st.slider("Surface downsample (stride)", min_value=1, max_value=4, value=3, step=1)
        unlabeled_max_points = st.slider("Unlabeled sample (points)", min_value=1000, max_value=200000, value=10000, step=1000)
        if show_volume and ds_surfaces > 1:
            st.caption("Hint: Volume uses native spacing; set Surface downsample=1 for exact overlay with volume.")

    # Load data
    img = None
    lbl = None
    if img_path and lbl_path:
        try:
            img, lbl = _cached_load_volume_ras(str(img_path), str(lbl_path))
        except Exception as e:
            st.error(f"Failed to load volumes: {e}")

    # Load masks
    seed_np: Optional[np.ndarray] = None
    sup_np: Optional[np.ndarray] = None
    stats = {}
    sp = sup_dir / f"{sel_id}_seedmask.npy"
    su = sup_dir / f"{sel_id}_supmask.npy"
    sj = sup_dir / f"{sel_id}_supmask_stats.json"
    if sp.exists():
        try:
            seed_np = _cached_np_load(str(sp))
        except Exception:
            seed_np = None
    if su.exists():
        try:
            sup_np = _cached_np_load(str(su))
        except Exception:
            sup_np = None
    if sj.exists():
        stats = _cached_json_load(str(sj))

    with col_l:
        if img is None or lbl is None:
            st.warning("Image/Label not found. Provide a valid data_root or datalist.")
        # Build a unique key so Streamlit replaces the 3D widget when controls change
        scene_key = (
            f"pv_scene|cv={'-'.join(map(str,sorted(class_vis)))}|vol={int(show_volume)}|lab={int(show_labels)}|sup={int(show_sup)}|seeds={int(show_seeds)}|ds={int(ds_surfaces)}|dec={decimate:.2f}"
        )
        legend_entries = _render_scene(
            image=img,
            label=lbl,
            seed_mask=seed_np,
            sup_mask=sup_np,
            show_volume=show_volume,
            show_labels=show_labels,
            show_sup=show_sup,
            show_unlabeled=show_unlabeled,
            unlabeled_max_points=int(unlabeled_max_points),
            show_seeds=show_seeds,
            class_vis=class_vis,
            vol_opacity=vol_opacity,
            sup_opacity=sup_opacity,
            decimate=decimate,
            ds_surfaces=int(ds_surfaces),
            scene_key=scene_key,
        )
        # Always show full legend mapping for clarity
        st.subheader("Legend")
        legend_full = [
            ("Class 0 (bg)", "#b0b0b0"),
            ("Class 1", "#ff3b30"),
            ("Class 2", "#ff9500"),
            ("Class 3", "#007aff"),
            ("Class 4", "#5856d6"),
            ("Supervised", "#34c759"),
        ]
        for label_txt, color in legend_full:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;'>"
                f"<div style='width:14px;height:14px;background:{color};border:1px solid #666;'></div>"
                f"<span>{label_txt}</span></div>",
                unsafe_allow_html=True,
            )

    # Stats panel
    st.subheader("Case Stats")
    cols = st.columns(3)
    with cols[0]:
        st.write(f"case id: {sel_id}")
        if stats:
            st.json(stats)
        else:
            # derive basic coverage if possible
            try:
                if sup_np is not None:
                    frac = float(sup_np.astype(np.float32).mean())
                    st.write({"sup_fraction": frac})
            except Exception:
                pass


if __name__ == "__main__":
    main()
