#!/usr/bin/env python3
"""
Streamlit 3D/2D viewer for unlabeled supervoxels (sv_ids) — boundary-first.

Features
- Pick an sv_ids directory and a case id to visualize
- Load image/label (RAS) to get spacing; optional volume overlay
- 3D boundary mesh from sv_ids (thin mask -> iso-surface)
- Optional: sample a subset of supervoxels and render semi-transparent surfaces
- 2D slice viewer for sv_ids (colorized)

Usage
  python -m streamlit run scripts/vis_sv_ids_streamlit.py -- \
    --sv-dir /home/peisheng/MONAI/runs/sv_fill_5k_nofill_ras2 \
    --data-root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
    --datalist datalist_train.json

Notes
- Assumes sv_ids are RAS-aligned. A toggle is provided to reorient using label
  header if needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import streamlit as st

try:
    import nibabel as nib
    from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
    _HAS_NIB = True
except Exception:
    _HAS_NIB = False

try:
    from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd
    _HAS_MONAI = True
except Exception:
    _HAS_MONAI = False

import pyvista as pv

try:
    pv.start_xvfb()
except Exception:
    pass

try:
    pv.global_theme.allow_empty_mesh = True
    pv.global_theme.smooth_shading = False
except Exception:
    pass


def _neighbor_arrays3(lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return 6 neighbor arrays (x+,-, y+,-, z+,-) with edge replication to keep shape."""
    X, Y, Z = lab.shape
    nxp = np.empty_like(lab); nxp[:-1, :, :] = lab[1:, :, :]; nxp[-1, :, :] = lab[-1, :, :]
    nxm = np.empty_like(lab); nxm[1:, :, :] = lab[:-1, :, :]; nxm[0, :, :] = lab[0, :, :]
    nyp = np.empty_like(lab); nyp[:, :-1, :] = lab[:, 1:, :]; nyp[:, -1, :] = lab[:, -1, :]
    nym = np.empty_like(lab); nym[:, 1:, :] = lab[:, :-1, :]; nym[:, 0, :] = lab[:, 0, :]
    nzp = np.empty_like(lab); nzp[:, :, :-1] = lab[:, :, 1:]; nzp[:, :, -1] = lab[:, :, -1]
    nzm = np.empty_like(lab); nzm[:, :, 1:] = lab[:, :, :-1]; nzm[:, :, 0] = lab[:, :, 0]
    return nxp, nxm, nyp, nym, nzp, nzm


def _label_boundary_class_masks(labels: np.ndarray, prefer_nonzero: bool = True, thick: int = 1) -> Dict[int, np.ndarray]:
    """Compute class-specific boundary masks for a labeled 3D volume.

    - Boundary voxels are where any 6-neighbor differs.
    - Assign each boundary voxel a class based on neighbors: prefer non-zero neighbor; else 0.
    - Returns dict: class -> boolean mask (X,Y,Z) with only boundary voxels for that class.
    """
    lab = labels
    if lab.ndim == 4 and lab.shape[0] == 1:
        lab = lab[0]
    assert lab.ndim == 3
    nxp, nxm, nyp, nym, nzp, nzm = _neighbor_arrays3(lab)
    neighbors = np.stack([nxp, nxm, nyp, nym, nzp, nzm], axis=-1)
    diff = neighbors != lab[..., None]
    # Boundary mask: any neighbor differs
    bmask = diff.any(axis=-1)
    # Select neighbor label for each boundary voxel
    selected = np.zeros_like(lab)
    if prefer_nonzero:
        cand_nz = diff & (neighbors != 0)
        has_nz = cand_nz.any(axis=-1)
        # argmax returns 0 if all False; guard with has_nz
        idx_nz = np.argmax(cand_nz, axis=-1)
        sel_nz = np.take_along_axis(neighbors, idx_nz[..., None], axis=-1)[..., 0]
        selected = np.where(has_nz, sel_nz, 0)
    else:
        has_any = diff.any(axis=-1)
        idx_any = np.argmax(diff, axis=-1)
        sel_any = np.take_along_axis(neighbors, idx_any[..., None], axis=-1)[..., 0]
        selected = np.where(has_any, sel_any, 0)
    # Only keep boundary voxels; elsewhere mark -1 to ignore
    sel = np.where(bmask, selected, -1)
    # Optional thickening
    if thick > 1:
        try:
            from scipy.ndimage import binary_dilation
            bmask = binary_dilation(bmask, iterations=int(thick) - 1)
            # Re-apply dilation to sel by expanding original selection into dilated mask
            sel = np.where(bmask, sel, -1)
        except Exception:
            pass
    out: Dict[int, np.ndarray] = {}
    # Collect masks per present class (exclude -1)
    present = np.unique(sel)
    for c in present:
        c = int(c)
        if c < 0:
            continue
        out[c] = (sel == c)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sv-dir", type=str, default="", help="Directory containing <id>_sv_ids.npy files")
    ap.add_argument("--sv-labeled-dir", type=str, default="", help="Directory containing fully labeled <id>_labels.npy (optional)")
    ap.add_argument("--data-root", type=str, default="", help="WP5 root containing data/ with NIfTI pairs")
    ap.add_argument("--datalist", type=str, default="", help="Optional datalist JSON mapping ids -> image/label paths")
    return ap.parse_args()


def _new_imagedata():
    if hasattr(pv, "ImageData"):
        return pv.ImageData()
    if hasattr(pv, "UniformGrid"):
        return pv.UniformGrid()
    raise AttributeError("PyVista lacks ImageData/UniformGrid")


@st.cache_data(show_spinner=False)
def _cached_np_load(path: str) -> np.ndarray:
    return np.load(path)


@st.cache_data(show_spinner=False)
def _cached_json_load(path: str) -> Dict:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _cached_load_volume_ras(image_path: str, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
    return _load_volume_ras(image_path, label_path)


def _load_volume_ras(image_path: str, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if _HAS_MONAI:
        t = Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ])
        d = t({"image": image_path, "label": label_path})
        return d["image"].astype(np.float32), d["label"].astype(np.int64)
    if _HAS_NIB:
        img_nii = nib.load(image_path)
        lbl_nii = nib.load(label_path)
        img = np.asarray(img_nii.get_fdata()).astype(np.float32)
        lbl = np.asarray(lbl_nii.get_fdata()).astype(np.int64)
        if img.ndim == 3:
            img = img[None]
        if lbl.ndim == 3:
            lbl = lbl[None]
        return img, lbl
    raise RuntimeError("Install monai[all] or nibabel to load NIfTI")


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


def _nifti_meta(nifti_path: Path) -> Tuple[Tuple[float, float, float], Optional[np.ndarray]]:
    if not _HAS_NIB:
        return (1.0, 1.0, 1.0), None
    nii = nib.load(str(nifti_path))
    A = nii.affine
    spac = tuple(float(np.linalg.norm(A[:3, i])) for i in range(3))
    native = io_orientation(A)
    ras = axcodes2ornt(("R", "A", "S"))
    xform = ornt_transform(native, ras)
    return spac, xform


def _apply_ornt(arr: Optional[np.ndarray], xform: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None or xform is None:
        return arr
    if arr.ndim == 4 and arr.shape[0] == 1:
        inner = nib.orientations.apply_orientation(arr[0], xform)
        return inner[None]
    if arr.ndim == 3:
        return nib.orientations.apply_orientation(arr, xform)
    return arr


def _pclip_zscore(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.0, eps: float = 1e-8) -> np.ndarray:
    arr = x.astype(np.float32)
    flat = arr.reshape(-1)
    lo = np.percentile(flat, p_low)
    hi = np.percentile(flat, p_high)
    arr = np.clip(arr, lo, hi)
    mu = arr.mean()
    sd = arr.std()
    return (arr - mu) / (sd + eps)


def _new_boundary_mask(ids: np.ndarray, thick: int = 1) -> np.ndarray:
    a = ids[0] if (ids.ndim == 4 and ids.shape[0] == 1) else ids
    b = np.zeros_like(a, dtype=bool)
    for axis in (0, 1, 2):
        slf = [slice(None)] * 3
        slb = [slice(None)] * 3
        slf[axis] = slice(1, None)
        slb[axis] = slice(0, -1)
        af, ab = a[tuple(slf)], a[tuple(slb)]
        d = (af != ab)
        b[tuple(slf)] |= d
        b[tuple(slb)] |= d
    if thick > 1:
        try:
            from scipy.ndimage import binary_dilation
            b = binary_dilation(b, iterations=int(thick) - 1)
        except Exception:
            pass
    return b


def _new_im_surface(mask: np.ndarray, spacing_xyz: Tuple[float, float, float], decimate: float, ds: int) -> Optional[pv.PolyData]:
    vol = mask
    if vol.ndim == 4:
        vol = vol[0]
    if not vol.any():
        return None
    ds = max(1, int(ds))
    if ds > 1:
        vol = vol[::ds, ::ds, ::ds]
    grid = _new_imagedata()
    X, Y, Z = vol.shape
    dz, dy, dx = float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0])
    grid.dimensions = (Z + 1, Y + 1, X + 1)
    grid.spacing = (dz * ds, dy * ds, dx * ds)
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
            orig = surf
            ds_surf = surf.decimate_pro(float(decimate))
            if ds_surf is not None and getattr(ds_surf, "n_points", 0) >= max(100, int(0.05 * orig.n_points)):
                surf = ds_surf
        except Exception:
            pass
    return surf


def _volume_actor(img: np.ndarray, spacing_xyz: Tuple[float, float, float], opacity: float = 0.15) -> Tuple[pv.DataSet, Dict]:
    vol = img[0] if img.ndim == 4 else img
    vol = _pclip_zscore(vol)
    grid = _new_imagedata()
    X, Y, Z = vol.shape
    dz, dy, dx = float(spacing_xyz[2]), float(spacing_xyz[1]), float(spacing_xyz[0])
    grid.dimensions = (Z + 1, Y + 1, X + 1)
    grid.spacing = (dz, dy, dx)
    grid.origin = (0.0, 0.0, 0.0)
    grid.cell_data["intensity"] = vol.ravel(order="F")
    mesh = grid.cell_data_to_point_data()
    add_kwargs = dict(scalars="intensity", cmap="gray", opacity=float(opacity))
    return mesh, add_kwargs


def _finalize_camera(plotter: pv.Plotter) -> None:
    try:
        plotter.add_axes()
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.reset_camera_clipping_range()
    except Exception:
        pass


def _st_pyvista_show(plotter: pv.Plotter, key: str, caption: str = "") -> None:
    try:
        from stpyvista import stpyvista  # type: ignore
        stpyvista(plotter, key=key)
    except Exception as e:
        if caption:
            st.caption(caption)
        img = plotter.screenshot(return_img=True)
        st.image(img, caption="Static render (install streamlit-pyvista for interactivity)")


def _discover_ids(sv_dir: Path) -> List[str]:
    ids: List[str] = []
    for p in sorted(sv_dir.glob("*_sv_ids.npy")):
        name = p.name
        if name.endswith("_sv_ids.npy"):
            ids.append(name[: -len("_sv_ids.npy")])
        else:
            # Fallback: naive split
            ids.append(name.split("_sv_ids.npy")[0])
    return ids


def main():
    args = parse_args()
    st.set_page_config(page_title="WP5 SV-IDs Viewer (boundary)", layout="wide")
    st.title("WP5 Unlabeled Supervoxels — Boundary Viewer")

    SV_DIR_DEFAULT = Path("/home/peisheng/MONAI/runs/sv_fill_5k_nofill_ras2")
    sv_dir = Path(args.sv_dir) if args.sv_dir else SV_DIR_DEFAULT
    st.sidebar.header("Configuration")
    st.sidebar.info(f"sv_ids dir (default): {SV_DIR_DEFAULT}")
    sv_dir_str = st.sidebar.text_input("sv_ids directory", value=str(sv_dir))
    sv_dir = Path(sv_dir_str)

    SV_LABELED_DEFAULT = Path("/home/peisheng/MONAI/runs/sv_fullgt_5k_ras2")
    sv_labeled_dir = Path(args.sv_labeled_dir) if args.sv_labeled_dir else SV_LABELED_DEFAULT
    st.sidebar.info(f"sv_labels dir (default): {SV_LABELED_DEFAULT}")
    sv_labeled_dir_str = st.sidebar.text_input("sv_labels directory (full GT)", value=str(sv_labeled_dir))
    sv_labeled_dir = Path(sv_labeled_dir_str)

    data_root = Path(args.data_root) if args.data_root else None
    data_root_str = st.sidebar.text_input("data_root (for NIfTI)", value=str(data_root) if data_root else "")
    data_root = Path(data_root_str) if data_root_str else None

    datalist = Path(args.datalist) if args.datalist else None
    datalist_str = st.sidebar.text_input("datalist (optional)", value=str(datalist) if datalist else "")
    datalist = Path(datalist_str) if datalist_str else None

    if not sv_dir.exists():
        st.warning("Select a valid sv_ids directory to begin.")
        return

    ids = _discover_ids(sv_dir)
    if not ids:
        st.error("No *_sv_ids.npy files found in the selected directory.")
        return

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

    # Load sv_ids
    sv_ids_np: Optional[np.ndarray] = None
    sv_ids_p = sv_dir / f"{sel_id}_sv_ids.npy"
    if sv_ids_p.exists():
        try:
            sv_ids_np = _cached_np_load(str(sv_ids_p))
        except Exception:
            sv_ids_np = None

    # Load fully labeled volume (optional)
    sv_labels_full_np: Optional[np.ndarray] = None
    sv_labels_p = sv_labeled_dir / f"{sel_id}_labels.npy"
    if sv_labels_p.exists():
        try:
            sv_labels_full_np = _cached_np_load(str(sv_labels_p))
        except Exception:
            sv_labels_full_np = None

    # Load image/label to get spacing and volume
    img = None
    lbl = None
    spacing_xyz = (1.0, 1.0, 1.0)
    if img_path and lbl_path:
        try:
            img, lbl = _cached_load_volume_ras(str(img_path), str(lbl_path))
            if _HAS_NIB:
                try:
                    spacing_xyz, _ = _nifti_meta(lbl_path)
                except Exception:
                    spacing_xyz = (1.0, 1.0, 1.0)
        except Exception as e:
            st.warning(f"Failed to load volume/label: {e}")

    # Orientation toggle (if ids are not RAS)
    col_cfg, _ = st.columns([1, 3])
    with col_cfg:
        st.subheader("Render Controls")
        sv_already_ras = st.checkbox("IDs are already RAS-aligned", value=True, help="Uncheck to reorient using label header")
        show_volume = st.checkbox("Show image volume", value=False)
        vol_opacity = st.slider("Volume opacity", 0.0, 1.0, 0.10, 0.05)
        svb_thick = st.slider("Boundary thickness (voxels)", 1, 3, 1, 1)
        sv_decimate = st.slider("Surface decimation", 0.0, 0.9, 0.6, 0.05)
        sv_ds = st.slider("Surface downsample (stride)", 1, 4, 2, 1)
        svb_opacity = st.slider("Boundary opacity", 0.0, 1.0, 0.35, 0.05)
        show_sample = st.checkbox("Show sampled SV surfaces", value=False)
        max_regions = st.slider("Max sampled regions", 50, 400, 200, 10)
        min_vox = st.slider("Min voxels per region", 50, 2000, 200, 50)
        st.markdown("---")
        show_labeled = st.checkbox("Show labeled 3D surfaces (full GT)", value=True)
        use_interfaces = st.checkbox("Use labeled interface mesh (recommended)", value=True)
        fg_erosion = st.slider("Foreground erosion (voxels)", 0, 2, 0, 1, help="Erode class masks before contour to reduce coplanar overlap")
        show_silhouette = st.checkbox("Silhouette edges", value=False)
        labeled_classes = st.multiselect("Labeled classes", options=[0,1,2,3,4,6], default=[1,2,3,4])
        labeled_opacity = st.slider("Labeled surfaces opacity", 0.0, 1.0, 0.35, 0.05)

    # Prepare sv_ids
    if sv_ids_np is None:
        st.error("sv_ids not found for this case.")
        return
    sv_ids_r = sv_ids_np
    if not sv_already_ras and _HAS_NIB and lbl_path:
        try:
            _, xform = _nifti_meta(lbl_path)
            sv_ids_r = _apply_ornt(sv_ids_np, xform)
        except Exception:
            pass

    # Side-by-side: Unlabeled boundaries | Labeled full-GT surfaces
    col_u, col_l = st.columns(2)
    with col_u:
        st.subheader("SV boundaries (3D)")
        key_b = f"pv_svB|id={sel_id}|ds={int(sv_ds)}|dec={sv_decimate:.2f}|vol={int(show_volume)}|th={int(svb_thick)}"
        plotter = pv.Plotter(off_screen=False, window_size=(1024, 768))
        plotter.set_background("white")
        if show_volume and img is not None:
            grid, add_kwargs = _volume_actor(img, spacing_xyz=spacing_xyz, opacity=vol_opacity)
            plotter.add_volume(grid, **add_kwargs)
        try:
            bmask = _new_boundary_mask(sv_ids_r, thick=int(svb_thick))
            surf = _new_im_surface(bmask, spacing_xyz=spacing_xyz, decimate=sv_decimate, ds=int(sv_ds))
            if surf is not None and getattr(surf, "n_points", 0) > 0:
                plotter.enable_depth_peeling()
                plotter.add_mesh(surf, color="#34c759", opacity=float(svb_opacity), name="sv_boundaries", show_edges=bool(show_silhouette))
        except Exception as e:
            st.warning(f"Boundary surface failed: {e}")
        _finalize_camera(plotter)
        _st_pyvista_show(plotter, key=key_b, caption="SV boundaries 3D")

    with col_l:
        st.subheader("Labeled SV surfaces (3D)")
        key_lbl = f"pv_svLBL|id={sel_id}|ds={int(sv_ds)}|dec={sv_decimate:.2f}|vol={int(show_volume)}|cls={'-'.join(map(str,sorted(labeled_classes)))}"
        plotter2 = pv.Plotter(off_screen=False, window_size=(1024, 768))
        plotter2.set_background("white")
        if show_volume and img is not None:
            grid2, add_kwargs2 = _volume_actor(img, spacing_xyz=spacing_xyz, opacity=vol_opacity)
            plotter2.add_volume(grid2, **add_kwargs2)
        if show_labeled and (sv_labels_full_np is not None):
            lab = sv_labels_full_np
            if not sv_already_ras and _HAS_NIB and lbl_path:
                try:
                    _, xform = _nifti_meta(lbl_path)
                    lab = _apply_ornt(lab, xform)
                except Exception:
                    pass
            if lab.ndim == 4 and lab.shape[0] == 1:
                lab = lab[0]
            # Consistent class colors
            class_colors = {
                0: "#b0b0b0",
                1: "#e41a1c",
                2: "#ff7f00",
                3: "#4daf4a",
                4: "#377eb8",
                6: "#8e8e93",
            }
            plotter2.enable_depth_peeling()
            if bool(use_interfaces):
                try:
                    boundary_maps = _label_boundary_class_masks(lab, prefer_nonzero=True, thick=int(svb_thick))
                    # Optionally erode only foreground classes to reduce coplanarity
                    for c in labeled_classes:
                        c = int(c)
                        if c not in boundary_maps:
                            continue
                        mask_c = boundary_maps[c]
                        if int(fg_erosion) > 0 and c != 0:
                            try:
                                from scipy.ndimage import binary_erosion
                                mask_c = binary_erosion(mask_c, iterations=int(fg_erosion))
                            except Exception:
                                pass
                        if not mask_c.any():
                            continue
                        surf_c = _new_im_surface(mask_c, spacing_xyz=spacing_xyz, decimate=sv_decimate, ds=int(sv_ds))
                        if surf_c is None or getattr(surf_c, "n_points", 0) == 0:
                            continue
                        plotter2.add_mesh(surf_c, color=class_colors.get(c, "white"), opacity=float(labeled_opacity), name=f"iface_cls{c}", show_edges=bool(show_silhouette))
                except Exception as e:
                    st.warning(f"Labeled interface rendering failed: {e}")
            else:
                # Fallback to per-class shells (legacy)
                for c in labeled_classes:
                    try:
                        if c not in class_colors:
                            continue
                        m = (lab == int(c))
                        if int(fg_erosion) > 0 and c != 0:
                            try:
                                from scipy.ndimage import binary_erosion
                                m = binary_erosion(m, iterations=int(fg_erosion))
                            except Exception:
                                pass
                        if not m.any():
                            continue
                        surf_c = _new_im_surface(m, spacing_xyz=spacing_xyz, decimate=sv_decimate, ds=int(sv_ds))
                        if surf_c is None or getattr(surf_c, "n_points", 0) == 0:
                            continue
                        plotter2.add_mesh(surf_c, color=class_colors[int(c)], opacity=float(labeled_opacity), name=f"cls{c}", show_edges=bool(show_silhouette))
                    except Exception:
                        continue
        else:
            st.caption("No labeled volume found or toggle off.")
        _finalize_camera(plotter2)
        _st_pyvista_show(plotter2, key=key_lbl, caption="Labeled SV surfaces 3D")
        # Legend for labeled classes
        st.markdown("**Legend (Labeled Classes)**")
        legend_items = [
            ("Class 0 (bg)", "#b0b0b0"),
            ("Class 1", "#e41a1c"),
            ("Class 2", "#ff7f00"),
            ("Class 3", "#4daf4a"),
            ("Class 4", "#377eb8"),
            ("Class 6", "#8e8e93"),
        ]
        # Only show legend entries for classes currently selected
        selected_set = set(int(c) for c in labeled_classes)
        for label_txt, color in legend_items:
            cls_num = 0 if "Class 0" in label_txt else (6 if "Class 6" in label_txt else int(label_txt.split()[1]))
            if cls_num not in selected_set:
                continue
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px;'>"
                f"<div style='width:14px;height:14px;background:{color};border:1px solid #666;'></div>"
                f"<span>{label_txt}</span></div>",
                unsafe_allow_html=True,
            )

    # Optional: sampled region surfaces
    if show_sample:
        st.subheader("Sampled supervoxel surfaces (3D)")
        a = sv_ids_r[0] if (sv_ids_r.ndim == 4 and sv_ids_r.shape[0] == 1) else sv_ids_r
        uniq, counts = np.unique(a, return_counts=True)
        order = np.argsort(-counts)
        keep: List[int] = []
        for idx in order:
            if counts[idx] < int(min_vox):
                break
            keep.append(int(uniq[idx]))
            if len(keep) >= int(max_regions):
                break
        key_s = f"pv_svSamp|id={sel_id}|n={len(keep)}|ds={int(sv_ds)}|dec={sv_decimate:.2f}|vol={int(show_volume)}"
        plot2 = pv.Plotter(off_screen=False, window_size=(1024, 768))
        plot2.set_background("white")
        if show_volume and img is not None:
            grid2, add_kwargs2 = _volume_actor(img, spacing_xyz=spacing_xyz, opacity=vol_opacity)
            plot2.add_volume(grid2, **add_kwargs2)
        rng = np.random.default_rng(123)
        for k in keep:
            try:
                m = (a == k)
                surfk = _new_im_surface(m, spacing_xyz=spacing_xyz, decimate=sv_decimate, ds=int(sv_ds))
                if surfk is None or getattr(surfk, "n_points", 0) == 0:
                    continue
                r, g, b = rng.integers(64, 256, size=3).tolist()
                color = f"#{r:02x}{g:02x}{b:02x}"
                plot2.add_mesh(surfk, color=color, opacity=0.35, name=f"sv_{k}")
            except Exception:
                continue
        _finalize_camera(plot2)
        _st_pyvista_show(plot2, key=key_s, caption="Sampled SV surfaces 3D")

    # 2D slice viewer
    st.subheader("SV IDs slice (2D)")
    if sv_ids_r.ndim != 3:
        st.caption(f"Unexpected IDs shape: {sv_ids_r.shape}")
    else:
        ax = st.selectbox("Axis", options=[0, 1, 2], index=0, help="0=Z, 1=Y, 2=X")
        max_idx = sv_ids_r.shape[ax] - 1
        idx = st.slider("Slice index", 0, int(max_idx), int(max_idx // 2))
        if ax == 0:
            sl = sv_ids_r[idx, :, :]
        elif ax == 1:
            sl = sv_ids_r[:, idx, :]
        else:
            sl = sv_ids_r[:, :, idx]
        # Colorize by id deterministically
        ids2d = sl.astype(np.int64)
        uniq = np.unique(ids2d)
        rng = np.random.default_rng(0)
        lut = {int(k): rng.integers(0, 255, size=3, dtype=np.uint8) for k in uniq}
        rgb = np.zeros(ids2d.shape + (3,), dtype=np.uint8)
        for k, c in lut.items():
            rgb[ids2d == k] = c
        st.image(rgb, clamp=False, caption=f"SV IDs axis={ax} slice={idx}")

    # Debug
    with st.expander("Debug: spacing/paths"):
        st.write({
            "img_path": str(img_path) if img_path else None,
            "lbl_path": str(lbl_path) if lbl_path else None,
            "spacing_xyz": spacing_xyz,
            "ids_shape": tuple(sv_ids_np.shape) if sv_ids_np is not None else None,
        })


if __name__ == "__main__":
    main()
