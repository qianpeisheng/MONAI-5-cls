#!/usr/bin/env python3
"""
Streamlit app: inspect pseudo labels vs GT (2D 3-axis + 3D) with debug overlays.

Usage
  python3 -m streamlit run scripts/vis_pseudolabels_streamlit.py -- \
    --datalist datalist_train_new.json \
    --pseudo-dir /path/to/pseudo_labels_dir

Notes
- This app is visualization-only (no training/inference).
- Supports NIfTI (.nii/.nii.gz) and NumPy (.npy) volumes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import streamlit as st

from wp5.weaklabel.vis_utils import (
    boundary_band_mask,
    error_mask,
    load_volume_ras,
    outer_bg_distance_mask,
    transpose_for_imshow,
    valid_label_mask,
)


# Optional deps
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:
    plt = None

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:
    go = None

try:
    from skimage import measure as skmeasure  # type: ignore
except Exception:
    skmeasure = None


WP5_LABELS: Dict[int, str] = {-1: "unlabeled", 0: "bg", 1: "fg1", 2: "fg2", 3: "fg3", 4: "fg4", 6: "ignore"}
WP5_COLORS_RGBA: Dict[int, Tuple[float, float, float, float]] = {
    -1: (0.6, 0.6, 0.6, 0.5),
    0: (0.0, 0.0, 0.0, 0.0),  # transparent background overlay
    1: (1.0, 0.0, 0.0, 0.55),
    2: (0.0, 1.0, 0.0, 0.55),
    3: (0.0, 0.3, 1.0, 0.55),
    4: (1.0, 1.0, 0.0, 0.55),
    6: (1.0, 0.0, 1.0, 0.55),
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalist", type=str, default="datalist_train_new.json", help="JSON list of {image,label,id}")
    ap.add_argument("--pseudo-dir", type=str, default="", help="Directory containing pseudo labels (e.g., <id>_labels.npy)")
    ap.add_argument("--sv-ids-dir", type=str, default="", help="Optional directory containing <id>_sv_ids.npy")
    ap.add_argument("--seed-mask-dir", type=str, default="", help="Optional directory containing seed masks (e.g., <id>_strategic_seeds.npy)")
    ap.add_argument("--outer-bg-dir", type=str, default="", help="Optional directory containing outer BG masks (e.g., <id>_outer_bg.npy)")
    ap.add_argument("--case-id", type=str, default="", help="Optional: preselect a case id")
    return ap.parse_args()


def _robust_minmax(vol: np.ndarray, pmin: float = 1.0, pmax: float = 99.0) -> Tuple[float, float]:
    flat = np.asarray(vol).reshape(-1)
    if flat.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(flat, pmin))
    hi = float(np.percentile(flat, pmax))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(flat))
        hi = float(np.nanmax(flat))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0, 1.0
    return lo, hi


def _file_sig(path: Optional[Path]) -> Optional[Tuple[str, int, int]]:
    if path is None or not path.exists():
        return None
    stt = path.stat()
    return (str(path), int(stt.st_mtime_ns), int(stt.st_size))


@st.cache_data(show_spinner=False)
def _load_json_list(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def _discover_available_label_ids(dir_path: str) -> List[str]:
    """
    Best-effort discovery of case ids available in a directory of predicted/pseudo labels.
    Currently supports the common `<id>_labels.npy` convention.
    """
    p = Path(dir_path)
    if not p.exists() or not p.is_dir():
        return []
    out = set()
    for f in p.glob("*_labels.npy"):
        name = f.name
        if name.endswith("_labels.npy"):
            out.add(name[: -len("_labels.npy")])
    return sorted(out)


@st.cache_data(show_spinner=False)
def _load_vol_cached(sig: Tuple[str, int, int]) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    p, _mtime, _sz = sig
    return load_volume_ras(p)


def _normalize_label(vol: np.ndarray) -> np.ndarray:
    v = np.asarray(vol)
    if v.dtype.kind in ("f", "c"):
        v = np.rint(v)
    return v.astype(np.int16, copy=False)


def _resolve_by_patterns(base_dir: Path, case_id: str, patterns: Sequence[str]) -> Optional[Path]:
    for pat in patterns:
        p = base_dir / pat.format(id=case_id)
        if p.exists():
            return p
    return None


def _resolve_pseudo_path(pseudo_dir: Path, case_id: str) -> Optional[Path]:
    if not pseudo_dir.exists():
        return None
    pats = (
        "{id}_labels.npy",
        "{id}_label.npy",
        "{id}_pseudo.npy",
        "{id}_pseudo_labels.npy",
        "{id}_pred.npy",
        "{id}_pred.nii.gz",
        "{id}_pred.nii",
        "{id}.npy",
        "{id}.nii.gz",
        "{id}.nii",
    )
    return _resolve_by_patterns(pseudo_dir, case_id, pats)


def _resolve_optional_volume(base_dir: Path, case_id: str, kind: str) -> Optional[Path]:
    if not base_dir.exists():
        return None
    kind = kind.lower()
    if kind == "sv_ids":
        pats = ("{id}_sv_ids.npy", "{id}_sv_ids.nii.gz", "{id}_sv_ids.nii")
    elif kind == "seed_mask":
        pats = (
            "{id}_strategic_seeds.npy",
            "{id}_seeds.npy",
            "{id}_seed_mask.npy",
            "{id}_source.npy",
            "{id}_source_mask.npy",
        )
    elif kind == "outer_bg":
        pats = ("{id}_outer_bg.npy", "{id}_outerbg.npy", "{id}_outer_bg_mask.npy")
    else:
        return None
    return _resolve_by_patterns(base_dir, case_id, pats)


def _colorize_label2d(lbl2d: np.ndarray, alpha_scale: float) -> np.ndarray:
    h, w = lbl2d.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    u = np.unique(lbl2d)
    for v in u:
        rgba = WP5_COLORS_RGBA.get(int(v))
        if rgba is None:
            continue
        m = lbl2d == v
        if not m.any():
            continue
        r, g, b, a = rgba
        out[m, 0] = r
        out[m, 1] = g
        out[m, 2] = b
        out[m, 3] = a * float(alpha_scale)
    return out


def _imshow_slice(ax, img2d: np.ndarray, *, vmin: float, vmax: float) -> None:
    ax.imshow(transpose_for_imshow(img2d), cmap="gray", origin="lower", vmin=vmin, vmax=vmax)


def _overlay_mask(ax, mask2d: np.ndarray, color_rgba: Tuple[float, float, float, float]) -> None:
    m = mask2d.astype(bool)
    if not m.any():
        return
    h, w = m.shape
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    overlay[m, :] = np.asarray(color_rgba, dtype=np.float32)
    ax.imshow(transpose_for_imshow(overlay), origin="lower")


def _extract_slice(vol3d: np.ndarray, axis: str, idx: int) -> np.ndarray:
    if axis == "x":
        s = vol3d[idx, :, :]
    elif axis == "y":
        s = vol3d[:, idx, :]
    elif axis == "z":
        s = vol3d[:, :, idx]
    else:
        raise ValueError(axis)
    # keep consistent with existing viewers: rotate + flip for display
    return np.flipud(np.rot90(s, k=1))


def _slice_counts(lbl2d: np.ndarray, classes: Sequence[int]) -> Dict[int, int]:
    return {int(c): int((lbl2d == int(c)).sum()) for c in classes}


def _slice_error_rate(pred2d: np.ndarray, gt2d: np.ndarray, *, ignore_label: int = 6, unlabeled_value: int = -1) -> float:
    m = valid_label_mask(gt2d, ignore_label=ignore_label, unlabeled_value=unlabeled_value)
    denom = int(m.sum())
    if denom == 0:
        return 0.0
    return float(((pred2d != gt2d) & m).sum()) / float(denom)


def _sample_points(coords: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if coords.shape[0] <= max_points:
        return coords
    rng = np.random.default_rng(seed)
    idx = rng.choice(coords.shape[0], size=int(max_points), replace=False)
    return coords[idx]


@st.cache_data(show_spinner=False)
def _error_mask_from_sigs(
    pred_sig: Tuple[str, int, int],
    gt_sig: Tuple[str, int, int],
    *,
    ignore_label: int = 6,
    unlabeled_value: int = -1,
) -> np.ndarray:
    pred, _ = _load_vol_cached(pred_sig)
    gt, _ = _load_vol_cached(gt_sig)
    pred = _normalize_label(pred)
    gt = _normalize_label(gt)
    return error_mask(pred, gt, ignore_label=ignore_label, unlabeled_value=unlabeled_value)


@st.cache_data(show_spinner=False)
def _boundary_band_from_sig(
    label_sig: Tuple[str, int, int],
    *,
    radius: float,
) -> np.ndarray:
    lbl, _ = _load_vol_cached(label_sig)
    lbl = _normalize_label(lbl)
    return boundary_band_mask(lbl, radius=radius)


@st.cache_data(show_spinner=False)
def _outer_bg_diag_from_sig(
    label_sig: Tuple[str, int, int],
    *,
    min_distance: float,
) -> np.ndarray:
    lbl, _ = _load_vol_cached(label_sig)
    lbl = _normalize_label(lbl)
    return outer_bg_distance_mask(lbl, min_distance=min_distance)


@st.cache_data(show_spinner=False)
def _mesh_from_binary(
    sig: Tuple[str, int, int],
    *,
    class_value: int,
    ds: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]]:
    if skmeasure is None:
        return None
    vol, spacing_xyz = _load_vol_cached(sig)
    lbl = _normalize_label(vol)
    ds = max(1, int(ds))
    m = (lbl == int(class_value))
    if not m.any():
        return None
    md = m[::ds, ::ds, ::ds].astype(np.uint8)
    try:
        verts, faces, _normals, _vals = skmeasure.marching_cubes(md, level=0.5, step_size=1)
    except Exception:
        return None
    if verts.size == 0 or faces.size == 0:
        return None
    spacing_scaled = (float(spacing_xyz[0]) * ds, float(spacing_xyz[1]) * ds, float(spacing_xyz[2]) * ds)
    return verts.astype(np.float32), faces.astype(np.int32), spacing_scaled


def main() -> None:
    args = parse_args()
    st.set_page_config(page_title="WP5 Pseudo Labels Inspector", layout="wide")
    st.title("WP5 Pseudo Labels Inspector (2D + 3D)")

    st.sidebar.header("Data Source")
    mode = st.sidebar.radio("Input mode", options=["Datalist", "Manual paths"], index=0)

    datalist_path = st.sidebar.text_input("datalist (JSON list)", value=str(args.datalist))
    pseudo_dir_str = st.sidebar.text_input("pseudo label dir", value=str(args.pseudo_dir or ""))
    pseudo_dir = Path(pseudo_dir_str) if pseudo_dir_str else Path("")

    # Convenience: if a run folder is provided, auto-resolve to its `labels/` subdir.
    if pseudo_dir_str and pseudo_dir.exists() and pseudo_dir.is_dir():
        labels_subdir = pseudo_dir / "labels"
        if labels_subdir.exists() and labels_subdir.is_dir() and not any(pseudo_dir.glob("*_labels.npy")):
            pseudo_dir = labels_subdir
            pseudo_dir_str = str(pseudo_dir)
            st.sidebar.info(f"Using labels subdir: {pseudo_dir_str}")

    sv_ids_dir_str = st.sidebar.text_input("sv_ids dir (optional)", value=str(args.sv_ids_dir or ""))
    seed_mask_dir_str = st.sidebar.text_input("seed mask dir (optional)", value=str(args.seed_mask_dir or ""))
    outer_bg_dir_str = st.sidebar.text_input("outer BG mask dir (optional)", value=str(args.outer_bg_dir or ""))
    sv_ids_dir = Path(sv_ids_dir_str) if sv_ids_dir_str else Path("")
    seed_mask_dir = Path(seed_mask_dir_str) if seed_mask_dir_str else Path("")
    outer_bg_dir = Path(outer_bg_dir_str) if outer_bg_dir_str else Path("")

    only_pseudo_avail = False
    if mode == "Datalist" and pseudo_dir_str:
        only_pseudo_avail = st.sidebar.checkbox("Show only cases with pseudo available", value=True)

    case_id: str = ""
    img_path: Optional[Path] = None
    gt_path: Optional[Path] = None
    pseudo_path: Optional[Path] = None

    if mode == "Datalist":
        recs = _load_json_list(datalist_path)
        ids = [str(r.get("id", "")) for r in recs if r.get("id")]
        if not ids:
            st.warning("No cases found. Provide a valid datalist or switch to Manual paths.")
            return
        # If pseudo-dir is set, optionally filter cases to those available.
        if pseudo_dir_str and only_pseudo_avail:
            avail = set(_discover_available_label_ids(pseudo_dir_str))
            ids_f = sorted([c for c in ids if c in avail])
            # Fallback: if there is no overlap, try the train datalist automatically.
            if not ids_f:
                fallback_candidates = []
                p = Path(datalist_path)
                fallback_candidates.append(p.with_name("datalist_train_new.json"))
                fallback_candidates.append(p.with_name("datalist_train.json"))
                for fb in fallback_candidates:
                    if fb.exists():
                        recs_fb = _load_json_list(str(fb))
                        ids_fb = [str(r.get("id", "")) for r in recs_fb if r.get("id")]
                        ids_fb_f = sorted([c for c in ids_fb if c in avail])
                        if ids_fb_f:
                            st.sidebar.info(f"No overlap with {datalist_path}. Using {fb} instead.")
                            datalist_path = str(fb)
                            recs = recs_fb
                            ids = ids_fb
                            ids_f = ids_fb_f
                            break
            if not ids_f and ids:
                st.sidebar.warning("No cases match pseudo availability; showing all datalist cases (pseudo may be missing).")
            else:
                ids = ids_f
        default_idx = 0
        if args.case_id and args.case_id in ids:
            default_idx = ids.index(args.case_id)
        case_id = st.sidebar.selectbox("case id", options=ids, index=default_idx)
        rec = next((r for r in recs if str(r.get("id", "")) == case_id), {})
        img_path = Path(rec.get("image", "")) if rec.get("image") else None
        gt_path = Path(rec.get("label", "")) if rec.get("label") else None
        pseudo_path = _resolve_pseudo_path(pseudo_dir, case_id) if pseudo_dir_str else None
        if pseudo_dir_str and pseudo_path is None:
            st.sidebar.warning("Pseudo label file not found for this case (check datalist vs pseudo-dir).")
    else:
        case_id = st.sidebar.text_input("case id (optional)", value=str(args.case_id or ""))
        img_path_str = st.sidebar.text_input("image path (optional)", value="")
        gt_path_str = st.sidebar.text_input("GT label path (optional)", value="")
        pseudo_path_str = st.sidebar.text_input("pseudo label path (required for compare)", value="")
        img_path = Path(img_path_str) if img_path_str else None
        gt_path = Path(gt_path_str) if gt_path_str else None
        pseudo_path = Path(pseudo_path_str) if pseudo_path_str else None

    if not case_id:
        case_id = "case"

    st.sidebar.header("Overlay / Debug")
    alpha = st.sidebar.slider("label overlay alpha", min_value=0.0, max_value=1.0, value=0.55, step=0.05)
    overlay_mode = st.sidebar.selectbox(
        "overlay",
        options=[
            "Pseudo",
            "GT",
            "Error (pseudo != GT)",
            "Boundary band (FG)",
            "Outer BG diagnostic",
            "SV IDs (boundaries)",
            "Seed mask",
        ],
        index=0,
    )
    boundary_radius = st.sidebar.slider("boundary band radius (voxels)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    outer_bg_min_dist = st.sidebar.slider("outer BG min distance to FG (voxels)", min_value=0.0, max_value=64.0, value=16.0, step=1.0)
    max_3d_points = st.sidebar.slider("3D max points (scatter)", min_value=5_000, max_value=200_000, value=40_000, step=5_000)
    mesh_ds = st.sidebar.slider("3D mesh downsample", min_value=1, max_value=8, value=3, step=1)

    # Resolve optional paths
    sv_ids_path = _resolve_optional_volume(sv_ids_dir, case_id, "sv_ids") if sv_ids_dir_str else None
    seed_mask_path = _resolve_optional_volume(seed_mask_dir, case_id, "seed_mask") if seed_mask_dir_str else None
    outer_bg_path = _resolve_optional_volume(outer_bg_dir, case_id, "outer_bg") if outer_bg_dir_str else None

    # Load volumes
    img = None
    pseudo = None
    gt = None
    spacing_xyz = (1.0, 1.0, 1.0)
    img_sig = _file_sig(img_path) if img_path is not None else None
    gt_sig = _file_sig(gt_path) if gt_path is not None else None
    pseudo_sig = _file_sig(pseudo_path) if pseudo_path is not None else None

    if img_sig is not None:
        img, spacing_xyz = _load_vol_cached(img_sig)
        img = np.asarray(img)
    if gt_sig is not None:
        gt, _sp = _load_vol_cached(gt_sig)
        gt = _normalize_label(gt)
        spacing_xyz = _sp if img is None else spacing_xyz
    if pseudo_sig is not None:
        pseudo, _sp = _load_vol_cached(pseudo_sig)
        pseudo = _normalize_label(pseudo)
        spacing_xyz = _sp if img is None else spacing_xyz

    if img is None and gt is None and pseudo is None:
        st.warning("Provide at least one volume (image/GT/pseudo) to begin.")
        return

    # Sanity: align shapes (best-effort warning only)
    base_shape = next(v.shape for v in (img, gt, pseudo) if v is not None)
    for name, v in (("image", img), ("gt", gt), ("pseudo", pseudo)):
        if v is not None and v.shape != base_shape:
            st.error(f"Shape mismatch: {name} has {v.shape} but expected {base_shape}.")
            return

    # Optional volumes
    sv_ids = None
    seed_mask = None
    outer_bg_mask = None
    if sv_ids_path is not None and sv_ids_path.exists():
        sig = _file_sig(sv_ids_path)
        if sig is not None:
            sv_ids, _ = _load_vol_cached(sig)
            sv_ids = _normalize_label(sv_ids)
    if seed_mask_path is not None and seed_mask_path.exists():
        sig = _file_sig(seed_mask_path)
        if sig is not None:
            seed_mask, _ = _load_vol_cached(sig)
            seed_mask = np.asarray(seed_mask).astype(bool)
            if seed_mask.ndim == 4 and seed_mask.shape[0] == 1:
                seed_mask = seed_mask[0]
    if outer_bg_path is not None and outer_bg_path.exists():
        sig = _file_sig(outer_bg_path)
        if sig is not None:
            outer_bg_mask, _ = _load_vol_cached(sig)
            outer_bg_mask = np.asarray(outer_bg_mask).astype(bool)
            if outer_bg_mask.ndim == 4 and outer_bg_mask.shape[0] == 1:
                outer_bg_mask = outer_bg_mask[0]

    # Coordinates (shared across views)
    X, Y, Z = base_shape
    c1, c2, c3 = st.columns(3)
    with c1:
        ix = st.slider("x (sagittal)", min_value=0, max_value=max(0, X - 1), value=min(X // 2, max(0, X - 1)))
    with c2:
        iy = st.slider("y (coronal)", min_value=0, max_value=max(0, Y - 1), value=min(Y // 2, max(0, Y - 1)))
    with c3:
        iz = st.slider("z (axial)", min_value=0, max_value=max(0, Z - 1), value=min(Z // 2, max(0, Z - 1)))

    # Derived debug masks (computed only when needed)
    err = None
    if pseudo is not None and gt is not None:
        if pseudo_sig is not None and gt_sig is not None:
            err = _error_mask_from_sigs(pseudo_sig, gt_sig)
        else:
            err = error_mask(pseudo, gt)

    boundary_src = gt if gt is not None else pseudo
    boundary_src_sig = gt_sig if gt_sig is not None else pseudo_sig
    boundary = None
    if boundary_src is not None:
        if boundary_src_sig is not None:
            boundary = _boundary_band_from_sig(boundary_src_sig, radius=boundary_radius)
        else:
            boundary = boundary_band_mask(boundary_src, radius=boundary_radius)

    outer_bg_diag = outer_bg_mask
    if outer_bg_diag is None and boundary_src is not None:
        if boundary_src_sig is not None:
            outer_bg_diag = _outer_bg_diag_from_sig(boundary_src_sig, min_distance=outer_bg_min_dist)
        else:
            outer_bg_diag = outer_bg_distance_mask(boundary_src, min_distance=outer_bg_min_dist)

    tabs = st.tabs(["2D Views", "3D View", "Stats"])

    with tabs[0]:
        if plt is None:
            st.error("matplotlib is required for 2D views. Install: pip install matplotlib")
            return

        def _render_axis(axis: str, idx: int, title: str) -> None:
            fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))
            ax.set_title(title)
            ax.axis("off")

            if img is not None:
                img2d = _extract_slice(np.asarray(img), axis, idx)
                vmin, vmax = _robust_minmax(img2d)
                _imshow_slice(ax, img2d, vmin=vmin, vmax=vmax)
            else:
                ax.imshow(np.zeros((64, 64), dtype=np.float32), cmap="gray", origin="lower")

            if overlay_mode == "Pseudo" and pseudo is not None:
                lbl2d = _extract_slice(pseudo, axis, idx)
                ax.imshow(transpose_for_imshow(_colorize_label2d(lbl2d, alpha)), origin="lower")
            elif overlay_mode == "GT" and gt is not None:
                lbl2d = _extract_slice(gt, axis, idx)
                ax.imshow(transpose_for_imshow(_colorize_label2d(lbl2d, alpha)), origin="lower")
            elif overlay_mode == "Error (pseudo != GT)" and err is not None:
                m2d = _extract_slice(err.astype(np.uint8), axis, idx).astype(bool)
                _overlay_mask(ax, m2d, (1.0, 0.0, 1.0, 0.55))
            elif overlay_mode == "Boundary band (FG)" and boundary is not None:
                m2d = _extract_slice(boundary.astype(np.uint8), axis, idx).astype(bool)
                _overlay_mask(ax, m2d, (1.0, 0.4, 0.0, 0.55))
            elif overlay_mode == "Outer BG diagnostic" and outer_bg_diag is not None:
                m2d = _extract_slice(outer_bg_diag.astype(np.uint8), axis, idx).astype(bool)
                _overlay_mask(ax, m2d, (0.0, 1.0, 1.0, 0.55))
            elif overlay_mode == "Seed mask" and seed_mask is not None:
                m2d = _extract_slice(seed_mask.astype(np.uint8), axis, idx).astype(bool)
                _overlay_mask(ax, m2d, (1.0, 1.0, 1.0, 0.9))
            elif overlay_mode == "SV IDs (boundaries)" and sv_ids is not None:
                b = np.zeros_like(sv_ids, dtype=bool)
                a = sv_ids
                for ax_i in (0, 1, 2):
                    slf = [slice(None)] * 3
                    slb = [slice(None)] * 3
                    slf[ax_i] = slice(1, None)
                    slb[ax_i] = slice(0, -1)
                    af, ab = a[tuple(slf)], a[tuple(slb)]
                    d = af != ab
                    b[tuple(slf)] |= d
                    b[tuple(slb)] |= d
                m2d = _extract_slice(b.astype(np.uint8), axis, idx).astype(bool)
                _overlay_mask(ax, m2d, (0.2, 0.8, 1.0, 0.85))

            st.pyplot(fig, clear_figure=True)

            # Per-slice stats (compact)
            if gt is not None or pseudo is not None:
                with st.expander("Slice stats", expanded=False):
                    classes = (0, 1, 2, 3, 4, 6, -1)
                    gt2d = _extract_slice(gt, axis, idx) if gt is not None else None
                    p2d = _extract_slice(pseudo, axis, idx) if pseudo is not None else None
                    rows = []
                    for c in classes:
                        rows.append(
                            {
                                "class": int(c),
                                "name": WP5_LABELS.get(int(c), str(int(c))),
                                "gt": int((gt2d == int(c)).sum()) if gt2d is not None else None,
                                "pseudo": int((p2d == int(c)).sum()) if p2d is not None else None,
                            }
                        )
                    st.table(rows)
                    if p2d is not None and gt2d is not None:
                        st.metric("slice error rate (valid voxels)", value=f"{100.0 * _slice_error_rate(p2d, gt2d):.2f}%")

        c1, c2, c3 = st.columns(3)
        with c1:
            _render_axis("x", ix, "Sagittal (x)")
        with c2:
            _render_axis("y", iy, "Coronal (y)")
        with c3:
            _render_axis("z", iz, "Axial (z)")

    with tabs[1]:
        if go is None:
            st.error("plotly is required for 3D view. Install: pip install plotly")
            return
        st.subheader("3D")
        view_src = st.selectbox("3D source", options=["Pseudo", "GT", "Error", "Boundary band", "Outer BG"], index=0)
        classes_default = [1, 2, 3, 4]
        classes_sel = st.multiselect("classes", options=[0, 1, 2, 3, 4, 6, -1], default=classes_default)
        render_mode = st.radio("render mode", options=["Scatter", "Mesh"], index=0, horizontal=True)
        if render_mode == "Mesh" and skmeasure is None:
            st.warning("Mesh mode requires scikit-image; falling back to Scatter.")
            render_mode = "Scatter"

        fig = go.Figure()
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            scene=dict(aspectmode="data"),
            showlegend=True,
        )

        def _add_scatter(mask: np.ndarray, name: str, color: str) -> None:
            coords = np.argwhere(mask)
            if coords.size == 0:
                return
            coords = _sample_points(coords, int(max_3d_points), seed=0)
            try:
                fig.add_trace(
                    go.Scatter3d(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        z=coords[:, 2],
                        mode="markers",
                        marker=dict(size=1.5, color=color, opacity=0.6),
                        name=name,
                    )
                )
            except Exception as e:
                st.warning(f"3D scatter failed for {name}: {e}")

        def _add_mesh_from_label(label_sig: Tuple[str, int, int], class_value: int, name: str, color: str) -> None:
            mesh = _mesh_from_binary(label_sig, class_value=int(class_value), ds=int(mesh_ds))
            if mesh is None:
                return
            verts, faces, spacing_scaled = mesh
            vx = verts[:, 0] * spacing_scaled[0]
            vy = verts[:, 1] * spacing_scaled[1]
            vz = verts[:, 2] * spacing_scaled[2]
            i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
            try:
                fig.add_trace(go.Mesh3d(x=vx, y=vy, z=vz, i=i, j=j, k=k, color=color, opacity=0.35, name=name))
            except Exception as e:
                st.warning(f"3D mesh failed for {name}: {e}")

        if not classes_sel and view_src in ("Pseudo", "GT"):
            st.info("Select at least one class.")
        else:
            if view_src == "Pseudo" and pseudo is not None:
                if render_mode == "Scatter":
                    for c in classes_sel:
                        _add_scatter(pseudo == int(c), f"pseudo:{c}", _color_for_class(int(c)))
                else:
                    psig = _file_sig(pseudo_path) if pseudo_path is not None else None
                    if psig is None:
                        st.warning("Mesh mode requires pseudo path on disk.")
                    else:
                        for c in classes_sel:
                            _add_mesh_from_label(psig, int(c), f"pseudo:{c}", _color_for_class(int(c)))
            elif view_src == "GT" and gt is not None:
                if render_mode == "Scatter":
                    for c in classes_sel:
                        _add_scatter(gt == int(c), f"gt:{c}", _color_for_class(int(c)))
                else:
                    gsig = _file_sig(gt_path) if gt_path is not None else None
                    if gsig is None:
                        st.warning("Mesh mode requires GT path on disk.")
                    else:
                        for c in classes_sel:
                            _add_mesh_from_label(gsig, int(c), f"gt:{c}", _color_for_class(int(c)))
            elif view_src == "Error" and err is not None:
                _add_scatter(err, "error", "#ff00ff")
            elif view_src == "Boundary band" and boundary is not None:
                _add_scatter(boundary, "boundary_band", "#ff6600")
            elif view_src == "Outer BG" and outer_bg_diag is not None:
                _add_scatter(outer_bg_diag, "outer_bg", "#00ffff")
            else:
                st.info("Select a source with available volumes.")

        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Stats")
        st.write(
            {
                "case_id": case_id,
                "image_path": str(img_path) if img_path else None,
                "gt_path": str(gt_path) if gt_path else None,
                "pseudo_path": str(pseudo_path) if pseudo_path else None,
                "shape": list(base_shape),
                "spacing_xyz": spacing_xyz,
            }
        )

        classes = (0, 1, 2, 3, 4, 6, -1)
        if gt is not None:
            st.write({"gt_counts": {str(k): v for k, v in _slice_counts(gt, classes).items()}})
        if pseudo is not None:
            st.write({"pseudo_counts": {str(k): v for k, v in _slice_counts(pseudo, classes).items()}})

        if pseudo is not None and gt is not None:
            m = valid_label_mask(gt, ignore_label=6, unlabeled_value=-1)
            denom = int(m.sum())
            err_ct = int(((pseudo != gt) & m).sum())
            st.write({"valid_voxels": denom, "error_voxels": err_ct, "error_rate": (float(err_ct) / float(denom)) if denom else 0.0})

        if plt is not None:
            st.divider()
            st.caption("Download current axial (z) view as PNG")
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.axis("off")
            if img is not None:
                img2d = _extract_slice(np.asarray(img), "z", int(iz))
                vmin, vmax = _robust_minmax(img2d)
                _imshow_slice(ax, img2d, vmin=vmin, vmax=vmax)
            if overlay_mode == "Pseudo" and pseudo is not None:
                lbl2d = _extract_slice(pseudo, "z", int(iz))
                ax.imshow(transpose_for_imshow(_colorize_label2d(lbl2d, alpha)), origin="lower")
            elif overlay_mode == "GT" and gt is not None:
                lbl2d = _extract_slice(gt, "z", int(iz))
                ax.imshow(transpose_for_imshow(_colorize_label2d(lbl2d, alpha)), origin="lower")
            elif overlay_mode == "Error (pseudo != GT)" and err is not None:
                m2d = _extract_slice(err.astype(np.uint8), "z", int(iz)).astype(bool)
                _overlay_mask(ax, m2d, (1.0, 0.0, 1.0, 0.55))
            elif overlay_mode == "Boundary band (FG)" and boundary is not None:
                m2d = _extract_slice(boundary.astype(np.uint8), "z", int(iz)).astype(bool)
                _overlay_mask(ax, m2d, (1.0, 0.4, 0.0, 0.55))
            elif overlay_mode == "Outer BG diagnostic" and outer_bg_diag is not None:
                m2d = _extract_slice(outer_bg_diag.astype(np.uint8), "z", int(iz)).astype(bool)
                _overlay_mask(ax, m2d, (0.0, 1.0, 1.0, 0.55))

            import io

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            st.download_button(
                "Download PNG",
                data=buf.getvalue(),
                file_name=f"{case_id}_z{iz}_{overlay_mode.replace(' ', '_')}.png",
                mime="image/png",
            )


def _color_for_class(c: int) -> str:
    if c == 0:
        return "#808080"
    if c == 1:
        return "#ff0000"
    if c == 2:
        return "#00ff00"
    if c == 3:
        return "#0050ff"
    if c == 4:
        return "#ffff00"
    if c == 6:
        return "#ff00ff"
    if c == -1:
        return "#aaaaaa"
    return "#ffffff"


if __name__ == "__main__":
    main()
