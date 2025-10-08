#!/usr/bin/env python3
"""
Streamlit visualization for WP5 3D segmentations.

Features:
- Browse cases from a datalist (JSON of {image,label,id}).
- Show 2D slices with overlays for ground-truth labels and predictions.
- Optional 3D view using Plotly (isosurface per selected class, downsampled).
- Displays summary metrics if a metrics JSON is available.

Launch:
  streamlit run scripts/vis_wp5_streamlit.py -- \
    --pred_dir runs/grid_clip_zscore/scratch_subset_100_eval/preds \
    --datalist datalist_test.json

Requirements: streamlit, numpy, nibabel, matplotlib, plotly
  pip install streamlit numpy nibabel matplotlib plotly
"""

from __future__ import annotations

import argparse
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


def parse_cli_args():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--pred_dir", type=str, default="", help="Directory containing *_pred.nii.gz files")
    ap.add_argument("--datalist", type=str, default="datalist_test.json", help="JSON datalist with image/label/id")
    ap.add_argument("--metrics", type=str, default="", help="Optional metrics JSON (summary)")
    known, _ = ap.parse_known_args()
    return known


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

    vmin, vmax = robust_minmax(img2d)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img2d.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    if lbl2d is not None:
        ax.imshow(colorize_seg2d(lbl2d.T, alpha=alpha), origin="lower")
    if prd2d is not None:
        ax.imshow(colorize_seg2d(prd2d.T, alpha=alpha), origin="lower")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)


def draw_slice_pair(image: np.ndarray, label: Optional[np.ndarray], pred: Optional[np.ndarray], axis: str, index: int, alpha: float):
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

    vmin, vmax = robust_minmax(img2d)
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
        ax1.set_title("Ground Truth")
        ax1.imshow(img2d.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        if lbl2d is not None:
            ax1.imshow(colorize_seg2d(lbl2d.T, alpha=alpha), origin="lower")
        ax1.axis("off")
        st.pyplot(fig1, clear_figure=True)
    with col2:
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
        ax2.set_title("Prediction")
        ax2.imshow(img2d.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        if prd2d is not None:
            ax2.imshow(colorize_seg2d(prd2d.T, alpha=alpha), origin="lower")
        ax2.axis("off")
        st.pyplot(fig2, clear_figure=True)


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


def plot_3d_pair(image: np.ndarray, gt: Optional[np.ndarray], pred: Optional[np.ndarray], cls: int, downsample: int = 2, opacity: float = 0.25, show_image: bool = False):
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
    fig = make_subplots(rows=1, cols=2, specs=specs, subplot_titles=("GT", "Prediction"))

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


def main():
    args = parse_cli_args()

    st.title("WP5 Segmentation Viewer")
    with st.sidebar:
        pred_dir = st.text_input("Predictions dir", value=args.pred_dir)
        datalist_path = st.text_input("Datalist JSON", value=args.datalist)
        metrics_path = st.text_input("Metrics JSON (optional)", value=args.metrics)
        alpha = st.slider("Overlay alpha", 0.1, 1.0, 0.4, 0.05)

    pred_dir_p = Path(pred_dir) if pred_dir else None
    datalist_p = Path(datalist_path) if datalist_path else None

    if datalist_p is None or not datalist_p.exists():
        st.info("Provide a valid datalist JSON path.")
        st.stop()
    if pred_dir_p is None or not pred_dir_p.exists():
        st.info("Provide a valid predictions directory (from eval).")
        st.stop()

    records = load_json(datalist_p)
    if not records:
        st.warning("Datalist is empty or could not be loaded.")
        st.stop()

    case_ids = [r.get("id", f"case_{i}") for i, r in enumerate(records)]
    case_sel = st.selectbox("Select case", options=case_ids, index=0)
    rec = next(r for r in records if r.get("id") == case_sel)
    img_path = Path(rec["image"]) if rec.get("image") else None
    lbl_path = Path(rec["label"]) if rec.get("label") else None
    pred_path = find_pred_path(pred_dir_p, img_path, case_sel) if (pred_dir_p and img_path) else None

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.write(f"Image: {img_path}")
        if img_path and img_path.exists():
            img_vol, _ = load_nii(img_path)
            st.write(f"Image shape: {img_vol.shape}")
        else:
            st.error("Image not found; check datalist paths.")
            st.stop()
    with col_b:
        st.write(f"Label: {lbl_path}")
        lbl_vol = None
        if lbl_path and lbl_path.exists():
            lbl_vol, _ = load_nii(lbl_path)
        st.write(f"Prediction: {pred_path}")
        pred_vol = None
        if pred_path and pred_path.exists():
            pred_vol, _ = load_volume_any(pred_path)
            # ensure segmentation shape is (X,Y,Z)
            if pred_vol.ndim == 4 and pred_vol.shape[0] == 1:
                pred_vol = pred_vol[0]
        if pred_vol is None:
            st.warning("Prediction not found for this case in pred_dir.")

    # 2D slice viewer
    st.subheader("2D Slice View")
    axis = st.radio("Axis", options=["z", "y", "x"], index=0, horizontal=True)
    dim = {"x": 0, "y": 1, "z": 2}[axis]
    size = int(img_vol.shape[dim])
    idx = st.slider("Slice index", 0, max(0, size - 1), size // 2)
    # Show separate GT/Pred overlays side-by-side
    draw_slice_pair(img_vol, lbl_vol, pred_vol, axis, idx, alpha)
    with st.expander("Show combined overlay (image + GT + Pred)"):
        draw_slice(img_vol, lbl_vol, pred_vol, axis, idx, alpha)

    # 3D viewer (optional)
    st.subheader("3D View (Plotly)")
    if go is None:
        st.info("Install plotly for 3D view: pip install plotly")
    else:
        cls = st.selectbox("Class isosurface (1-4)", options=[1, 2, 3, 4], index=1)
        ds = st.slider("Downsample (higher=faster)", 1, 6, 3)
        op = st.slider("Surface opacity", 0.05, 0.8, 0.25, 0.05)
        show_img_vol = st.checkbox("Show underlying image volume", value=False)
        plot_3d_pair(img_vol, lbl_vol, pred_vol, cls=cls, downsample=ds, opacity=op, show_image=show_img_vol)

    # Metrics (optional)
    if not metrics_path:
        cand = pred_dir_p.parent / "metrics" / "summary.json"
        if cand.exists():
            metrics_path = str(cand)
    if metrics_path:
        mp = Path(metrics_path)
        if mp.exists():
            js = load_json(mp)
            if isinstance(js, dict) and "average" in js:
                st.subheader("Summary Metrics (0..4)")
                avg = js.get("average", {})
                st.write({"dice": avg.get("dice"), "iou": avg.get("iou"), "hd": avg.get("hd"), "asd": avg.get("asd")})
                st.caption(str(mp))


if __name__ == "__main__":
    # Streamlit entry
    main()
