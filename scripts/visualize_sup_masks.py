#!/usr/bin/env python3
"""
Visualize supervision masks produced in a WP5-style run.

For each case in a given sup_masks directory, this script:
 - Loads `*_supmask.npy` (boolean) and `*_pseudolabel.npy` (int labels)
 - Saves a 2D figure with three orthogonal slices that intersect supervised voxels
 - Saves a 3D scatter plot of supervised voxels colored by pseudo-label class

Usage:
  python scripts/visualize_sup_masks.py \
    --sup-dir runs/fixed_points_scratch50/ratio_0.00001/sup_masks \
    --out-dir runs/fixed_points_scratch50/ratio_0.00001/vis

Notes:
 - Designed to be lightweight: uses Matplotlib only.
 - If a case has no supervised voxels, it still saves centered slices and an empty 3D plot.
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)


def find_best_slice(mask: np.ndarray, axis: int) -> int:
    """Pick the slice index along `axis` with the most True voxels.
    Falls back to center slice if mask is all False.
    Expects mask shape (C=1, X, Y, Z) or (X, Y, Z).
    """
    if mask.ndim == 4:
        mask3d = mask[0]
    else:
        mask3d = mask

    axes = [0, 1, 2]
    axes.remove(axis)
    # Sum over the two other axes to count trues per slice.
    counts = mask3d.sum(axis=tuple(axes))
    if counts.max() == 0:
        # No supervised voxels; return center slice
        return mask3d.shape[axis] // 2
    return int(np.argmax(counts))


def class_colors(num_classes: int = 7) -> List[Tuple[float, float, float]]:
    """Return distinct RGB colors for up to 7 classes.
    Index 0 is background (light gray), others colorful.
    """
    palette = [
        (0.8, 0.8, 0.8),  # 0 background
        (0.89, 0.10, 0.11),  # 1 red
        (0.12, 0.47, 0.71),  # 2 blue
        (0.20, 0.63, 0.17),  # 3 green
        (0.60, 0.31, 0.64),  # 4 purple
        (1.00, 0.50, 0.00),  # 5 orange (unused by default)
        (0.65, 0.34, 0.16),  # 6 brown (ignored in metrics normally)
    ]
    return palette[:num_classes]


def overlay_slice(ax, label_slice: np.ndarray, sup_slice: np.ndarray, title: str):
    """Render a single slice: show supervised voxels colored by their label.

    label_slice: 2D array of ints (labels)
    sup_slice: 2D boolean array, True where supervised
    """
    # Background
    ax.imshow(np.zeros_like(label_slice), cmap="gray", vmin=0, vmax=1)
    colors = class_colors(7)
    classes = np.unique(label_slice[sup_slice])
    # Draw supervised pixels by class to get discrete colors
    for c in classes:
        if c < 0:
            continue
        mask_c = (label_slice == c) & sup_slice
        if not np.any(mask_c):
            continue
        rgb = np.array(colors[int(c)])
        rgba = np.zeros(mask_c.shape + (4,), dtype=float)
        rgba[..., :3] = rgb
        rgba[..., 3] = mask_c.astype(float) * 0.95
        ax.imshow(rgba)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_2d_fig(case_id: str, sup: np.ndarray, plabel: np.ndarray, out_path: Path, stats: dict = None):
    """Save a 2D figure with axial, coronal, sagittal supervised overlays."""
    if sup.ndim == 4:
        sup3d = sup[0]
        pl3d = plabel[0]
    else:
        sup3d = sup
        pl3d = plabel

    # Choose slices that intersect supervised voxels
    z = find_best_slice(sup3d, axis=2)
    y = find_best_slice(sup3d, axis=1)
    x = find_best_slice(sup3d, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    # Axial: (X,Y) at Z=z
    overlay_slice(axes[0], pl3d[:, :, z], sup3d[:, :, z], f"Axial z={z}")
    # Coronal: (X,Z) at Y=y
    overlay_slice(axes[1], pl3d[:, y, :], sup3d[:, y, :], f"Coronal y={y}")
    # Sagittal: (Y,Z) at X=x
    overlay_slice(axes[2], pl3d[x, :, :], sup3d[x, :, :], f"Sagittal x={x}")

    suptitle = case_id
    if stats is not None and "sup_fraction" in stats:
        suptitle += f"  | sup_fraction={stats['sup_fraction']:.2e}"
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_3d_scatter(case_id: str, sup: np.ndarray, plabel: np.ndarray, out_path: Path, max_points: int = 5000):
    """Save a 3D scatter of supervised voxels colored by pseudo-label class.

    Limits the number of points for rendering performance.
    """
    if sup.ndim == 4:
        sup3d = sup[0]
        pl3d = plabel[0]
    else:
        sup3d = sup
        pl3d = plabel

    coords = np.argwhere(sup3d)
    if coords.size == 0:
        # Create an empty plot with annotation
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"{case_id} (no supervised voxels)")
        for axis in 'xyz':
            getattr(ax, f"set_{axis}label")(axis)
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        return

    # Downsample if too many points
    if len(coords) > max_points:
        sel = np.random.RandomState(42).choice(len(coords), size=max_points, replace=False)
        coords = coords[sel]

    labels_at = pl3d[tuple(coords.T)]
    colors = class_colors(7)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    for c in np.unique(labels_at):
        c = int(c)
        idx = labels_at == c
        if not np.any(idx):
            continue
        xyz = coords[idx]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=8, c=[colors[c]], label=str(c), depthshade=False)
    ax.legend(title="class", loc="upper right", fontsize=8)
    ax.set_title(case_id, fontsize=10)
    ax.view_init(elev=20, azim=30)
    # Keep axes tight to the volume
    ax.set_xlim(0, sup3d.shape[0])
    ax.set_ylim(0, sup3d.shape[1])
    ax.set_zlim(0, sup3d.shape[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sup-dir", required=True, help="Directory containing *_supmask.npy and *_pseudolabel.npy")
    ap.add_argument("--out-dir", required=True, help="Output directory for figures")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of cases (0=all)")
    args = ap.parse_args()

    sup_dir = Path(args.sup_dir)
    out_dir = Path(args.out_dir)
    out_2d = out_dir / "2d"
    out_3d = out_dir / "3d"
    out_2d.mkdir(parents=True, exist_ok=True)
    out_3d.mkdir(parents=True, exist_ok=True)

    sup_files = sorted(glob.glob(str(sup_dir / "*_supmask.npy")))
    total = len(sup_files)
    if args.limit and args.limit > 0:
        sup_files = sup_files[: args.limit]

    print(f"Found {total} supmask files; processing {len(sup_files)}")

    for i, sup_path in enumerate(sup_files, 1):
        sup_path = Path(sup_path)
        case_id = sup_path.name.replace("_supmask.npy", "")
        plabel_path = sup_path.with_name(case_id + "_pseudolabel.npy")
        stats_path = sup_path.with_name(case_id + "_supmask_stats.json")

        if not plabel_path.exists():
            print(f"[WARN] Missing pseudolabel for {case_id}, skipping")
            continue

        sup = np.load(sup_path)
        plabel = np.load(plabel_path)
        stats = None
        if stats_path.exists():
            try:
                stats = json.loads(stats_path.read_text())
            except Exception:
                stats = None

        out2d = out_2d / f"{case_id}_slices.png"
        out3d = out_3d / f"{case_id}_3d.png"

        save_2d_fig(case_id, sup, plabel, out2d, stats)
        save_3d_scatter(case_id, sup, plabel, out3d)

        if i % 25 == 0 or i == len(sup_files):
            print(f"Processed {i}/{len(sup_files)}")


if __name__ == "__main__":
    main()

