#!/usr/bin/env python3
"""
Generate RAS-aligned supervoxels for WP5 NIfTI volumes and optionally fill
supervoxels from seed points (via GT labels at seed locations) or assign every
supervoxel a label by majority vote from the full ground-truth (GT) label map.

Outputs per case id (<id>):
  - <id>_sv_ids.npy      (int32, shape (X,Y,Z), RAS)
  - <id>_labels.npy      (int16/int32, shape (X,Y,Z), RAS; 0 for unlabeled/background by default)
  - <id>_sv_meta.json    (meta with spacing, counts, timings, etc.)

Notes
- Orientation: volumes, labels, seeds are reoriented to RAS using the chosen
  reference header (default: label NIfTI) so arrays are consistent with the
  Streamlit viewer (MONAI Orientationd RAS).
- Spacing: SLIC receives voxel spacing (dx,dy,dz) to account for anisotropy.
- Filling policy (default): an SV is assigned to the majority GT class among
  seed voxels inside it. Unseeded SVs remain 0 in labels.
- Full-GT assignment (optional with --assign-all-from-gt): every SV is assigned
  to the majority GT class among all its voxels (respecting --ignore-class).

Example (10k, with seeds fill):
  python3 scripts/gen_supervoxels_wp5.py \
    --data-root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
    --sup-dir /home/peisheng/MONAI/runs/sup_masks_0p1pct_global_d0_5_nov_ras \
    --out-dir /home/peisheng/MONAI/runs/sv_fill_0p1pct_n10k_ras2 \
    --n-segments 10000 --compactness 0.02 --sigma 0.5 \
    --datalist datalist_train.json --ref-header label

Example (1.5k, no fill):
  python3 scripts/gen_supervoxels_wp5.py \
    --data-root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
    --out-dir /home/peisheng/MONAI/runs/sv_fill_0p1pct_ras2 \
    --n-segments 1500 --compactness 0.02 --sigma 0.5 \
    --datalist datalist_train.json --no-fill --ref-header label

Example (5k, assign every SV from GT via majority vote, ignore class 6):
  python3 scripts/gen_supervoxels_wp5.py \
    --data-root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
    --out-dir /home/peisheng/MONAI/runs/sv_fill_5k_fullgt_ras \
    --n-segments 5000 --compactness 0.02 --sigma 0.5 \
    --datalist datalist_train.json --ref-header label --assign-all-from-gt --ignore-class 6
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import nibabel as nib
    from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
    _HAS_NIB = True
except Exception:
    _HAS_NIB = False

try:
    from skimage.segmentation import slic
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="WP5 root containing data/ with NIfTI pairs")
    ap.add_argument("--out-dir", required=True, help="Output directory for supervoxels run")
    ap.add_argument("--sup-dir", default="", help="Optional sup_masks run dir to read <id>_seedmask.npy for fill")
    ap.add_argument("--datalist", default="", help="Optional datalist JSON with case 'id' entries")
    ap.add_argument("--ids", default="", help="Optional comma-separated ids to limit")
    ap.add_argument("--ref-header", choices=["image", "label"], default="label", help="Which NIfTI header to use for orientation+spacing")
    ap.add_argument("--n-segments", type=int, default=10000, help="Target number of supervoxels")
    ap.add_argument("--compactness", type=float, default=0.02, help="SLIC compactness")
    ap.add_argument("--sigma", type=float, default=0.5, help="Gaussian smoothing for SLIC")
    ap.add_argument(
        "--mode",
        choices=["slic", "slic-grad-mag", "slic-grad-vec"],
        default="slic",
        help="Feature mode for SLIC: intensity only, intensity+|grad|, or intensity+[gx,gy,gz]",
    )
    ap.add_argument("--grad-sigma", type=float, default=1.0, help="Smoothing sigma for gradient features (voxels)")
    ap.add_argument("--no-fill", action="store_true", help="Do not fill supervoxels from seeds")
    ap.add_argument("--ignore-class", type=int, default=None, help="Optional label to ignore when filling (e.g., 6)")
    ap.add_argument(
        "--assign-all-from-gt",
        action="store_true",
        help=(
            "Assign every supervoxel a label by majority vote over full GT labels; "
            "respects --ignore-class. Overrides --no-fill and does not require --sup-dir."
        ),
    )
    return ap.parse_args()


def load_ids(data_root: Path, datalist: Optional[Path], ids_csv: str) -> Sequence[str]:
    ids = []
    if datalist and datalist.exists():
        try:
            lst = json.loads(datalist.read_text())
            ids = [str(rec.get("id")) for rec in lst if rec.get("id")]
        except Exception:
            ids = []
    if not ids:
        # discover by image presence
        for p in (data_root / "data").glob("*_image.nii"):
            cid = p.name[:-10]
            if (data_root / "data" / f"{cid}_label.nii").exists():
                ids.append(cid)
    if ids_csv:
        want = set([s for s in ids_csv.split(",") if s])
        ids = [i for i in ids if i in want]
    return sorted(ids)


def nifti_meta(img_path: Path) -> Tuple[Tuple[float, float, float], Optional[np.ndarray]]:
    if not _HAS_NIB:
        return (1.0, 1.0, 1.0), None
    nii = nib.load(str(img_path))
    A = nii.affine
    spac = tuple(float(np.linalg.norm(A[:3, i])) for i in range(3))  # (dx,dy,dz)
    native = io_orientation(A)
    ras = axcodes2ornt(("R", "A", "S"))
    xform = ornt_transform(native, ras)
    return spac, xform


def apply_ornt(arr: Optional[np.ndarray], xform: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None or xform is None:
        return arr
    if arr.ndim == 4 and arr.shape[0] == 1:
        inner = nib.orientations.apply_orientation(arr[0], xform)
        return inner[None]
    if arr.ndim == 3:
        return nib.orientations.apply_orientation(arr, xform)
    return arr


def pclip_zscore(x: np.ndarray, p_low: float = 1.0, p_high: float = 99.0, eps: float = 1e-8) -> np.ndarray:
    arr = x.astype(np.float32)
    flat = arr.reshape(-1)
    lo = np.percentile(flat, p_low)
    hi = np.percentile(flat, p_high)
    arr = np.clip(arr, lo, hi)
    mu = arr.mean()
    sd = arr.std()
    return (arr - mu) / (sd + eps)


def compute_supervoxels(
    img: np.ndarray,
    n_segments: int,
    compactness: float,
    sigma: float,
    spacing_xyz: Tuple[float, float, float],
    channel_axis: Optional[int] = None,
) -> np.ndarray:
    if not _HAS_SKIMAGE:
        raise RuntimeError("scikit-image is required: pip install -U scikit-image")
    dx, dy, dz = spacing_xyz
    # scikit-image expects spacing in (z,y,x) order
    spacing_zyx = (dz, dy, dx)
    # SLIC returns labels starting at 0 when start_label=0; we prefer 0-based
    seg = slic(
        image=img,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        spacing=spacing_zyx,
        start_label=0,
        channel_axis=channel_axis,
    )
    return seg.astype(np.int32)


def fill_from_seeds(sv_ids: np.ndarray, seedmask: Optional[np.ndarray], gt_labels: Optional[np.ndarray], ignore_class: Optional[int]) -> Tuple[np.ndarray, int, float]:
    X, Y, Z = sv_ids.shape
    out = np.zeros((X, Y, Z), dtype=np.int16)
    if seedmask is None or gt_labels is None:
        return out, 0, 0.0
    seeds = seedmask.astype(bool)
    if seeds.ndim == 4:
        seeds = seeds[0]
    gt = gt_labels
    if gt.ndim == 4:
        gt = gt[0]
    coords = np.argwhere(seeds)
    if coords.size == 0:
        return out, 0, 0.0
    # Build mapping sv_id -> class votes
    votes: Dict[int, Counter] = {}
    for i, j, k in coords:
        sv = int(sv_ids[i, j, k])
        cls = int(gt[i, j, k])
        if (ignore_class is not None) and (cls == ignore_class):
            continue
        if sv not in votes:
            votes[sv] = Counter()
        votes[sv][cls] += 1
    # Assign majority class per seeded SV
    n_filled_svs = 0
    for sv, ctr in votes.items():
        if not ctr:
            continue
        cls = int(ctr.most_common(1)[0][0])
        out[sv_ids == sv] = cls
        n_filled_svs += 1
    fill_fraction = float((out != 0).sum()) / float(out.size)
    return out, n_filled_svs, fill_fraction


def assign_all_from_gt(
    sv_ids: np.ndarray,
    gt_labels: np.ndarray,
    ignore_class: Optional[int],
) -> Tuple[np.ndarray, int, float]:
    """Assign a label to every supervoxel based on majority vote over all voxels.

    - Respects ``ignore_class`` by excluding those voxels from voting.
    - If all voxels in a supervoxel are ignored, assigns 0.
    Returns ``(labels, n_nonzero_svs, fill_fraction)`` where ``labels`` has
    the same shape as ``sv_ids`` and contains per-voxel supervoxel labels.
    """
    X, Y, Z = sv_ids.shape
    out = np.zeros((X, Y, Z), dtype=np.int16)
    gt = gt_labels
    if gt.ndim == 4:
        gt = gt[0]
    # Precompute unique supervoxel ids
    uniq = np.unique(sv_ids)
    n_nonzero_svs = 0
    for sv in uniq:
        mask = (sv_ids == sv)
        vals = gt[mask].astype(np.int64)
        if ignore_class is not None:
            vals = vals[vals != ignore_class]
        if vals.size == 0:
            cls = 0
        else:
            # Compute majority with stable tie-breaker: pick the smallest label among maxima
            u, c = np.unique(vals, return_counts=True)
            maxc = c.max()
            cls = int(u[c == maxc].min())
        out[mask] = cls
        if cls != 0:
            n_nonzero_svs += 1
    fill_fraction = float((out != 0).sum()) / float(out.size)
    return out, int(n_nonzero_svs), float(fill_fraction)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    sup_dir = Path(args.sup_dir) if args.sup_dir else None
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = load_ids(data_root, Path(args.datalist) if args.datalist else None, args.ids)
    img_dir = data_root / "data"

    for cid in ids:
        t0 = time.time()
        ip = img_dir / f"{cid}_image.nii"
        lp = img_dir / f"{cid}_label.nii"
        if not ip.exists() or not lp.exists():
            print(f"skip {cid}: missing image/label")
            continue
        # Choose header for orientation + spacing
        ref_nii_path = lp if args.ref_header == "label" else ip
        spacing_xyz, xform = nifti_meta(ref_nii_path) if _HAS_NIB else ((1.0, 1.0, 1.0), None)

        # Load arrays
        if not _HAS_NIB:
            raise RuntimeError("nibabel is required: pip install -U nibabel")
        img = nib.load(str(ip))
        lab = nib.load(str(lp))
        img_arr = np.asarray(img.get_fdata()).astype(np.float32)
        lab_arr = np.asarray(lab.get_fdata()).astype(np.int16)
        # Reorient to RAS per chosen reference
        img_arr = apply_ornt(img_arr, xform)
        lab_arr = apply_ornt(lab_arr, xform)
        # Optional seeds
        seed_arr = None
        if sup_dir and (sup_dir / f"{cid}_seedmask.npy").exists():
            seed_src = np.load(str(sup_dir / f"{cid}_seedmask.npy"))
            seed_arr = apply_ornt(seed_src, xform)

        # Normalize and construct features per mode
        img_norm = pclip_zscore(img_arr)

        channel_axis = None
        img_for_slic: np.ndarray
        if args.mode == "slic":
            img_for_slic = img_norm
            channel_axis = None
        else:
            # Compute gradient-based features
            try:
                from skimage.filters import gaussian  # type: ignore
                img_smooth = gaussian(img_norm, sigma=float(args.grad_sigma), preserve_range=True)
            except Exception:
                img_smooth = img_norm
            dx, dy, dz = spacing_xyz
            gx, gy, gz = np.gradient(img_smooth, dx, dy, dz)
            if args.mode == "slic-grad-mag":
                gmag = np.sqrt(gx * gx + gy * gy + gz * gz).astype(np.float32)
                # z-score each channel for balance
                ch0 = pclip_zscore(img_norm)
                ch1 = pclip_zscore(gmag)
                img_for_slic = np.stack([ch0, ch1], axis=-1)
            elif args.mode == "slic-grad-vec":
                chs = [pclip_zscore(c.astype(np.float32)) for c in (img_norm, gx, gy, gz)]
                img_for_slic = np.stack(chs, axis=-1)
            else:
                img_for_slic = img_norm
            channel_axis = -1

        sv_ids = compute_supervoxels(
            img=img_for_slic,
            n_segments=args.n_segments,
            compactness=args.compactness,
            sigma=args.sigma,
            spacing_xyz=spacing_xyz,
            channel_axis=channel_axis,
        )

        # Label assignment mode
        if args.assign_all_from_gt:
            labels, n_filled_svs, fill_frac = assign_all_from_gt(
                sv_ids=sv_ids,
                gt_labels=lab_arr,
                ignore_class=args.ignore_class,
            )
            mode = "full_gt"
        elif args.no_fill:
            labels = np.zeros_like(lab_arr, dtype=np.int16)
            n_filled_svs = 0
            fill_frac = 0.0
            mode = "no_fill"
        else:
            labels, n_filled_svs, fill_frac = fill_from_seeds(
                sv_ids=sv_ids,
                seedmask=seed_arr,
                gt_labels=lab_arr,
                ignore_class=args.ignore_class,
            )
            mode = "seed_fill"

        # Save
        np.save(str(out_dir / f"{cid}_sv_ids.npy"), sv_ids.astype(np.int32))
        np.save(str(out_dir / f"{cid}_labels.npy"), labels.astype(np.int16))
        meta: Dict = {
            "id": cid,
            "shape": list(sv_ids.shape),
            "n_voxels": int(sv_ids.size),
            "n_svs": int(np.unique(sv_ids).size),
            "n_filled_svs": int(n_filled_svs),
            "fill_fraction": float(fill_frac),
            "seed_fraction": float(float(seed_arr.astype(bool).sum()) / float(seed_arr.size)) if seed_arr is not None else 0.0,
            "spacing_xyz": list(spacing_xyz),
            "orientation": "RAS",
            "mode": mode,
            "slic_mode": args.mode,
            "grad_sigma": float(args.grad_sigma),
            "seconds": float(time.time() - t0),
        }
        (out_dir / f"{cid}_sv_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"{cid}: sv={meta['n_svs']} filled={meta['n_filled_svs']} fill_frac={meta['fill_fraction']:.4f} time={meta['seconds']:.2f}s")


if __name__ == "__main__":
    main()
