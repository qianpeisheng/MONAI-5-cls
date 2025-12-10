#!/usr/bin/env python3
"""
Fill every supervoxel with a label from the complete GT by majority vote.

Inputs per case id (<id>):
  - SV ids (unlabeled): <sv_dir>/<id>_sv_ids.npy (RAS, shape (X,Y,Z) or (1,X,Y,Z))
  - GT label NIfTI: <data_root>/data/<id>_label.nii

Outputs to <out_dir>:
  - <id>_labels.npy          (int16, same shape as sv_ids, every SV labeled)
  - <id>_sv_meta.json        (n_svs, n_filled_svs==n_svs, bg_fraction, seconds, etc.)

Notes
- Reorients GT labels to RAS using the NIfTI affine to match RAS sv_ids.
- Majority vote is computed per supervoxel over GT labels; optional --ignore-class (e.g., 6) masks those voxels before voting.
- Efficient vectorized implementation: per-class bincount over sv_idx groups.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

try:
    import nibabel as nib
    from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
    _HAS_NIB = True
except Exception:
    _HAS_NIB = False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sv-dir", required=True, help="Folder with <id>_sv_ids.npy (unlabeled)")
    ap.add_argument("--out-dir", required=True, help="Output folder for fully labeled SV volumes")
    ap.add_argument("--data-root", required=True, help="Root containing data/<id>_label.nii")
    ap.add_argument("--datalist", default="", help="Optional datalist JSON with case 'id' entries")
    ap.add_argument("--ids", default="", help="Optional comma-separated ids to restrict")
    ap.add_argument("--ignore-class", type=int, default=None, help="GT class to ignore in voting (e.g., 6)")
    ap.add_argument("--assume-ras", action="store_true", help="Assume GT is already RAS (skip reorientation)")
    return ap.parse_args()


def load_datalist_ids(datalist: Optional[Path]) -> set[str]:
    if not datalist or not datalist.exists():
        return set()
    try:
        lst = json.loads(datalist.read_text())
        return {str(rec.get("id")) for rec in lst if rec.get("id")}
    except Exception:
        return set()


def discover_ids(sv_dir: Path) -> set[str]:
    ids: set[str] = set()
    for p in sv_dir.glob("*_sv_ids.npy"):
        n = p.name
        if n.endswith("_sv_ids.npy"):
            ids.add(n[: -len("_sv_ids.npy")])
    return ids


def nifti_to_ras_label(lbl_path: Path, assume_ras: bool = False) -> np.ndarray:
    if not _HAS_NIB:
        raise RuntimeError("nibabel is required to read NIfTI")
    nii = nib.load(str(lbl_path))
    arr = np.asarray(nii.get_fdata()).astype(np.int64)
    if assume_ras:
        return arr
    A = nii.affine
    native = io_orientation(A)
    ras = axcodes2ornt(("R", "A", "S"))
    xform = ornt_transform(native, ras)
    arr_ras = nib.orientations.apply_orientation(arr, xform)
    return arr_ras


def vote_labels_for_sv(sv_ids: np.ndarray, gt: np.ndarray, ignore_class: Optional[int]) -> Tuple[np.ndarray, Dict[str, float]]:
    """Assign each supervoxel an integer class by majority of GT labels.

    Returns the dense labeled volume (same shape as sv_ids) and basic stats.
    """
    if sv_ids.ndim == 4 and sv_ids.shape[0] == 1:
        sv_ids = sv_ids[0]
    if gt.ndim == 4 and gt.shape[0] == 1:
        gt = gt[0]
    assert sv_ids.shape == gt.shape, f"shape mismatch sv_ids {sv_ids.shape} vs gt {gt.shape}"

    s = sv_ids.ravel().astype(np.int64)
    y = gt.ravel().astype(np.int64)

    if ignore_class is not None:
        mask = (y != int(ignore_class))
        s = s[mask]
        y = y[mask]

    # Map supervoxel ids to 0..K-1 groups
    sv_unique, sv_idx = np.unique(s, return_inverse=True)
    K = int(sv_unique.size)

    # Compute per-SV majority via per-class bincount
    # Enumerate present labels to bound bincount length
    labels_all = np.unique(y)
    label_min = int(labels_all.min())
    label_max = int(labels_all.max())
    R = int(label_max - label_min + 1)

    # For each label value L, count occurrences per SV: bincount(sv_idx[y==L])
    counts = np.zeros((K, R), dtype=np.int64)
    for L in labels_all.tolist():
        sel = (y == int(L))
        if not np.any(sel):
            continue
        bc = np.bincount(sv_idx[sel], minlength=K)
        counts[:, int(L - label_min)] = bc

    # Majority label index per SV, map back to label values
    maj_idx = counts.argmax(axis=1)
    maj_val = (maj_idx + label_min).astype(np.int64)

    # Broadcast to voxels: labeled_s = maj_val[sv_group_of_voxel]
    # Recompute sv_idx_full for full volume (including ignored class voxels)
    s_full = sv_ids.ravel().astype(np.int64)
    # Map s_full values to positions in sv_unique via searchsorted (sv_unique is sorted)
    pos = np.searchsorted(sv_unique, s_full)
    labeled_full = maj_val[pos].reshape(sv_ids.shape)

    # Stats
    bg = int((labeled_full == 0).sum())
    totvox = int(labeled_full.size)
    stats = {
        "n_svs": float(K),
        "bg_fraction_vox": float(bg) / float(totvox) if totvox > 0 else 0.0,
    }
    return labeled_full.astype(np.int16, copy=False), stats


def main() -> None:
    args = parse_args()
    sv_dir = Path(args.sv_dir)
    out_dir = Path(args.out_dir)
    data_root = Path(args.data_root)
    img_root = data_root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = discover_ids(sv_dir)
    ids_dl = load_datalist_ids(Path(args.datalist)) if args.datalist else set()
    if ids_dl:
        ids &= ids_dl
    if args.ids:
        ids_cli = {s for s in args.ids.split(",") if s}
        ids &= ids_cli
    ids = sorted(ids)
    if not ids:
        print("No IDs found to process.")
        return

    for cid in ids:
        t0 = time.time()
        sv_ids_p = sv_dir / f"{cid}_sv_ids.npy"
        gt_lbl_p = img_root / f"{cid}_label.nii"
        if not (sv_ids_p.exists() and gt_lbl_p.exists()):
            print(f"skip {cid}: missing files")
            continue
        try:
            sv_ids = np.load(str(sv_ids_p))
            gt = nifti_to_ras_label(gt_lbl_p, assume_ras=args.assume_ras)
            labeled, stats = vote_labels_for_sv(sv_ids, gt, args.ignore_class)
            # Save outputs
            np.save(str(out_dir / f"{cid}_labels.npy"), labeled)
            meta: Dict = {
                "id": cid,
                "shape": list(labeled.shape),
                "n_voxels": int(labeled.size),
                "n_svs": int(np.unique(sv_ids if sv_ids.ndim==3 else sv_ids[0]).size),
                "n_filled_svs": int(np.unique(sv_ids if sv_ids.ndim==3 else sv_ids[0]).size),
                "bg_fraction_vox": stats.get("bg_fraction_vox", 0.0),
                "source": "fill_sv_from_gt_majority",
                "seconds": float(time.time() - t0),
            }
            (out_dir / f"{cid}_sv_meta.json").write_text(json.dumps(meta, indent=2))
            print(f"{cid}: n_svs={meta['n_svs']} wrote labels, time={meta['seconds']:.2f}s")
        except Exception as e:
            print(f"error {cid}: {e}")


if __name__ == "__main__":
    main()

