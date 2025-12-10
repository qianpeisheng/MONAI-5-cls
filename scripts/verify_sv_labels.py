#!/usr/bin/env python3
"""
Verify supervoxel files for WP5: consistency and agreement with GT labels.

Checks per case id:
- Presence and counts of labeled classes (including 0 background)
- Shape agreement between sv_ids, labeled sv array, and GT label volume
- Per-supervoxel consistency (does each SV have a single assigned label?)
- Per-supervoxel GT-majority vs assigned label match rate
- Per-voxel agreement with GT (ignoring an optional class, e.g., 6)

Writes a summary CSV and per-case JSON into an output folder.

Usage example (your current folders):
  python3 scripts/verify_sv_labels.py \
    --sv-dir /home/peisheng/MONAI/runs/sv_fill_5k_nofill_ras2 \
    --sv-labeled-dir /home/peisheng/MONAI/runs/sv_fullgt_5k_ras2 \
    --data-root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
    --datalist datalist_train.json \
    --out-dir runs/verify_sv_labels_5k \
    --ignore-class 6
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import nibabel as nib
    from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
    _HAS_NIB = True
except Exception:
    _HAS_NIB = False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sv-dir", required=True, help="Directory with <id>_sv_ids.npy (unlabeled)")
    ap.add_argument("--sv-labeled-dir", required=True, help="Directory with <id>_labels.npy (fully labeled SV)")
    ap.add_argument("--data-root", required=True, help="Root with data/<id>_(image|label).nii")
    ap.add_argument("--datalist", default="", help="Optional datalist JSON to restrict case ids")
    ap.add_argument("--ids", default="", help="Optional comma-separated ids to restrict")
    ap.add_argument("--ignore-class", type=int, default=None, help="GT class to ignore (e.g., 6)")
    ap.add_argument("--out-dir", default="runs/verify_sv_labels", help="Output folder for CSV/JSON")
    ap.add_argument("--assume-ras", action="store_true", help="Assume arrays are already RAS; do not reorient GT")
    ap.add_argument("--sv-only", action="store_true", help="Only compute class distribution of labeled SV volumes (no GT checks)")
    return ap.parse_args()


def load_datalist_ids(datalist: Optional[Path]) -> set[str]:
    if not datalist or not datalist.exists():
        return set()
    try:
        lst = json.loads(datalist.read_text())
        return {str(rec.get("id")) for rec in lst if rec.get("id")}
    except Exception:
        return set()


def discover_ids(sv_dir: Path, labeled_dir: Path) -> set[str]:
    ids: set[str] = set()
    for p in sv_dir.glob("*_sv_ids.npy"):
        name = p.name
        if name.endswith("_sv_ids.npy"):
            ids.add(name[: -len("_sv_ids.npy")])
    for p in labeled_dir.glob("*_labels.npy"):
        name = p.name
        if name.endswith("_labels.npy"):
            ids.add(name[: -len("_labels.npy")])
    return ids


def nifti_to_ras_label(lbl_path: Path) -> np.ndarray:
    if not _HAS_NIB:
        raise RuntimeError("nibabel is required to read NIfTI")
    nii = nib.load(str(lbl_path))
    arr = np.asarray(nii.get_fdata()).astype(np.int64)
    A = nii.affine
    native = io_orientation(A)
    ras = axcodes2ornt(("R", "A", "S"))
    xform = ornt_transform(native, ras)
    arr_ras = nib.orientations.apply_orientation(arr, xform)
    return arr_ras


def per_sv_stats(sv_ids: np.ndarray, sv_labels: np.ndarray, gt_ras: np.ndarray, ignore_class: Optional[int]) -> Dict:
    # Basic sanity
    assert sv_ids.shape == sv_labels.shape == gt_ras.shape, "Shape mismatch among arrays"
    X, Y, Z = sv_ids.shape
    nvox = int(X * Y * Z)

    # Class distribution in assigned labels
    uniq_lbl, cnt_lbl = np.unique(sv_labels, return_counts=True)
    class_counts = {int(k): int(v) for k, v in zip(uniq_lbl.tolist(), cnt_lbl.tolist())}

    # SV consistency: does each SV contain a single assigned class?
    ids = np.unique(sv_ids)
    inconsistent_sv = 0
    for sv in ids:
        m = (sv_ids == sv)
        u = np.unique(sv_labels[m])
        if u.size > 1:
            inconsistent_sv += 1

    # Agreement with GT (voxel-wise), optionally ignoring a class
    gt_cmp = gt_ras.copy()
    if ignore_class is not None:
        mask = (gt_cmp == ignore_class)
        total_cmp = int((~mask).sum())
        vox_match = int(((sv_labels == gt_cmp) & (~mask)).sum())
    else:
        total_cmp = nvox
        vox_match = int((sv_labels == gt_cmp).sum())
    vox_acc = float(vox_match) / float(total_cmp) if total_cmp > 0 else 0.0

    # Per-SV purity wrt GT and majority label match
    # Compute for SVs that have any non-ignored voxels
    purity_sum = 0.0
    purity_cnt = 0
    majority_match_vox = 0
    for sv in ids:
        m = (sv_ids == sv)
        if ignore_class is not None:
            m = m & (gt_ras != ignore_class)
        total = int(m.sum())
        if total == 0:
            continue
        vals = gt_ras[m]
        u, c = np.unique(vals, return_counts=True)
        j = int(np.argmax(c))
        maj = int(u[j])
        maj_cnt = int(c[j])
        purity = float(maj_cnt) / float(total)
        purity_sum += purity
        purity_cnt += 1
        # Majority label vs assigned (decide assigned label for this SV via mode)
        assigned = int(np.bincount(sv_labels[m]).argmax())
        if assigned == maj:
            majority_match_vox += total

    purity_mean = float(purity_sum) / float(purity_cnt) if purity_cnt > 0 else 0.0
    majority_match_rate = float(majority_match_vox) / float(total_cmp) if total_cmp > 0 else 0.0

    return dict(
        n_voxels=nvox,
        n_svs=int(ids.size),
        class_counts=class_counts,
        inconsistent_sv=int(inconsistent_sv),
        voxel_acc=vox_acc,
        purity_mean=purity_mean,
        majority_match_rate=majority_match_rate,
    )


def main() -> None:
    args = parse_args()
    sv_dir = Path(args.sv_dir)
    sv_labeled_dir = Path(args.sv_labeled_dir)
    data_root = Path(args.data_root)
    img_root = data_root / "data"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = discover_ids(sv_dir, sv_labeled_dir)
    ids_dl = load_datalist_ids(Path(args.datalist)) if args.datalist else set()
    if ids_dl:
        ids &= ids_dl
    if args.ids:
        ids_cli = {s for s in args.ids.split(",") if s}
        ids &= ids_cli
    ids = sorted(ids)
    if not ids:
        print("No IDs found to verify.")
        return

    # Quick SV-only distribution mode
    csv_svonly = out_dir / "summary_svonly.csv"
    with csv_svonly.open("w", newline="") as fp2:
        w2 = csv.writer(fp2)
        w2.writerow(["id", "n_svs", "sv_class_counts", "sv_bg_fraction", "sv_bg_is_majority", "sv_top_class", "sv_top_fraction"])
        total_sv_counts: Dict[int, int] = {}
        total_svs = 0
        for cid in ids:
            sv_lbl_p = sv_labeled_dir / f"{cid}_labels.npy"
            sv_ids_p = sv_dir / f"{cid}_sv_ids.npy"
            if not (sv_lbl_p.exists() and sv_ids_p.exists()):
                print(f"skip {cid}: missing {sv_lbl_p} or {sv_ids_p}")
                continue
            try:
                sv_lbl = np.load(str(sv_lbl_p))
                sv_ids = np.load(str(sv_ids_p))
                if sv_lbl.ndim == 4 and sv_lbl.shape[0] == 1:
                    sv_lbl = sv_lbl[0]
                if sv_ids.ndim == 4 and sv_ids.shape[0] == 1:
                    sv_ids = sv_ids[0]
                # Flatten once for speed
                s = sv_ids.ravel().astype(np.int64)
                y = sv_lbl.ravel().astype(np.int64)
                # Map supervoxel ids to 0..K-1
                sv_unique, sv_idx = np.unique(s, return_inverse=True)
                K = sv_unique.size
                # Sort by sv_idx to compute per-SV mode efficiently
                order = np.argsort(sv_idx)
                sv_idx_sorted = sv_idx[order]
                y_sorted = y[order]
                seg_starts = np.r_[0, np.flatnonzero(np.diff(sv_idx_sorted)) + 1]
                seg_ends = np.r_[seg_starts[1:], sv_idx_sorted.size]
                # Precompute set of possible labels to bound bincount
                labels_all = np.unique(y_sorted)
                label_min = int(labels_all.min())
                label_max = int(labels_all.max())
                label_range = label_max - label_min + 1
                sv_class_counts: Dict[int, int] = {}
                for st, en in zip(seg_starts, seg_ends):
                    seg = y_sorted[st:en]
                    # Shift labels to 0..R-1 for bincount
                    cnt = np.bincount((seg - label_min), minlength=label_range)
                    lab_idx = int(cnt.argmax())
                    lab_val = int(lab_idx + label_min)
                    sv_class_counts[lab_val] = sv_class_counts.get(lab_val, 0) + 1
                n_svs = int(K)
                # Update totals
                for k, v in sv_class_counts.items():
                    total_sv_counts[k] = total_sv_counts.get(k, 0) + v
                total_svs += n_svs
                # Background majority stats over SVs
                bg_sv = sv_class_counts.get(0, 0)
                sv_bg_fraction = float(bg_sv) / float(n_svs) if n_svs > 0 else 0.0
                if sv_class_counts:
                    top_cls, top_cnt = max(sv_class_counts.items(), key=lambda kv: kv[1])
                    sv_top_class = int(top_cls)
                    sv_top_fraction = float(top_cnt) / float(n_svs) if n_svs > 0 else 0.0
                else:
                    sv_top_class, sv_top_fraction = -1, 0.0
                sv_bg_is_majority = int(bg_sv >= (sv_class_counts.get(sv_top_class, 0)))
                w2.writerow([
                    cid,
                    n_svs,
                    json.dumps({int(k): int(v) for k, v in sv_class_counts.items()}),
                    f"{sv_bg_fraction:.6f}",
                    sv_bg_is_majority,
                    sv_top_class,
                    f"{sv_top_fraction:.6f}",
                ])
            except Exception as e:
                print(f"error {cid}: {e}")
        # Totals row
        if total_svs > 0:
            bg_total = total_sv_counts.get(0, 0)
            top_k_tot = max(total_sv_counts.items(), key=lambda kv: kv[1])[0] if total_sv_counts else -1
            top_frac_tot = float(total_sv_counts.get(top_k_tot, 0)) / float(total_svs)
            w2.writerow([
                "ALL",
                total_svs,
                json.dumps({int(k): int(v) for k, v in total_sv_counts.items()}),
                f"{(bg_total/total_svs):.6f}",
                int(bg_total >= max(total_sv_counts.values()) if total_sv_counts else 0),
                int(top_k_tot),
                f"{top_frac_tot:.6f}",
            ])

    print(f"SV-only supervoxel distribution written to: {csv_svonly}")

    # If requested, also run full GT-consistency checks
    if not args.sv_only:
        csv_path = out_dir / "summary.csv"
        with csv_path.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow([
                "id", "shape", "n_voxels", "n_svs", "class_counts", "inconsistent_sv",
                "voxel_acc", "purity_mean", "majority_match_rate",
            ])

            for cid in ids:
                sv_ids_p = sv_dir / f"{cid}_sv_ids.npy"
                sv_lbl_p = sv_labeled_dir / f"{cid}_labels.npy"
                gt_lbl_p = img_root / f"{cid}_label.nii"
                if not (sv_ids_p.exists() and sv_lbl_p.exists() and gt_lbl_p.exists()):
                    print(f"skip {cid}: missing files")
                    continue
                try:
                    sv_ids = np.load(str(sv_ids_p))
                    sv_lbl = np.load(str(sv_lbl_p))
                    gt = nifti_to_ras_label(gt_lbl_p) if not args.assume_ras else np.asarray(nib.load(str(gt_lbl_p)).get_fdata()).astype(np.int64)
                    # Ensure 3D arrays
                    if sv_ids.ndim == 4 and sv_ids.shape[0] == 1:
                        sv_ids = sv_ids[0]
                    if sv_lbl.ndim == 4 and sv_lbl.shape[0] == 1:
                        sv_lbl = sv_lbl[0]
                    if gt.ndim == 4 and gt.shape[0] == 1:
                        gt = gt[0]
                    if sv_ids.shape != sv_lbl.shape:
                        print(f"warn {cid}: shape mismatch sv_ids {sv_ids.shape} vs sv_labels {sv_lbl.shape}")
                    if gt.shape != sv_lbl.shape:
                        print(f"warn {cid}: GT label shape {gt.shape} differs from sv_labels {sv_lbl.shape}")
                    shp = tuple(sv_ids.shape)
                    stats = per_sv_stats(sv_ids, sv_lbl, gt, args.ignore_class)
                    (out_dir / f"{cid}.json").write_text(json.dumps({"id": cid, "shape": shp, **stats}, indent=2))
                    writer.writerow([
                        cid,
                        shp,
                        stats["n_voxels"],
                        stats["n_svs"],
                        json.dumps(stats["class_counts"]),
                        stats["inconsistent_sv"],
                        f"{stats['voxel_acc']:.6f}",
                        f"{stats['purity_mean']:.6f}",
                        f"{stats['majority_match_rate']:.6f}",
                    ])
                except Exception as e:
                    print(f"error {cid}: {e}")
        print(f"GT-consistency summary written to: {csv_path}")


if __name__ == "__main__":
    main()
