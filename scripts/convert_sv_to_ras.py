#!/usr/bin/env python3
"""
Convert saved supervoxel and supervision mask arrays to RAS orientation
using the source NIfTI affine from the dataset. Writes to sibling folders
with a configurable suffix (default: _ras) and updates meta JSON.

Usage example
  python3 scripts/convert_sv_to_ras.py \
    --data-root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
    --in-sv-dirs /home/peisheng/MONAI/runs/sv_fill_0p1pct /home/peisheng/MONAI/runs/sv_fill_0p1pct_n10k \
    --in-sup-dir /home/peisheng/MONAI/runs/sup_masks_0p1pct_global_d0_5_nov \
    --out-suffix _ras \
    --datalist datalist_train.json

Notes
- Arrays are assumed to be 3D (X,Y,Z) or 4D (1,X,Y,Z). If 4D with leading
  channel 1, the inner 3D is reoriented and re-expanded to 4D.
- Dtypes and shapes are preserved aside from orientation permutation/flips.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    ap.add_argument("--data-root", required=True, help="Root containing data/ with <id>_image.nii")
    ap.add_argument("--in-sv-dirs", nargs="*", default=[], help="One or more SV fill run folders")
    ap.add_argument("--in-sup-dir", default="", help="Optional sup_masks run folder (for seeds/supmask)")
    ap.add_argument("--out-suffix", default="_ras", help="Suffix to append to output folders")
    ap.add_argument("--datalist", default="", help="Optional datalist JSON with case 'id' entries")
    ap.add_argument("--ids", default="", help="Optional comma-separated list of case ids to restrict")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return ap.parse_args()


def load_datalist_ids(datalist: Optional[Path]) -> set[str]:
    if not datalist or not datalist.exists():
        return set()
    try:
        lst = json.loads(datalist.read_text())
        return {str(rec.get("id")) for rec in lst if rec.get("id")}
    except Exception:
        return set()


def discover_ids_in_dir(run_dir: Path, suffixes: Sequence[str]) -> set[str]:
    ids: set[str] = set()
    for suf in suffixes:
        for p in run_dir.glob(f"*{suf}"):
            b = p.name[: -len(suf)]
            ids.add(b)
    return ids


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


def apply_ornt(arr: np.ndarray, xform: Optional[np.ndarray]) -> np.ndarray:
    if xform is None:
        return arr
    if arr.ndim == 4 and arr.shape[0] == 1:
        inner = nib.orientations.apply_orientation(arr[0], xform)
        return inner[None]
    if arr.ndim == 3:
        return nib.orientations.apply_orientation(arr, xform)
    return arr


def ensure_out_dir(inp: Path, suffix: str) -> Path:
    out = inp.with_name(inp.name + suffix)
    out.mkdir(parents=True, exist_ok=True)
    return out


def process_sv_dir(run_dir: Path, out_dir: Path, ids: Iterable[str], img_root: Path, overwrite: bool) -> None:
    for cid in sorted(set(ids)):
        imgp = img_root / f"{cid}_image.nii"
        if not imgp.exists():
            print(f"[SV] skip {cid}: image not found {imgp}")
            continue
        spac, xform = nifti_meta(imgp)
        lab_in = run_dir / f"{cid}_labels.npy"
        ids_in = run_dir / f"{cid}_sv_ids.npy"
        meta_in = run_dir / f"{cid}_sv_meta.json"
        lab_out = out_dir / lab_in.name
        ids_out = out_dir / ids_in.name
        meta_out = out_dir / meta_in.name

        if (not overwrite) and lab_out.exists() and ids_out.exists():
            continue
        if not lab_in.exists():
            print(f"[SV] skip {cid}: labels missing {lab_in}")
            continue
        # Load and convert
        try:
            arr_lab = np.load(str(lab_in))
            arr_ids = np.load(str(ids_in)) if ids_in.exists() else None
            arr_lab_ras = apply_ornt(arr_lab, xform)
            arr_ids_ras = apply_ornt(arr_ids, xform) if arr_ids is not None else None
        except Exception as e:
            print(f"[SV] error {cid}: {e}")
            continue
        # Save
        np.save(str(lab_out), arr_lab_ras)
        if arr_ids_ras is not None:
            np.save(str(ids_out), arr_ids_ras)
        # Meta
        meta: Dict = {}
        if meta_in.exists():
            try:
                meta = json.loads(meta_in.read_text())
            except Exception:
                meta = {}
        meta["orientation"] = "RAS"
        meta["spacing_xyz"] = list(spac)
        meta["converted_from"] = str(run_dir)
        meta["shape"] = list(arr_lab_ras.shape[-3:])
        try:
            meta_out.write_text(json.dumps(meta, indent=2))
        except Exception as e:
            print(f"[SV] meta write failed for {cid}: {e}")


def process_sup_dir(run_dir: Path, out_dir: Path, ids: Iterable[str], img_root: Path, overwrite: bool) -> None:
    for cid in sorted(set(ids)):
        imgp = img_root / f"{cid}_image.nii"
        if not imgp.exists():
            print(f"[SUP] skip {cid}: image not found {imgp}")
            continue
        spac, xform = nifti_meta(imgp)
        seed_in = run_dir / f"{cid}_seedmask.npy"
        sup_in = run_dir / f"{cid}_supmask.npy"
        stats_in = run_dir / f"{cid}_supmask_stats.json"
        seed_out = out_dir / seed_in.name
        sup_out = out_dir / sup_in.name
        stats_out = out_dir / stats_in.name

        if (not overwrite) and seed_out.exists() and sup_out.exists():
            continue
        if not seed_in.exists() and not sup_in.exists():
            continue
        try:
            arr_seed = np.load(str(seed_in)) if seed_in.exists() else None
            arr_sup = np.load(str(sup_in)) if sup_in.exists() else None
            arr_seed_ras = apply_ornt(arr_seed, xform) if arr_seed is not None else None
            arr_sup_ras = apply_ornt(arr_sup, xform) if arr_sup is not None else None
        except Exception as e:
            print(f"[SUP] error {cid}: {e}")
            continue
        if arr_seed_ras is not None:
            np.save(str(seed_out), arr_seed_ras)
        if arr_sup_ras is not None:
            np.save(str(sup_out), arr_sup_ras)
        # Stats passthrough
        if stats_in.exists():
            try:
                stats = json.loads(stats_in.read_text())
                stats["orientation"] = "RAS"
                stats["spacing_xyz"] = list(spac)
                stats["converted_from"] = str(run_dir)
                stats_out.write_text(json.dumps(stats, indent=2))
            except Exception:
                pass


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    img_root = data_root / "data"
    if not img_root.exists():
        print(f"data root invalid: {img_root}", file=sys.stderr)
        sys.exit(1)

    # Build ID set
    ids_from_datalist = load_datalist_ids(Path(args.datalist)) if args.datalist else set()
    ids_from_cli = set([s for s in args.ids.split(",") if s]) if args.ids else set()

    sv_dirs = [Path(p) for p in args.in_sv_dirs if p]
    sup_dir = Path(args.in_sup_dir) if args.in_sup_dir else None

    sv_ids: set[str] = set()
    for d in sv_dirs:
        if d.exists():
            sv_ids |= discover_ids_in_dir(d, suffixes=["_labels.npy", "_sv_ids.npy"])
    sup_ids: set[str] = set()
    if sup_dir and sup_dir.exists():
        sup_ids = discover_ids_in_dir(sup_dir, suffixes=["_supmask.npy", "_seedmask.npy"])

    # Restrict by datalist/ids if provided
    def restrict(ids: set[str]) -> set[str]:
        if ids_from_cli:
            ids &= ids_from_cli
        if ids_from_datalist:
            ids &= ids_from_datalist
        return ids

    sv_ids = restrict(sv_ids)
    sup_ids = restrict(sup_ids)

    # Convert
    for d in sv_dirs:
        if not d.exists():
            print(f"skip missing SV dir: {d}")
            continue
        out_d = ensure_out_dir(d, args.out_suffix)
        process_sv_dir(d, out_d, sv_ids, img_root, args.overwrite)

    if sup_dir and sup_dir.exists():
        out_sup = ensure_out_dir(sup_dir, args.out_suffix)
        process_sup_dir(sup_dir, out_sup, sup_ids, img_root, args.overwrite)

    print("Done.")


if __name__ == "__main__":
    main()

