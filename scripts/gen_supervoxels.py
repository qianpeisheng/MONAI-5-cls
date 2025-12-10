#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

# Ensure repository root (parent of this scripts/ folder) is on sys.path so
# `wp5` package is importable even when invoking via absolute path.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from wp5.weaklabel.sv_utils import (
    load_image,
    load_pseudolabel,
    load_seed_mask,
    seed_labels_from_mask_and_pseudo,
    majority_fill,
)
from wp5.weaklabel.supervoxels import run_supervoxels


def _write_outputs(out_root: Path, res: Dict[str, object]) -> None:
    cid = res["id"]
    sv_path = out_root / f"{cid}_sv_ids.npy"
    lab_path = out_root / f"{cid}_labels.npy"
    meta_path = out_root / f"{cid}_sv_meta.json"
    # Save arrays
    np.save(sv_path, res["sv_ids"])
    np.save(lab_path, res["dense_labels"])
    # Save meta
    meta = {k: v for k, v in res.items() if k not in {"sv_ids", "dense_labels"}}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def _fmt_case_log(res: Dict[str, object]) -> str:
    sid = res["id"]
    nsv = res["n_svs"]
    before = res["seed_fraction"]
    after = res["fill_fraction"]
    sec = res["seconds"]
    return (
        f"{sid}: SVs={nsv} seed%={before:.6f} fill%={after:.6f} time={sec:.2f}s"
    )


def _row_from_res(res: Dict[str, object]) -> Dict[str, object]:
    return {
        "id": res["id"],
        "shape": str(tuple(res["shape"])),
        "n_voxels": res["n_voxels"],
        "n_svs": res["n_svs"],
        "n_filled_svs": res["n_filled_svs"],
        "seed_fraction": f"{res['seed_fraction']:.8f}",
        "fill_fraction": f"{res['fill_fraction']:.8f}",
        "seconds": f"{res['seconds']:.3f}",
    }


def _worker_run(job: Tuple[str, str, str, str], args_dict: Dict[str, object]) -> Dict[str, object]:
    """Top-level worker for ProcessPoolExecutor.

    Writes outputs inside the worker to avoid pickling large arrays back.
    Returns only metadata for CSV/logging.
    """
    cid, img_path_s, seedmask_path_s, pseudolabel_path_s = job
    img_path = Path(img_path_s)
    seedmask_path = Path(seedmask_path_s)
    pseudolabel_path = Path(pseudolabel_path_s)
    out_root = Path(args_dict["out_root"])  # type: ignore[arg-type]

    t0 = perf_counter()
    img = load_image(str(img_path), normalize=True)
    seedmask = load_seed_mask(str(seedmask_path))
    pseudo = load_pseudolabel(str(pseudolabel_path))
    if img.shape != seedmask.shape:
        raise ValueError(f"Shape mismatch for {cid}: image {img.shape} vs seeds {seedmask.shape}")
    if pseudo.shape != img.shape:
        raise ValueError(f"Shape mismatch for {cid}: image {img.shape} vs pseudolabel {pseudo.shape}")

    seed_labels = seed_labels_from_mask_and_pseudo(
        seedmask, pseudo, ignore_classes=set(args_dict["ignore_classes"])  # type: ignore[arg-type]
    )

    sv_ids = run_supervoxels(
        img,
        n_segments=int(args_dict["n_segments"]),
        compactness=float(args_dict["compactness"]),
        sigma=float(args_dict["sigma"]),
        downsample=int(args_dict["downsample"]),
        enforce_connectivity=True,
    )
    n_svs = int(np.unique(sv_ids).size)

    dense_labels, n_filled = majority_fill(
        seed_labels,
        sv_ids,
        unlabeled_values=list(args_dict["unlabeled_values"]),  # type: ignore[list-item]
        tie_policy=str(args_dict["tie_policy"]),
        output_unlabeled_value=int(args_dict["output_unlabeled_value"]),
    )

    total = int(np.prod(img.shape))
    seed_count = int(seedmask.sum())
    before = seed_count / float(total)
    after = float(np.mean(dense_labels != int(args_dict["output_unlabeled_value"])) )
    dt = perf_counter() - t0

    # Write outputs inside worker
    np.save(out_root / f"{cid}_sv_ids.npy", sv_ids)
    np.save(out_root / f"{cid}_labels.npy", dense_labels)
    meta = {
        "id": cid,
        "shape": list(img.shape),
        "n_voxels": total,
        "n_svs": n_svs,
        "n_filled_svs": int(n_filled),
        "seed_fraction": before,
        "fill_fraction": after,
        "seconds": dt,
    }
    with open(out_root / f"{cid}_sv_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Generate 3D supervoxels and intra-SV majority label propagation from sparse seeds.\n"
            "Inputs: images under --img-root and seeds under --seed-root (expects *_seedmask.npy and *_pseudolabel.npy).\n"
            "Outputs: *_sv_ids.npy and *_labels.npy per case, plus *_sv_meta.json."
        )
    )
    ap.add_argument("--img-root", required=True, help="Root dir containing image volumes (e.g., NIfTI or .npy)")
    ap.add_argument(
        "--seed-root",
        required=True,
        help=(
            "Root dir containing seeds and pseudolabels (.npy). Must include *_seedmask.npy (bool) and *_pseudolabel.npy (int)."
        ),
    )
    ap.add_argument("--out", default="data/pseudo/sv_fill", help="Output root directory")
    # SLIC params
    ap.add_argument("--n-segments", type=int, default=10000, help="Approximate number of supervoxels")
    ap.add_argument("--compactness", type=float, default=0.05, help="SLIC compactness")
    ap.add_argument("--sigma", type=float, default=0.5, help="Pre-smoothing sigma for SLIC")
    ap.add_argument("--downsample", type=int, default=1, help="Integer downsample factor (1=no downsample)")
    # Policies
    ap.add_argument(
        "--tie-policy",
        default="skip",
        choices=["skip", "min", "max", "prefer_foreground"],
        help="Tie-break policy for SV majority vote",
    )
    ap.add_argument(
        "--unlabeled-values",
        nargs="*",
        type=int,
        default=[-1, 6],
        help="Seed label values to treat as unlabeled/ignored in majority fill",
    )
    ap.add_argument(
        "--output-unlabeled-value",
        type=int,
        default=6,
        help="Value to assign to voxels in SVs without labeled seeds (default 6 to align with WP5 ignore)",
    )
    ap.add_argument(
        "--ignore-classes",
        nargs="*",
        type=int,
        default=[6],
        help="Label classes to ignore when deriving seed labels from pseudolabels (e.g., 6)",
    )
    ap.add_argument(
        "--ext-priority",
        default=".nii.gz,.nii,.npy",
        help="Comma-separated image filename suffixes to try (in order)",
    )
    ap.add_argument("--workers", type=int, default=4, help="Number of parallel workers (process-level)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--log-to-file", action="store_true", help="Duplicate console logs to out/run.log")
    return ap.parse_args()


def find_image_for_id(img_root: Path, case_id: str, ext_priority: List[str]) -> Optional[Path]:
    # Try known patterns: {id}_image.<ext> then {id}.<ext>
    for ext in ext_priority:
        cand = img_root / f"{case_id}_image{ext}"
        if cand.exists():
            return cand
    for ext in ext_priority:
        cand = img_root / f"{case_id}{ext}"
        if cand.exists():
            return cand
    return None


def enumerate_cases(seed_root: Path) -> List[Tuple[str, Path, Path]]:
    # Returns list of (case_id, seedmask_path, pseudolabel_path)
    items: List[Tuple[str, Path, Path]] = []
    for p in seed_root.glob("*_seedmask.npy"):
        case_id = p.name.replace("_seedmask.npy", "")
        pl = seed_root / f"{case_id}_pseudolabel.npy"
        if not pl.exists():
            # skip if pseudolabel missing; intra-SV per-class fill needs it.
            continue
        items.append((case_id, p, pl))
    return sorted(items)


def process_case(
    case_id: str,
    img_path: Path,
    seedmask_path: Path,
    pseudolabel_path: Path,
    args: argparse.Namespace,
) -> Dict[str, object]:
    # Serial path uses the same logic as worker
    args_dict = {
        "out_root": str(Path(args.out)),
        "n_segments": args.n_segments,
        "compactness": args.compactness,
        "sigma": args.sigma,
        "downsample": args.downsample,
        "tie_policy": args.tie_policy,
        "unlabeled_values": args.unlabeled_values,
        "output_unlabeled_value": args.output_unlabeled_value,
        "ignore_classes": args.ignore_classes,
    }
    job = (case_id, str(img_path), str(seedmask_path), str(pseudolabel_path))
    return _worker_run(job, args_dict)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    img_root = Path(args.img_root)
    seed_root = Path(args.seed_root)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # logging
    log_fp = None
    if args.log_to_file:
        log_fp = open(out_root / "run.log", "w", buffering=1)

    def log(msg: str) -> None:
        print(msg, flush=True)
        if log_fp is not None:
            print(msg, file=log_fp, flush=True)

    cases = enumerate_cases(seed_root)
    if not cases:
        log(f"No cases found in {seed_root} (expected '*_seedmask.npy' and '*_pseudolabel.npy').")
        return

    ext_priority = [s.strip() for s in args.ext_priority.split(",") if s.strip()]
    jobs: List[Tuple[str, Path, Path, Path]] = []
    for cid, seed_p, pseudo_p in cases:
        img_p = find_image_for_id(img_root, cid, ext_priority)
        if img_p is None:
            log(f"[WARN] Missing image for {cid} under {img_root}; skipping.")
            continue
        jobs.append((cid, img_p, seed_p, pseudo_p))

    if not jobs:
        log("No paired image+seed cases to process; aborting.")
        return

    # Prepare summary csv
    csv_path = out_root / "summary.csv"
    csv_f = open(csv_path, "w", newline="")
    writer = csv.DictWriter(
        csv_f,
        fieldnames=[
            "id",
            "shape",
            "n_voxels",
            "n_svs",
            "n_filled_svs",
            "seed_fraction",
            "fill_fraction",
            "seconds",
        ],
    )
    writer.writeheader()

    log(
        f"Processing {len(jobs)} cases with n_segments={args.n_segments} compactness={args.compactness} "
        f"sigma={args.sigma} downsample={args.downsample} workers={args.workers}"
    )

    # Arguments to pass to subprocesses (ensure picklable types only)
    args_dict: Dict[str, object] = {
        "out_root": str(out_root),
        "n_segments": args.n_segments,
        "compactness": args.compactness,
        "sigma": args.sigma,
        "downsample": args.downsample,
        "tie_policy": args.tie_policy,
        "unlabeled_values": list(args.unlabeled_values),
        "output_unlabeled_value": args.output_unlabeled_value,
        "ignore_classes": list(args.ignore_classes),
    }

    results: List[Dict[str, object]] = []
    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_worker_run, (cid, str(ip), str(sp), str(pp)), args_dict): (cid, ip)
                    for (cid, ip, sp, pp) in jobs}
            for fut in as_completed(futs):
                res = fut.result()
                results.append(res)
                log(_fmt_case_log(res))
                writer.writerow(_row_from_res(res))
                csv_f.flush()
    else:
        for (cid, ip, sp, pp) in jobs:
            res = _worker_run((cid, str(ip), str(sp), str(pp)), args_dict)
            results.append(res)
            log(_fmt_case_log(res))
            writer.writerow(_row_from_res(res))
            csv_f.flush()

    # Final message before closing log file handle
    log(f"Done. Wrote outputs to {out_root} and summary to {csv_path}.")
    csv_f.close()
    if log_fp is not None:
        log_fp.close()




if __name__ == "__main__":
    main()
