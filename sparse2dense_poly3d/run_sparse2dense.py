from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .io_utils import (
    ensure_dir,
    get_spacing_from_header,
    load_label,
    mirror_out_path,
    parse_datalist,
    save_nifti_like,
    write_json,
)
from .reconstruct import reconstruct_from_points
from .sampler import select_points_1pct


def dice_per_class(pred: np.ndarray, gt: np.ndarray, classes=(0, 1, 2, 3, 4)) -> Dict[int, float]:
    out = {}
    for c in classes:
        p = pred == c
        g = gt == c
        inter = float((p & g).sum())
        denom = float(p.sum() + g.sum())
        d = (2.0 * inter / denom) if denom > 0 else (1.0 if p.sum() == g.sum() else 0.0)
        out[c] = d
    return out


def compute_hd95_monai(pred: np.ndarray, gt: np.ndarray, classes=(0, 1, 2, 3, 4)) -> Dict[int, float]:
    # Use MONAI's HausdorffDistanceMetric on CPU
    from monai.metrics import HausdorffDistanceMetric
    from monai.networks import one_hot

    # convert to one-hot [B=1, C, X, Y, Z]
    X, Y, Z = gt.shape
    num_classes = max(classes) + 1
    gt_clamped = np.clip(gt, 0, max(classes))
    pred_clamped = np.clip(pred, 0, max(classes))
    gtt = torch.from_numpy(gt_clamped[None, None]).long()
    prt = torch.from_numpy(pred_clamped[None, None]).long()
    gt_oh = one_hot(gtt, num_classes=num_classes).float()
    pr_oh = one_hot(prt, num_classes=num_classes).float()
    hd = HausdorffDistanceMetric(include_background=True, percentile=95.0, reduction="none")
    vals = hd(pr_oh, gt_oh)  # shape (1, C)
    vals = vals[0].cpu().numpy()
    return {c: float(vals[c]) for c in classes}


def main():
    ap = argparse.ArgumentParser(description="Sparse2Dense Polygon-Like 3D Label Generator (1% few-shot)")
    ap.add_argument("--dataset_root", type=str, required=True)
    ap.add_argument("--train_list", type=str, required=True)
    ap.add_argument("--out_pseudolabels_dir", type=str, required=True)
    ap.add_argument("--out_intermediates_dir", type=str, required=True)
    ap.add_argument("--budget_ratio", type=float, default=0.01)
    ap.add_argument("--boundary_ratio", type=float, default=0.7)
    ap.add_argument("--interior_ratio", type=float, default=0.2)
    ap.add_argument("--guard_ratio", type=float, default=0.1)
    ap.add_argument("--rbf_kernel", type=str, default="multiquadric")
    ap.add_argument("--rbf_eps", type=float, default=2.0)
    ap.add_argument("--margin_vox", type=float, default=1.5)
    ap.add_argument("--smooth", type=float, default=1e-3)
    ap.add_argument("--spacing_from_header", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--constraints", type=str, default="wp5_class_constraints.yaml", help="YAML constraints file (precedence, disjoint, inclusive)")
    ap.add_argument("--include_background", type=lambda s: s.lower() == "true", default=True, help="Include background (class 0) in point selection (small ratio)")
    ap.add_argument("--bg_ratio", type=float, default=0.05, help="Fraction of total point budget reserved for background points")
    ap.add_argument("--limit", type=int, default=-1, help="limit number of cases for a dry run")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    train = parse_datalist(args.train_list)
    if args.limit > 0:
        train = train[: args.limit]

    out_pl = Path(args.out_pseudolabels_dir)
    out_mid = Path(args.out_intermediates_dir)
    ensure_dir(out_pl)
    ensure_dir(out_mid)

    # CSV summary
    summary_csv = out_mid / "dataset_summary.csv"
    if not summary_csv.exists():
        with open(summary_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "dice_avg_0_4", "hd95_avg_0_4"])

    # Load constraints if available
    constraints = None
    if args.constraints:
        cpath = Path(args.constraints)
        if cpath.exists():
            try:
                import yaml  # type: ignore
                constraints = yaml.safe_load(cpath.read_text())
            except Exception as e:
                print(f"Failed to load constraints from {cpath}: {e}")

    for rec in train:
        lid = rec.get("id") or Path(rec["label"]).stem.replace("_label", "")
        lbl_path = rec["label"]
        print(f"Processing {lid} ...")
        gt, affine, header = load_label(lbl_path)
        spacing = get_spacing_from_header(header) if args.spacing_from_header else (1.0, 1.0, 1.0)

        # Handle ignore class 6 by preserving it verbatim later
        mask_ignore6 = (gt == 6)
        gt_work = gt.copy()
        gt_work[mask_ignore6] = 0  # exclude from point selection/recon

        # Select points and build selected mask
        points, sel_mask, counts = select_points_1pct(
            gt_work,
            classes=(1, 2, 3, 4),
            budget_ratio=args.budget_ratio,
            include_background=args.include_background,
            bg_ratio=args.bg_ratio,
            boundary_ratio=args.boundary_ratio,
            interior_ratio=args.interior_ratio,
            guard_ratio=args.guard_ratio,
            margin_vox=args.margin_vox,
            rng=rng,
        )

        # Save intermediates for this case
        case_dir = out_mid / str(lid)
        ensure_dir(case_dir)
        np.savez_compressed(case_dir / "selected_points.npz", **{str(k): {"coords": v.coords, "sdf": v.sdf} for k, v in points.items()})
        save_nifti_like(case_dir / "mask_selected_points.nii", sel_mask.astype(np.int16), affine, header=header, dtype=np.int16)

        # Reconstruct pseudo label
        pred = reconstruct_from_points(
            shape=gt_work.shape,
            points=points,
            rbf_kernel=args.rbf_kernel,
            rbf_eps=args.rbf_eps,
            smooth=args.smooth,
            bbox_margin=4,
            chunk_z=32,
            constraints=constraints,
        )
        # Restore ignore class 6 region verbatim
        pred[mask_ignore6] = 6

        # Save pseudo label with mirrored name/affine
        out_path = mirror_out_path(out_pl, Path(lbl_path))
        save_nifti_like(out_path, pred.astype(np.int16), affine, header=header, dtype=np.int16)

        # Sidecar JSON
        write_json(
            case_dir / f"{lid}.json",
            {
                "selected_points_counts": counts,
                "selected_mask_relpath": str((case_dir / "mask_selected_points.nii").relative_to(out_mid)),
                "budget_ratio": args.budget_ratio,
                "include_background": bool(args.include_background),
                "bg_ratio": float(args.bg_ratio),
                "constraints": str(args.constraints),
                "source_label_relpath": str(Path(lbl_path)),
            },
        )

        # Metrics over 0..4 (ignore 6)
        d = dice_per_class(pred * (gt_work != 6), gt_work, classes=(0, 1, 2, 3, 4))
        try:
            h = compute_hd95_monai(pred * (gt_work != 6), gt_work, classes=(0, 1, 2, 3, 4))
            hd_avg = np.mean([h[c] for c in (0, 1, 2, 3, 4)])
        except Exception as e:
            print(f"HD95 metric failed ({e}); writing NaN")
            hd_avg = float("nan")
        dice_avg = float(np.mean([d[c] for c in (0, 1, 2, 3, 4)]))
        with open(summary_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([lid, dice_avg, hd_avg])

    print("Done.")


if __name__ == "__main__":
    main()
