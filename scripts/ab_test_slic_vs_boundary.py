#!/usr/bin/env python3
"""
A/B test: SLIC vs Boundary-Preserving Supervoxels

Compares both methods on a single volume in two scenarios:
1. Full GT labels available (majority voting baseline)
2. 0.1% sparse GT labels (Graph LP propagation)

Usage:
    python scripts/ab_test_slic_vs_boundary.py \
        --case_id SN13B0_I17_3D_B1_1B250409 \
        --output_dir tmp/ab_test_slic_vs_boundary
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wp5.weaklabel.supervoxels import run_supervoxels
from wp5.weaklabel.sv_utils import majority_fill, relabel_sequential
from wp5.weaklabel.graph_label_propagation import graph_label_propagation


def compute_dice(pred: np.ndarray, gt: np.ndarray, class_id: int) -> float:
    """Compute Dice score for a specific class."""
    pred_mask = (pred == class_id)
    gt_mask = (gt == class_id)

    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return 2.0 * intersection / union


def compute_all_dice(pred: np.ndarray, gt: np.ndarray, num_classes: int = 5) -> dict:
    """Compute Dice for all classes and mean."""
    dice_scores = {}
    for cls in range(num_classes):
        dice_scores[f"dice_class_{cls}"] = compute_dice(pred, gt, cls)

    # Mean Dice (excluding background class 0)
    dice_scores["mean_dice"] = np.mean([dice_scores[f"dice_class_{i}"] for i in range(1, num_classes)])

    return dice_scores


def compute_sv_centroids(sv_ids: np.ndarray, unique_svs: np.ndarray) -> np.ndarray:
    """Compute supervoxel centroids as features."""
    Z, Y, X = sv_ids.shape
    N = len(unique_svs)
    centroids = np.zeros((N, 3), dtype=np.float32)

    for i, sv_id in enumerate(unique_svs):
        mask = (sv_ids == sv_id)
        coords = np.argwhere(mask)  # (n_voxels, 3)
        centroids[i] = coords.mean(axis=0)

    return centroids


def sample_sparse_labels(gt_labels: np.ndarray, fraction: float = 0.001, seed: int = 42) -> np.ndarray:
    """Sample sparse labels uniformly at random."""
    rng = np.random.RandomState(seed)

    sparse_labels = np.full_like(gt_labels, -1, dtype=np.int32)

    # Sample voxels
    total_voxels = gt_labels.size
    n_samples = int(total_voxels * fraction)

    # Flatten and sample
    flat_gt = gt_labels.ravel()
    indices = rng.choice(total_voxels, size=n_samples, replace=False)

    # Set sampled labels
    sparse_labels.flat[indices] = flat_gt[indices]

    return sparse_labels


def sparse_voxel_to_sv_labels(sparse_voxel_labels: np.ndarray, sv_ids: np.ndarray) -> np.ndarray:
    """Convert sparse voxel labels to SV-level labels using majority vote."""
    unique_svs = np.unique(sv_ids)
    N = len(unique_svs)

    sv_labels = np.full(N, -1, dtype=np.int64)

    for i, sv_id in enumerate(unique_svs):
        sv_mask = (sv_ids == sv_id)
        voxel_labels = sparse_voxel_labels[sv_mask]

        # Filter unlabeled (-1)
        labeled_voxels = voxel_labels[voxel_labels >= 0]

        if len(labeled_voxels) == 0:
            continue

        # Majority vote
        unique, counts = np.unique(labeled_voxels, return_counts=True)
        sv_labels[i] = unique[np.argmax(counts)]

    return sv_labels


def sv_labels_to_dense(sv_ids: np.ndarray, sv_labels: np.ndarray, unique_svs: np.ndarray) -> np.ndarray:
    """Broadcast SV labels to dense voxel labels."""
    dense_labels = np.zeros_like(sv_ids, dtype=np.int16)

    for i, sv_id in enumerate(unique_svs):
        dense_labels[sv_ids == sv_id] = sv_labels[i]

    return dense_labels


def run_sv_method(
    image_vol: np.ndarray,
    gt_labels: np.ndarray,
    method: str,
    n_segments: int,
    sparse_fraction: float,
    k: int,
    alpha: float,
    num_classes: int,
) -> dict:
    """Run one supervoxel method and evaluate."""

    print(f"\n{'='*60}")
    print(f"Method: {method.upper()}")
    print(f"{'='*60}")

    results = {"method": method, "n_segments": n_segments}

    # 1. Generate supervoxels
    print(f"Generating supervoxels (n_segments={n_segments})...")
    t0 = perf_counter()
    sv_ids = run_supervoxels(image_vol, n_segments=n_segments, method=method)
    sv_time = perf_counter() - t0

    unique_svs = np.unique(sv_ids)
    n_actual_svs = len(unique_svs)

    print(f"  Generated {n_actual_svs} supervoxels in {sv_time:.2f}s")
    results["n_actual_svs"] = int(n_actual_svs)
    results["sv_generation_time"] = sv_time

    # 2. Scenario 1: Full GT labels (majority voting)
    print(f"\nScenario 1: Full GT labels (majority voting)")
    print("-" * 60)

    t0 = perf_counter()
    dense_voted, n_filled = majority_fill(
        gt_labels.astype(np.int32),
        sv_ids,
        unlabeled_values=(-1, 6),
        tie_policy="skip",
        output_unlabeled_value=6
    )
    vote_time = perf_counter() - t0

    dice_voted = compute_all_dice(dense_voted, gt_labels, num_classes)

    print(f"  Filled {n_filled}/{n_actual_svs} SVs in {vote_time:.2f}s")
    print(f"  Mean Dice: {dice_voted['mean_dice']:.4f}")
    for cls in range(num_classes):
        print(f"    Class {cls}: {dice_voted[f'dice_class_{cls}']:.4f}")

    results["full_gt"] = {
        "n_filled_svs": int(n_filled),
        "voting_time": vote_time,
        **dice_voted
    }

    # 3. Scenario 2: Sparse labels (Graph LP)
    print(f"\nScenario 2: {sparse_fraction*100:.1f}% sparse labels (Graph LP)")
    print("-" * 60)

    # Sample sparse voxel labels
    sparse_voxel_labels = sample_sparse_labels(gt_labels, fraction=sparse_fraction)
    n_sparse_voxels = np.sum(sparse_voxel_labels >= 0)
    sparse_voxel_fraction = n_sparse_voxels / gt_labels.size

    print(f"  Sampled {n_sparse_voxels} voxels ({sparse_voxel_fraction*100:.2f}%)")

    # Convert to SV-level sparse labels
    sv_labels_sparse = sparse_voxel_to_sv_labels(sparse_voxel_labels, sv_ids)
    n_labeled_svs = np.sum(sv_labels_sparse >= 0)
    labeled_sv_fraction = n_labeled_svs / n_actual_svs

    print(f"  {n_labeled_svs}/{n_actual_svs} SVs have labels ({labeled_sv_fraction*100:.1f}%)")

    # Compute SV features (centroids)
    sv_features = compute_sv_centroids(sv_ids, unique_svs)

    # Run Graph LP
    print(f"  Running Graph LP (k={k}, alpha={alpha})...")
    t0 = perf_counter()
    pred_sv_labels = graph_label_propagation(
        sv_features,
        sv_labels_sparse,
        num_classes,
        k=k,
        alpha=alpha,
        sigma=None,
    )
    lp_time = perf_counter() - t0

    # Convert to dense voxel labels
    dense_lp = sv_labels_to_dense(sv_ids, pred_sv_labels, unique_svs)

    dice_lp = compute_all_dice(dense_lp, gt_labels, num_classes)

    print(f"  Graph LP completed in {lp_time:.2f}s")
    print(f"  Mean Dice: {dice_lp['mean_dice']:.4f}")
    for cls in range(num_classes):
        print(f"    Class {cls}: {dice_lp[f'dice_class_{cls}']:.4f}")

    results["sparse_lp"] = {
        "n_sparse_voxels": int(n_sparse_voxels),
        "sparse_voxel_fraction": sparse_voxel_fraction,
        "n_labeled_svs": int(n_labeled_svs),
        "labeled_sv_fraction": labeled_sv_fraction,
        "k": k,
        "alpha": alpha,
        "lp_time": lp_time,
        **dice_lp
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="A/B test SLIC vs boundary supervoxels")
    parser.add_argument("--case_id", type=str, default="SN13B0_I17_3D_B1_1B250409")
    parser.add_argument("--image_path", type=str,
                        default="/data3/wp5/wp5-code/dataloaders/wp5-dataset/data/{case_id}_image.nii")
    parser.add_argument("--label_path", type=str,
                        default="/data3/wp5/wp5-code/dataloaders/wp5-dataset/data/{case_id}_label.nii")
    parser.add_argument("--n_segments", type=int, default=12000)
    parser.add_argument("--sparse_fraction", type=float, default=0.001, help="Fraction of voxels to label (0.001 = 0.1%)")
    parser.add_argument("--k", type=int, default=15, help="Number of neighbors for Graph LP")
    parser.add_argument("--alpha", type=float, default=0.9, help="Propagation parameter for Graph LP")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="tmp/ab_test_slic_vs_boundary")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup paths
    image_path = Path(args.image_path.format(case_id=args.case_id))
    label_path = Path(args.label_path.format(case_id=args.case_id))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"A/B Test: SLIC vs Boundary-Preserving Supervoxels")
    print(f"{'='*60}")
    print(f"Case ID: {args.case_id}")
    print(f"Image: {image_path}")
    print(f"Labels: {label_path}")
    print(f"n_segments: {args.n_segments}")
    print(f"Sparse fraction: {args.sparse_fraction*100:.1f}%")
    print(f"Graph LP: k={args.k}, alpha={args.alpha}")

    # Load data
    print(f"\nLoading data...")
    loader = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ])

    data = loader({"image": str(image_path), "label": str(label_path)})
    image_vol = data["image"][0].numpy().astype(np.float32)
    gt_labels = data["label"][0].numpy().astype(np.int16)

    print(f"  Image shape: {image_vol.shape}")
    print(f"  GT labels shape: {gt_labels.shape}")
    print(f"  GT classes: {np.unique(gt_labels)}")

    # Run both methods
    results_slic = run_sv_method(
        image_vol, gt_labels, "slic", args.n_segments,
        args.sparse_fraction, args.k, args.alpha, args.num_classes
    )

    results_boundary = run_sv_method(
        image_vol, gt_labels, "boundary", args.n_segments,
        args.sparse_fraction, args.k, args.alpha, args.num_classes
    )

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"SUMMARY COMPARISON")
    print(f"{'='*60}")

    print(f"\nSupervoxel Generation:")
    print(f"  SLIC: {results_slic['n_actual_svs']} SVs in {results_slic['sv_generation_time']:.2f}s")
    print(f"  Boundary: {results_boundary['n_actual_svs']} SVs in {results_boundary['sv_generation_time']:.2f}s")

    print(f"\nScenario 1 - Full GT (Majority Voting):")
    print(f"  SLIC Mean Dice: {results_slic['full_gt']['mean_dice']:.4f}")
    print(f"  Boundary Mean Dice: {results_boundary['full_gt']['mean_dice']:.4f}")
    print(f"  Difference: {results_boundary['full_gt']['mean_dice'] - results_slic['full_gt']['mean_dice']:+.4f}")

    print(f"\nScenario 2 - {args.sparse_fraction*100:.1f}% Sparse (Graph LP):")
    print(f"  SLIC Mean Dice: {results_slic['sparse_lp']['mean_dice']:.4f}")
    print(f"  Boundary Mean Dice: {results_boundary['sparse_lp']['mean_dice']:.4f}")
    print(f"  Difference: {results_boundary['sparse_lp']['mean_dice'] - results_slic['sparse_lp']['mean_dice']:+.4f}")

    # Per-class comparison
    print(f"\nPer-Class Dice (Sparse LP):")
    print(f"  {'Class':<10} {'SLIC':<10} {'Boundary':<10} {'Diff':<10}")
    print(f"  {'-'*40}")
    for cls in range(args.num_classes):
        slic_dice = results_slic['sparse_lp'][f'dice_class_{cls}']
        boundary_dice = results_boundary['sparse_lp'][f'dice_class_{cls}']
        diff = boundary_dice - slic_dice
        print(f"  {cls:<10} {slic_dice:<10.4f} {boundary_dice:<10.4f} {diff:+10.4f}")

    # Save results
    results_all = {
        "case_id": args.case_id,
        "config": {
            "n_segments": args.n_segments,
            "sparse_fraction": args.sparse_fraction,
            "k": args.k,
            "alpha": args.alpha,
            "num_classes": args.num_classes,
            "seed": args.seed,
        },
        "slic": results_slic,
        "boundary": results_boundary,
        "comparison": {
            "full_gt_dice_diff": results_boundary['full_gt']['mean_dice'] - results_slic['full_gt']['mean_dice'],
            "sparse_lp_dice_diff": results_boundary['sparse_lp']['mean_dice'] - results_slic['sparse_lp']['mean_dice'],
        }
    }

    output_file = output_dir / f"{args.case_id}_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results_all, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
