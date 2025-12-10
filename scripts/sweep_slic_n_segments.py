#!/usr/bin/env python3
"""
Sweep n_segments for SLIC + Graph LP

Tests different n_segments values (1k to 20k) to find optimal supervoxel
granularity for Graph Label Propagation.

Usage:
    python scripts/sweep_slic_n_segments.py \
        --case_id SN13B0_I17_3D_B1_1B250409 \
        --sparse_fraction 0.001 \
        --k 15 \
        --alpha 0.9 \
        --output_dir tmp/slic_n_segments_sweep
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


def run_one_experiment(
    image_vol: np.ndarray,
    gt_labels: np.ndarray,
    sparse_voxel_labels: np.ndarray,
    n_segments: int,
    k: int,
    alpha: float,
    num_classes: int,
) -> dict:
    """Run SLIC + Graph LP for one n_segments value."""

    print(f"\n{'='*60}")
    print(f"n_segments = {n_segments}")
    print(f"{'='*60}")

    results = {"n_segments": n_segments}

    # 1. Generate supervoxels
    print(f"Generating SLIC supervoxels...")
    t0 = perf_counter()
    sv_ids = run_supervoxels(image_vol, n_segments=n_segments, method="slic", enforce_connectivity=False)
    sv_time = perf_counter() - t0

    unique_svs = np.unique(sv_ids)
    n_actual_svs = len(unique_svs)

    print(f"  Generated {n_actual_svs} supervoxels in {sv_time:.2f}s")
    results["n_actual_svs"] = int(n_actual_svs)
    results["sv_generation_time"] = sv_time

    # 2. Convert to SV-level sparse labels
    sv_labels_sparse = sparse_voxel_to_sv_labels(sparse_voxel_labels, sv_ids)
    n_labeled_svs = np.sum(sv_labels_sparse >= 0)
    labeled_sv_fraction = n_labeled_svs / n_actual_svs

    print(f"  {n_labeled_svs}/{n_actual_svs} SVs have labels ({labeled_sv_fraction*100:.1f}%)")
    results["n_labeled_svs"] = int(n_labeled_svs)
    results["labeled_sv_fraction"] = labeled_sv_fraction

    # 3. Compute SV features (centroids)
    sv_features = compute_sv_centroids(sv_ids, unique_svs)

    # 4. Run Graph LP
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

    # 5. Convert to dense voxel labels
    dense_lp = sv_labels_to_dense(sv_ids, pred_sv_labels, unique_svs)

    # 6. Compute Dice scores
    dice_scores = compute_all_dice(dense_lp, gt_labels, num_classes)

    print(f"  Graph LP completed in {lp_time:.2f}s")
    print(f"  Mean Dice: {dice_scores['mean_dice']:.4f}")
    for cls in range(num_classes):
        print(f"    Class {cls}: {dice_scores[f'dice_class_{cls}']:.4f}")

    results["lp_time"] = lp_time
    results.update(dice_scores)

    return results


def main():
    parser = argparse.ArgumentParser(description="Sweep n_segments for SLIC + Graph LP")
    parser.add_argument("--case_id", type=str, default="SN13B0_I17_3D_B1_1B250409")
    parser.add_argument("--image_path", type=str,
                        default="/data3/wp5/wp5-code/dataloaders/wp5-dataset/data/{case_id}_image.nii")
    parser.add_argument("--label_path", type=str,
                        default="/data3/wp5/wp5-code/dataloaders/wp5-dataset/data/{case_id}_label.nii")
    parser.add_argument("--n_segments_min", type=int, default=1000, help="Minimum n_segments")
    parser.add_argument("--n_segments_max", type=int, default=20000, help="Maximum n_segments")
    parser.add_argument("--n_segments_step", type=int, default=1000, help="Step size for n_segments")
    parser.add_argument("--sparse_fraction", type=float, default=0.001, help="Fraction of voxels to label (0.001 = 0.1%)")
    parser.add_argument("--k", type=int, default=15, help="Number of neighbors for Graph LP")
    parser.add_argument("--alpha", type=float, default=0.9, help="Propagation parameter for Graph LP")
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="tmp/slic_n_segments_sweep")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup paths
    image_path = Path(args.image_path.format(case_id=args.case_id))
    label_path = Path(args.label_path.format(case_id=args.case_id))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"SLIC n_segments Sweep with Graph LP")
    print(f"{'='*60}")
    print(f"Case ID: {args.case_id}")
    print(f"Image: {image_path}")
    print(f"Labels: {label_path}")
    print(f"n_segments range: {args.n_segments_min} to {args.n_segments_max} (step={args.n_segments_step})")
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

    # Sample sparse labels once (reused for all experiments)
    print(f"\nSampling sparse labels ({args.sparse_fraction*100:.1f}%)...")
    sparse_voxel_labels = sample_sparse_labels(gt_labels, fraction=args.sparse_fraction, seed=args.seed)
    n_sparse_voxels = np.sum(sparse_voxel_labels >= 0)
    print(f"  Sampled {n_sparse_voxels} voxels ({n_sparse_voxels/gt_labels.size*100:.2f}%)")

    # Run sweep
    n_segments_values = range(args.n_segments_min, args.n_segments_max + 1, args.n_segments_step)
    all_results = []

    for n_segments in n_segments_values:
        result = run_one_experiment(
            image_vol,
            gt_labels,
            sparse_voxel_labels,
            n_segments,
            args.k,
            args.alpha,
            args.num_classes,
        )
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'n_segments':<12} {'Actual SVs':<12} {'Labeled SVs':<12} {'Mean Dice':<12}")
    print(f"{'-'*48}")
    for result in all_results:
        print(f"{result['n_segments']:<12} {result['n_actual_svs']:<12} "
              f"{result['n_labeled_svs']:<12} {result['mean_dice']:<12.4f}")

    # Find best n_segments
    best_result = max(all_results, key=lambda x: x['mean_dice'])
    print(f"\nBest configuration:")
    print(f"  n_segments: {best_result['n_segments']}")
    print(f"  Actual SVs: {best_result['n_actual_svs']}")
    print(f"  Mean Dice: {best_result['mean_dice']:.4f}")

    # Save results
    results_all = {
        "case_id": args.case_id,
        "config": {
            "n_segments_min": args.n_segments_min,
            "n_segments_max": args.n_segments_max,
            "n_segments_step": args.n_segments_step,
            "sparse_fraction": args.sparse_fraction,
            "k": args.k,
            "alpha": args.alpha,
            "num_classes": args.num_classes,
            "seed": args.seed,
        },
        "n_sparse_voxels": int(n_sparse_voxels),
        "results": all_results,
        "best": {
            "n_segments": best_result['n_segments'],
            "n_actual_svs": best_result['n_actual_svs'],
            "mean_dice": best_result['mean_dice'],
        }
    }

    output_file = output_dir / f"{args.case_id}_n_segments_sweep.json"
    with open(output_file, 'w') as f:
        json.dump(results_all, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
