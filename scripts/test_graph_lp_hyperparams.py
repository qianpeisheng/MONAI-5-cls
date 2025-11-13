#!/usr/bin/env python3
"""
Hyperparameter exploration for graph label propagation.

Tests different k and alpha values on a single real volume to find good defaults.
"""

import argparse
import json
import sys
from pathlib import Path
from itertools import product

import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wp5.weaklabel.graph_label_propagation import graph_label_propagation
from scripts.propagate_sv_labels_multi_k import compute_sv_centroids


def compute_dice(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> dict:
    """
    Compute per-class and average Dice scores.

    Args:
        pred: (X, Y, Z) predicted labels
        gt: (X, Y, Z) ground truth labels
        num_classes: number of classes

    Returns:
        dict with per-class Dice and average
    """
    dice_scores = {}

    for c in range(num_classes):
        pred_mask = (pred == c)
        gt_mask = (gt == c)

        intersection = np.sum(pred_mask & gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask)

        if union == 0:
            dice_scores[f"dice_class_{c}"] = float('nan')
        else:
            dice_scores[f"dice_class_{c}"] = 2.0 * intersection / union

    # Average over non-NaN classes
    valid_dices = [v for v in dice_scores.values() if not np.isnan(v)]
    dice_scores["dice_avg"] = np.mean(valid_dices) if valid_dices else float('nan')

    return dice_scores


def sv_labels_to_dense(sv_ids: np.ndarray, sv_labels: dict) -> np.ndarray:
    """
    Broadcast SV-level labels to voxel-level dense array.

    Args:
        sv_ids: (X, Y, Z) int32, supervoxel IDs
        sv_labels: Dict {sv_id: label}

    Returns:
        dense_labels: (X, Y, Z) int16, dense voxel labels
    """
    dense_labels = np.zeros_like(sv_ids, dtype=np.int16)

    for sv_id, label in sv_labels.items():
        dense_labels[sv_ids == sv_id] = label

    return dense_labels


def test_hyperparams(
    sv_ids: np.ndarray,
    sv_features: np.ndarray,
    sv_labels_sparse: dict,
    gt_labels: np.ndarray,
    num_classes: int,
    k_values: list,
    alpha_values: list,
) -> list:
    """
    Test different (k, alpha) combinations.

    Args:
        sv_ids: (X, Y, Z) supervoxel IDs
        sv_features: (N, D) supervoxel features
        sv_labels_sparse: dict {sv_id: label} sparse labels
        gt_labels: (X, Y, Z) ground truth labels
        num_classes: number of classes
        k_values: list of k values to test
        alpha_values: list of alpha values to test

    Returns:
        list of result dicts
    """
    # Get all unique SV IDs (in order)
    unique_svs = np.unique(sv_ids)
    N = len(unique_svs)

    # Create sv_labels array with -1 for unlabeled
    sv_labels_array = np.full(N, -1, dtype=np.int64)
    for i, sv_id in enumerate(unique_svs):
        if sv_id in sv_labels_sparse:
            sv_labels_array[i] = sv_labels_sparse[sv_id]

    results = []

    # Test all combinations
    combos = list(product(k_values, alpha_values))
    print(f"\nTesting {len(combos)} (k, alpha) combinations...")

    for k, alpha in tqdm(combos, desc="Hyperparameter sweep"):
        try:
            # Run graph label propagation
            pred_sv_labels = graph_label_propagation(
                sv_features,
                sv_labels_array,
                num_classes,
                k=k,
                alpha=alpha,
                sigma=None,  # Use median heuristic
            )

            # Convert SV labels to dense voxel labels
            pred_sv_labels_dict = {int(unique_svs[i]): int(pred_sv_labels[i])
                                    for i in range(N)}
            pred_dense = sv_labels_to_dense(sv_ids, pred_sv_labels_dict)

            # Compute Dice vs ground truth
            dice_scores = compute_dice(pred_dense, gt_labels, num_classes)

            result = {
                "k": k,
                "alpha": alpha,
                **dice_scores,
            }

            results.append(result)

            print(f"  k={k:2d}, alpha={alpha:.2f} → Dice={dice_scores['dice_avg']:.4f}")

        except Exception as e:
            print(f"  k={k:2d}, alpha={alpha:.2f} → ERROR: {e}")
            results.append({
                "k": k,
                "alpha": alpha,
                "error": str(e),
            })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test graph LP hyperparameters on single volume"
    )
    parser.add_argument(
        "--sv_dir",
        type=str,
        default="/data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n12000_c0.05_s1.0_ras2_voted",
        help="Directory with supervoxel IDs",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/data3/wp5/labels",
        help="Directory with ground truth labels",
    )
    parser.add_argument(
        "--case_id",
        type=str,
        default=None,
        help="Case ID to test (if None, use first available)",
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default="5,10,20",
        help="Comma-separated k values to test (default: 5,10,20)",
    )
    parser.add_argument(
        "--alpha_values",
        type=str,
        default="0.9,0.95,0.99",
        help="Comma-separated alpha values to test (default: 0.9,0.95,0.99)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        help="Number of classes (default: 5)",
    )
    parser.add_argument(
        "--sparse_fraction",
        type=float,
        default=0.01,
        help="Fraction of SVs to use as labeled seeds (default: 0.01)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON file for results (default: print to stdout)",
    )

    args = parser.parse_args()

    np.random.seed(args.seed)

    sv_dir = Path(args.sv_dir)
    gt_dir = Path(args.gt_dir)

    # Find case
    if args.case_id:
        case_id = args.case_id
    else:
        sv_files = list(sv_dir.glob("*_sv_ids.npy"))
        if not sv_files:
            print(f"ERROR: No supervoxel files found in {sv_dir}")
            return 1
        case_id = sv_files[0].stem.replace("_sv_ids", "")

    print(f"Testing case: {case_id}")

    # Load supervoxel IDs
    sv_ids_path = sv_dir / f"{case_id}_sv_ids.npy"
    if not sv_ids_path.exists():
        print(f"ERROR: SV IDs not found: {sv_ids_path}")
        return 1

    sv_ids = np.load(sv_ids_path)
    unique_svs = np.unique(sv_ids)
    N = len(unique_svs)

    print(f"Total SVs: {N}")

    # Compute features (centroids for now)
    print("Computing SV features (centroids)...")
    sv_features = compute_sv_centroids(sv_ids, unique_svs)

    # Load ground truth labels
    gt_path = gt_dir / f"{case_id}_labels.npy"
    if not gt_path.exists():
        print(f"ERROR: Ground truth not found: {gt_path}")
        return 1

    gt_labels = np.load(gt_path).astype(np.int16)

    # Create sparse labels from ground truth (simulate strategic sampling)
    print(f"Creating sparse labels ({args.sparse_fraction*100:.1f}% of SVs)...")
    n_labeled = max(10, int(args.sparse_fraction * N))
    labeled_indices = np.random.choice(N, n_labeled, replace=False)

    # Get ground truth labels for labeled SVs
    sv_labels_sparse = {}
    for i in labeled_indices:
        sv_id = int(unique_svs[i])
        # Get majority label in this SV from ground truth
        sv_mask = (sv_ids == sv_id)
        sv_gt_labels = gt_labels[sv_mask]
        # Use mode (most common label)
        label = int(np.bincount(sv_gt_labels).argmax())
        sv_labels_sparse[sv_id] = label

    print(f"Labeled SVs: {len(sv_labels_sparse)}")

    # Parse hyperparameters
    k_values = [int(k) for k in args.k_values.split(',')]
    alpha_values = [float(a) for a in args.alpha_values.split(',')]

    print(f"k values: {k_values}")
    print(f"alpha values: {alpha_values}")

    # Run hyperparameter sweep
    results = test_hyperparams(
        sv_ids,
        sv_features,
        sv_labels_sparse,
        gt_labels,
        args.num_classes,
        k_values,
        alpha_values,
    )

    # Find best configuration
    valid_results = [r for r in results if "error" not in r and not np.isnan(r["dice_avg"])]
    if valid_results:
        best = max(valid_results, key=lambda r: r["dice_avg"])
        print(f"\n{'='*70}")
        print("BEST CONFIGURATION:")
        print(f"  k={best['k']}, alpha={best['alpha']} → Dice={best['dice_avg']:.4f}")
        print(f"{'='*70}")

    # Save or print results
    output = {
        "case_id": case_id,
        "n_svs": N,
        "n_labeled": len(sv_labels_sparse),
        "sparse_fraction": args.sparse_fraction,
        "num_classes": args.num_classes,
        "k_values": k_values,
        "alpha_values": alpha_values,
        "results": results,
        "best": best if valid_results else None,
    }

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")
    else:
        print("\n" + "="*70)
        print("ALL RESULTS:")
        print(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
