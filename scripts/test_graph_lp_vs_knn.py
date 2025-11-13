#!/usr/bin/env python3
"""
Test graph label propagation vs KNN on the same seeds.

Uses existing sparse voxel seeds from previous KNN experiments
and compares graph LP performance at different budgets (0.1%, 0.5%, 1.0%).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from wp5.weaklabel.graph_label_propagation import graph_label_propagation
from scripts.propagate_sv_labels_multi_k import compute_sv_centroids


def compute_dice(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> dict:
    """Compute per-class and average Dice scores."""
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

    # Foreground average (classes 1-4)
    fg_dices = [dice_scores[f"dice_class_{c}"] for c in range(1, num_classes)
                if not np.isnan(dice_scores[f"dice_class_{c}"])]
    dice_scores["dice_fg"] = np.mean(fg_dices) if fg_dices else float('nan')

    return dice_scores


def voxel_seeds_to_sv_labels(
    seed_labels: np.ndarray,
    sv_ids: np.ndarray,
) -> dict:
    """
    Convert sparse voxel labels to sparse SV labels.

    Args:
        seed_labels: (X, Y, Z) array with -1 for unlabeled, 0-4 for labeled
        sv_ids: (X, Y, Z) array of supervoxel IDs

    Returns:
        sv_labels: dict {sv_id: label} for labeled SVs only
    """
    # Find labeled voxels
    labeled_mask = (seed_labels >= 0)
    labeled_coords = np.argwhere(labeled_mask)

    sv_labels = {}

    for coord in labeled_coords:
        x, y, z = coord
        sv_id = int(sv_ids[x, y, z])
        label = int(seed_labels[x, y, z])

        # Take first label if SV already has one (shouldn't happen with strategic sampling)
        if sv_id not in sv_labels:
            sv_labels[sv_id] = label

    return sv_labels


def sv_labels_to_dense(sv_ids: np.ndarray, sv_labels: dict) -> np.ndarray:
    """Broadcast SV-level labels to voxel-level dense array."""
    dense_labels = np.zeros_like(sv_ids, dtype=np.int16)

    for sv_id, label in sv_labels.items():
        dense_labels[sv_ids == sv_id] = label

    return dense_labels


def test_graph_lp(
    sv_ids: np.ndarray,
    seed_labels: np.ndarray,
    gt_labels: np.ndarray,
    num_classes: int,
    k_values: list,
    alpha_values: list,
) -> dict:
    """
    Test graph LP with different hyperparameters.

    Args:
        sv_ids: (X, Y, Z) supervoxel IDs
        seed_labels: (X, Y, Z) sparse voxel seeds (-1 for unlabeled)
        gt_labels: (X, Y, Z) ground truth labels
        num_classes: number of classes
        k_values: list of k values to test
        alpha_values: list of alpha values to test

    Returns:
        results dict with all combinations
    """
    # Get unique SVs
    unique_svs = np.unique(sv_ids)
    N = len(unique_svs)

    print(f"  Total SVs: {N}")

    # Convert voxel seeds to SV labels
    sv_labels_sparse = voxel_seeds_to_sv_labels(seed_labels, sv_ids)
    n_labeled_svs = len(sv_labels_sparse)
    n_labeled_voxels = np.sum(seed_labels >= 0)

    print(f"  Labeled voxels: {n_labeled_voxels} ({100*n_labeled_voxels/sv_ids.size:.3f}%)")
    print(f"  Labeled SVs: {n_labeled_svs} ({100*n_labeled_svs/N:.1f}%)")

    # Compute SV features (centroids)
    print("  Computing SV centroids...")
    sv_features = compute_sv_centroids(sv_ids, unique_svs)

    # Create sv_labels array with -1 for unlabeled
    sv_labels_array = np.full(N, -1, dtype=np.int64)
    for i, sv_id in enumerate(unique_svs):
        if sv_id in sv_labels_sparse:
            sv_labels_array[i] = sv_labels_sparse[sv_id]

    results = {
        "n_svs": int(N),
        "n_labeled_voxels": int(n_labeled_voxels),
        "n_labeled_svs": int(n_labeled_svs),
        "sv_coverage": float(n_labeled_svs / N),
        "graph_lp": [],
    }

    # Test all combinations
    print(f"  Testing {len(k_values)} k × {len(alpha_values)} alpha = {len(k_values)*len(alpha_values)} combinations...")

    for k in tqdm(k_values, desc="    k values"):
        for alpha in alpha_values:
            try:
                # Run graph LP
                pred_sv_labels = graph_label_propagation(
                    sv_features,
                    sv_labels_array,
                    num_classes,
                    k=k,
                    alpha=alpha,
                    sigma=None,
                )

                # Convert to dense voxel labels
                pred_sv_labels_dict = {int(unique_svs[i]): int(pred_sv_labels[i])
                                        for i in range(N)}
                pred_dense = sv_labels_to_dense(sv_ids, pred_sv_labels_dict)

                # Compute Dice
                dice_scores = compute_dice(pred_dense, gt_labels, num_classes)

                result = {
                    "method": "graph_lp",
                    "k": k,
                    "alpha": alpha,
                    **dice_scores,
                }

                results["graph_lp"].append(result)

            except Exception as e:
                print(f"      ERROR k={k}, alpha={alpha}: {e}")
                results["graph_lp"].append({
                    "method": "graph_lp",
                    "k": k,
                    "alpha": alpha,
                    "error": str(e),
                })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare graph LP vs KNN using same seeds"
    )
    parser.add_argument(
        "--sv_dir",
        type=str,
        default="/data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n12000_c0.05_s1.0_ras2_voted",
        help="Supervoxel directory (must match report: 8075 SVs)",
    )
    parser.add_argument(
        "--case_id",
        type=str,
        default="SN13B0_I17_3D_B1_1B250409",
        help="Case ID to test",
    )
    parser.add_argument(
        "--budget",
        type=str,
        choices=["0p1pct", "0p5pct", "1p0pct"],
        required=True,
        help="Budget to test (0.1%%, 0.5%%, or 1.0%% of voxels)",
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default="5,10,20,30,50,100",
        help="Comma-separated k values to test",
    )
    parser.add_argument(
        "--alpha_values",
        type=str,
        default="0.5,0.7,0.9,0.95,0.99",
        help="Comma-separated alpha values to test",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=5,
        help="Number of classes",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    sv_dir = Path(args.sv_dir)
    case_id = args.case_id

    # Load supervoxel IDs
    sv_ids_path = sv_dir / f"{case_id}_sv_ids.npy"
    if not sv_ids_path.exists():
        print(f"ERROR: SV IDs not found: {sv_ids_path}")
        return 1

    sv_ids = np.load(sv_ids_path)
    print(f"Case: {case_id}")
    print(f"SV partition: {args.sv_dir}")
    print(f"Total SVs: {len(np.unique(sv_ids))}")

    # Load ground truth
    gt_path = sv_dir / f"{case_id}_labels.npy"
    if not gt_path.exists():
        print(f"ERROR: Ground truth not found: {gt_path}")
        return 1

    gt_labels = np.load(gt_path).astype(np.int16)

    # Load seed labels
    seed_file = Path(f"tmp/test_seeds/{case_id}_{args.budget}_seed_labels.npy")

    if not seed_file.exists():
        print(f"ERROR: Seed file not found: {seed_file}")
        print(f"\nPlease generate seeds first using:")
        print(f"  python3 scripts/generate_test_seeds.py --case_id {case_id}")
        return 1

    seed_labels = np.load(seed_file)
    print(f"Loaded seeds from: {seed_file}")

    # Parse hyperparameters
    k_values = [int(k) for k in args.k_values.split(',')]
    alpha_values = [float(a) for a in args.alpha_values.split(',')]

    print(f"\nBudget: {args.budget}")
    print(f"k values: {k_values}")
    print(f"alpha values: {alpha_values}")
    print()

    # Test graph LP
    results = test_graph_lp(
        sv_ids,
        seed_labels,
        gt_labels,
        args.num_classes,
        k_values,
        alpha_values,
    )

    # Find best
    valid_results = [r for r in results["graph_lp"] if "error" not in r]
    if valid_results:
        best = max(valid_results, key=lambda r: r["dice_fg"])

        print(f"\n{'='*70}")
        print("BEST GRAPH LP CONFIGURATION:")
        print(f"  k={best['k']}, alpha={best['alpha']}")
        print(f"  Foreground Dice: {best['dice_fg']:.4f}")
        print(f"  Overall Dice: {best['dice_avg']:.4f}")
        print(f"{'='*70}")

        results["best"] = best

    # Save results
    output = {
        "case_id": case_id,
        "budget": args.budget,
        "sv_dir": args.sv_dir,
        "k_values": k_values,
        "alpha_values": alpha_values,
        **results,
    }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")
    else:
        print("\n" + "="*70)
        print("RESULTS:")
        print(json.dumps(results, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
