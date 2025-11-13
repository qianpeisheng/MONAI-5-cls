#!/usr/bin/env python3
"""
Generate sparse voxel seeds for testing.

This creates the same sparse seeds as the voxel-level KNN tests,
matching the exact budgets (0.1%, 0.5%, 1.0% of voxels).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

def generate_sparse_seeds(
    gt_labels: np.ndarray,
    budget_ratio: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate sparse voxel seeds.

    Args:
        gt_labels: (X, Y, Z) ground truth labels
        budget_ratio: fraction of voxels to label (e.g., 0.001 for 0.1%)
        seed: random seed

    Returns:
        seed_labels: (X, Y, Z) array with -1 for unlabeled, 0-4 for labeled
    """
    np.random.seed(seed)

    # Count total voxels
    total_voxels = gt_labels.size
    n_seeds = int(budget_ratio * total_voxels)

    print(f"  Total voxels: {total_voxels}")
    print(f"  Budget: {budget_ratio*100:.3f}%")
    print(f"  Seeds to sample: {n_seeds}")

    # Initialize seed labels to -1 (unlabeled)
    seed_labels = np.full_like(gt_labels, -1, dtype=np.int16)

    # Get all voxel coordinates
    all_coords = np.argwhere(np.ones_like(gt_labels, dtype=bool))

    # Randomly sample coordinates
    sampled_indices = np.random.choice(len(all_coords), n_seeds, replace=False)
    sampled_coords = all_coords[sampled_indices]

    # Assign labels from ground truth
    for coord in sampled_coords:
        x, y, z = coord
        seed_labels[x, y, z] = gt_labels[x, y, z]

    # Count labeled voxels per class
    labeled_mask = (seed_labels >= 0)
    n_labeled = np.sum(labeled_mask)
    print(f"  Actually labeled: {n_labeled}")

    for c in range(5):
        n_c = np.sum(seed_labels == c)
        print(f"    Class {c}: {n_c} ({100*n_c/n_labeled:.1f}%)")

    return seed_labels


def main():
    parser = argparse.ArgumentParser(description="Generate test seeds")
    parser.add_argument(
        "--sv_dir",
        type=str,
        default="/data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n12000_c0.05_s1.0_ras2_voted",
        help="Supervoxel directory with ground truth",
    )
    parser.add_argument(
        "--case_id",
        type=str,
        default="SN13B0_I17_3D_B1_1B250409",
        help="Case ID",
    )
    parser.add_argument(
        "--budgets",
        type=str,
        default="0.001,0.005,0.01",
        help="Comma-separated budget ratios (e.g., 0.001 for 0.1%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tmp/test_seeds",
        help="Output directory",
    )

    args = parser.parse_args()

    sv_dir = Path(args.sv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    gt_path = sv_dir / f"{args.case_id}_labels.npy"
    if not gt_path.exists():
        print(f"ERROR: Ground truth not found: {gt_path}")
        return 1

    gt_labels = np.load(gt_path).astype(np.int16)

    print(f"Case: {args.case_id}")
    print(f"GT shape: {gt_labels.shape}")
    print()

    # Parse budgets
    budgets = [float(b) for b in args.budgets.split(',')]

    # Generate seeds for each budget
    for budget_ratio in budgets:
        budget_str = f"{int(budget_ratio*100*10)}pct"  # 0.001 -> 0p1pct
        print(f"Generating seeds for budget: {budget_str}")

        seed_labels = generate_sparse_seeds(gt_labels, budget_ratio, args.seed)

        # Save seed labels
        output_file = output_dir / f"{args.case_id}_{budget_str}_seed_labels.npy"
        np.save(output_file, seed_labels)
        print(f"  Saved: {output_file}")
        print()

    print(f"All seeds saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
