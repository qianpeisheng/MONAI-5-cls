#!/usr/bin/env python3
"""
Oracle-corrected supervoxel labeling.

For any supervoxel containing at least one sparse seed, assign it the
GT majority label (oracle). This shows the maximum possible benefit of
having perfect SV-level labels.

Usage:
    python3 scripts/oracle_correct_sv_labels.py \
        --seed_file tmp/test_seeds/SN13B0_I17_3D_B1_1B250409_0p1pct_seed_labels.npy \
        --sv_ids /data3/wp5/.../SN13B0_I17_3D_B1_1B250409_sv_ids.npy \
        --gt_labels /data3/wp5/.../SN13B0_I17_3D_B1_1B250409_labels.npy \
        --output_json tmp/oracle_sv_labels_0p1pct.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np


def compute_sv_gt_majority(sv_ids: np.ndarray, gt_labels: np.ndarray, sv_list: np.ndarray) -> dict:
    """
    Compute GT majority label for each supervoxel in sv_list.

    Args:
        sv_ids: (X, Y, Z) supervoxel IDs
        gt_labels: (X, Y, Z) ground truth labels
        sv_list: (N,) array of SV IDs to compute majority for

    Returns:
        sv_majority: dict {sv_id: majority_label}
    """
    sv_majority = {}

    for sv_id in sv_list:
        # Get all voxels in this SV
        mask = (sv_ids == sv_id)
        sv_gt_labels = gt_labels[mask]

        # Compute majority (mode)
        if len(sv_gt_labels) == 0:
            continue

        # Use bincount for majority voting
        majority_label = int(np.bincount(sv_gt_labels).argmax())
        sv_majority[int(sv_id)] = majority_label

    return sv_majority


def oracle_correct_sv_labels(
    seed_labels: np.ndarray,
    sv_ids: np.ndarray,
    gt_labels: np.ndarray,
) -> dict:
    """
    Create oracle-corrected SV labels.

    For each SV containing at least one sparse seed, assign it the GT majority
    label (instead of the seed's label).

    Args:
        seed_labels: (X, Y, Z) sparse seed labels (-1 for unlabeled)
        sv_ids: (X, Y, Z) supervoxel IDs
        gt_labels: (X, Y, Z) ground truth labels

    Returns:
        results: dict with:
            - sv_labels: {sv_id: oracle_label}
            - metadata: statistics about corrections
    """
    # Find all labeled voxels
    labeled_mask = (seed_labels >= 0)
    n_labeled_voxels = np.sum(labeled_mask)

    print(f"Total labeled voxels: {n_labeled_voxels}")

    # Find which SVs contain labeled voxels
    labeled_sv_ids = np.unique(sv_ids[labeled_mask])
    n_labeled_svs = len(labeled_sv_ids)

    print(f"Labeled SVs: {n_labeled_svs}")

    # Compute GT majority for each labeled SV
    print("Computing GT majority for each labeled SV...")
    sv_gt_majority = compute_sv_gt_majority(sv_ids, gt_labels, labeled_sv_ids)

    # Also get the seed-based labels for comparison
    sv_seed_labels = {}
    labeled_coords = np.argwhere(labeled_mask)

    for coord in labeled_coords:
        x, y, z = coord
        sv_id = int(sv_ids[x, y, z])
        seed_label = int(seed_labels[x, y, z])

        # Take first seed if SV has multiple (shouldn't happen with strategic sampling)
        if sv_id not in sv_seed_labels:
            sv_seed_labels[sv_id] = seed_label

    # Compare seed labels vs GT majority
    n_corrections = 0
    corrections_detail = []

    for sv_id in labeled_sv_ids:
        seed_label = sv_seed_labels.get(sv_id, -1)
        gt_majority = sv_gt_majority.get(sv_id, -1)

        if seed_label != gt_majority:
            n_corrections += 1
            corrections_detail.append({
                "sv_id": int(sv_id),
                "seed_label": int(seed_label),
                "gt_majority": int(gt_majority),
            })

    pct_corrections = 100.0 * n_corrections / n_labeled_svs if n_labeled_svs > 0 else 0

    print(f"\nCorrections made: {n_corrections} / {n_labeled_svs} ({pct_corrections:.1f}%)")

    # Count corrections by class transition
    if corrections_detail:
        print("\nCorrection breakdown:")
        transition_counts = Counter((c["seed_label"], c["gt_majority"]) for c in corrections_detail)
        for (seed_cls, gt_cls), count in transition_counts.most_common():
            print(f"  Class {seed_cls} → {gt_cls}: {count} times")

    # Prepare results
    results = {
        "sv_labels": {str(k): int(v) for k, v in sv_gt_majority.items()},
        "metadata": {
            "n_labeled_voxels": int(n_labeled_voxels),
            "n_labeled_svs": int(n_labeled_svs),
            "n_corrections": int(n_corrections),
            "pct_corrections": float(pct_corrections),
            "corrections_detail": corrections_detail,
            "seed_labels_comparison": {str(k): int(v) for k, v in sv_seed_labels.items()},
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Oracle-correct SV labels using GT majority")
    parser.add_argument("--seed_file", type=str, required=True,
                       help="Sparse seed labels .npy file")
    parser.add_argument("--sv_ids", type=str, required=True,
                       help="Supervoxel IDs .npy file")
    parser.add_argument("--gt_labels", type=str, required=True,
                       help="Ground truth labels .npy file")
    parser.add_argument("--output_json", type=str, required=True,
                       help="Output JSON file for oracle-corrected SV labels")

    args = parser.parse_args()

    # Load inputs
    print(f"Loading seed labels from: {args.seed_file}")
    seed_labels = np.load(args.seed_file)

    print(f"Loading SV IDs from: {args.sv_ids}")
    sv_ids = np.load(args.sv_ids)

    print(f"Loading GT labels from: {args.gt_labels}")
    gt_labels = np.load(args.gt_labels)

    # Validate shapes
    if seed_labels.shape != sv_ids.shape or seed_labels.shape != gt_labels.shape:
        print(f"ERROR: Shape mismatch!")
        print(f"  seed_labels: {seed_labels.shape}")
        print(f"  sv_ids: {sv_ids.shape}")
        print(f"  gt_labels: {gt_labels.shape}")
        return 1

    print(f"Volume shape: {seed_labels.shape}")
    print()

    # Oracle correction
    results = oracle_correct_sv_labels(seed_labels, sv_ids, gt_labels)

    # Save output
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nOracle-corrected SV labels saved to: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
