#!/usr/bin/env python3
"""
Voxel-level kNN label propagation (no supervoxels).

Propagates sparse seed labels to all unlabeled voxels using k-NN with
weighted voting directly on voxel coordinates.

Comparison to SV-level propagation:
- SV-level: ~8K nodes, fast (~1 min/case)
- Voxel-level: ~1M nodes, slower (~10-30 min/case)

Usage:
    python3 scripts/propagate_voxel_labels_multi_k.py \
        --seeds_dir tmp/test_one_case \
        --k_values 1,3,5,7,10,15,20,30,50 \
        --output_dir tmp/test_propagation_voxel_0p1pct \
        --seed 42
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def propagate_voxel_multi_k(
    seed_mask: np.ndarray,
    k_values: List[int],
) -> Dict[int, np.ndarray]:
    """
    Propagate voxel labels for multiple k values using kNN.

    Args:
        seed_mask: (X, Y, Z) int16, -1=unlabeled, 0-4=class labels
        k_values: List of k values to test

    Returns:
        results: Dict {k: dense_labels} for each k
    """
    print("  Extracting labeled and unlabeled voxels...")

    # Find labeled voxels
    labeled_mask = (seed_mask >= 0)
    labeled_coords = np.argwhere(labeled_mask)
    labeled_values = seed_mask[labeled_mask]

    n_labeled = len(labeled_coords)
    print(f"  Labeled voxels: {n_labeled:,}")

    # Find unlabeled voxels
    unlabeled_mask = (seed_mask == -1)
    unlabeled_coords = np.argwhere(unlabeled_mask)

    n_unlabeled = len(unlabeled_coords)
    print(f"  Unlabeled voxels: {n_unlabeled:,}")

    if n_unlabeled == 0:
        # All voxels already labeled
        print("  All voxels already labeled, no propagation needed")
        return {k: seed_mask.copy() for k in k_values}

    # Build KD-tree on labeled voxel coordinates
    print("  Building KD-tree...")
    tree = cKDTree(labeled_coords)

    # Query with max(k) to get all neighbors at once
    k_max = max(k_values)
    k_query = min(k_max, n_labeled)

    print(f"  Querying k_max={k_query} neighbors for {n_unlabeled:,} voxels...")
    print(f"    (This may take several minutes...)")

    distances, indices = tree.query(unlabeled_coords, k=k_query, workers=-1)

    # Handle case when k=1 (returns 1D arrays)
    if k_query == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    # For each k, compute propagated labels
    results = {}

    for k in k_values:
        print(f"  Propagating with k={k}...")

        dense_labels = seed_mask.copy()
        k_actual = min(k, n_labeled)

        # Vectorized voting (faster than loop)
        # For each unlabeled voxel, compute weighted vote
        for i in tqdm(range(n_unlabeled), desc=f"    k={k}", leave=False, disable=n_unlabeled<10000):
            # Take top k neighbors
            neighbor_indices = indices[i, :k_actual]
            neighbor_distances = distances[i, :k_actual]

            # Get labels of k nearest neighbors
            neighbor_labels = labeled_values[neighbor_indices]

            # Weighted vote (inverse distance weighting)
            weights = 1.0 / (neighbor_distances + 1e-6)
            votes = Counter()
            for label, weight in zip(neighbor_labels, weights):
                votes[int(label)] += weight

            # Assign majority label
            coord = tuple(unlabeled_coords[i])
            dense_labels[coord] = votes.most_common(1)[0][0]

        results[k] = dense_labels

    return results


def process_case(
    case_id: str,
    seeds_dir: Path,
    output_dir: Path,
    k_values: List[int],
) -> Dict:
    """Process one case: load seeds, propagate, save all k variants."""

    # Load seed mask
    seed_file = seeds_dir / f"{case_id}_strategic_seeds.npy"
    if not seed_file.exists():
        print(f"WARNING: Seeds not found for {case_id}, skipping")
        return None

    seed_mask = np.load(seed_file)

    # Check format
    unique_vals = np.unique(seed_mask)
    if -1 not in unique_vals:
        print(f"WARNING: Seed file does not contain -1 values (old format?)")
        print(f"  Unique values: {unique_vals}")
        print(f"  Skipping {case_id}")
        return None

    # Propagate for all k values
    print(f"\nProcessing: {case_id}")
    print(f"  Seed mask shape: {seed_mask.shape}")
    results = propagate_voxel_multi_k(seed_mask, k_values)

    # Save outputs
    case_output_dir = output_dir / "cases" / case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)

    # Save dense labels for each k
    for k, dense_labels in results.items():
        k_str = f"{k:02d}"
        np.save(case_output_dir / f"propagated_k{k_str}_labels.npy", dense_labels)

    # Compute and save metadata
    n_labeled = np.sum(seed_mask >= 0)
    n_total = seed_mask.size

    meta = {
        "case_id": case_id,
        "n_labeled_voxels": int(n_labeled),
        "n_total_voxels": int(n_total),
        "k_values": k_values,
        "shape": list(seed_mask.shape),
        "method": "voxel-level kNN (no supervoxels)",
    }

    with open(case_output_dir / "propagation_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


def create_training_symlinks(output_dir: Path, cases: List[str], k_values: List[int]):
    """
    Create flat k_variants directories with symlinks for training.

    Structure:
        k_variants/k01/<case_id>_labels.npy -> ../../cases/<case_id>/propagated_k01_labels.npy
    """
    k_variants_dir = output_dir / "k_variants"

    for k in k_values:
        k_str = f"{k:02d}"
        k_dir = k_variants_dir / f"k{k_str}"
        k_dir.mkdir(parents=True, exist_ok=True)

        for case_id in cases:
            src = output_dir / "cases" / case_id / f"propagated_k{k_str}_labels.npy"
            dst = k_dir / f"{case_id}_labels.npy"

            if src.exists():
                rel_src = Path("../..") / "cases" / case_id / f"propagated_k{k_str}_labels.npy"
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(rel_src)

    print(f"\nCreated training directories in {k_variants_dir}/")
    for k in k_values:
        k_str = f"{k:02d}"
        n_files = len(list((k_variants_dir / f"k{k_str}").glob("*.npy")))
        print(f"  k{k_str}/: {n_files} cases")


def main():
    parser = argparse.ArgumentParser(description="Voxel-level kNN label propagation")
    parser.add_argument("--seeds_dir", type=str, required=True,
                       help="Directory with seed files (*_strategic_seeds.npy)")
    parser.add_argument("--k_values", type=str, default="1,3,5,7,10,15,20,30,50",
                       help="Comma-separated k values (default: 1,3,5,7,10,15,20,30,50)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for propagated labels")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Parse k values
    k_values = [int(k) for k in args.k_values.split(',')]
    print(f"K values: {k_values}")

    seeds_dir = Path(args.seeds_dir)
    output_dir = Path(args.output_dir)

    # Find all seed files
    seed_files = list(seeds_dir.glob("*_strategic_seeds.npy"))
    cases = [f.stem.replace("_strategic_seeds", "") for f in seed_files]

    print(f"Found {len(cases)} cases with seeds")

    if len(cases) == 0:
        print("ERROR: No seed files found")
        sys.exit(1)

    # Process all cases
    all_meta = []
    for case_id in tqdm(cases, desc="Processing cases"):
        meta = process_case(case_id, seeds_dir, output_dir, k_values)
        if meta:
            all_meta.append(meta)

    # Create training symlinks
    if all_meta:
        create_training_symlinks(output_dir, [m["case_id"] for m in all_meta], k_values)

        # Summary
        avg_labeled = np.mean([m["n_labeled_voxels"] for m in all_meta])
        avg_total = np.mean([m["n_total_voxels"] for m in all_meta])

        summary = {
            "n_cases": len(all_meta),
            "k_values": k_values,
            "avg_labeled_voxels": float(avg_labeled),
            "avg_total_voxels": float(avg_total),
            "avg_coverage_input": float(avg_labeled / avg_total),
            "method": "voxel-level kNN (no supervoxels)",
            "seed": args.seed,
        }

        with open(output_dir / "propagation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*70)
        print("VOXEL-LEVEL PROPAGATION COMPLETE!")
        print("="*70)
        print(f"  Processed: {len(all_meta)} cases")
        print(f"  K values: {k_values}")
        print(f"  Avg input coverage: {avg_labeled/avg_total*100:.1f}%")
        print(f"  Output: {output_dir}")
        print(f"  Training dirs: {output_dir}/k_variants/k{{01,03,05,...}}/")
        print("="*70)


if __name__ == "__main__":
    main()
