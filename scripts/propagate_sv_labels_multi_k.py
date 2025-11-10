#!/usr/bin/env python3
"""
Multi-k supervoxel label propagation.

Propagates sparse SV labels to all SVs using k-NN with weighted voting,
generating separate outputs for multiple k values in one pass.

Usage:
    python3 scripts/propagate_sv_labels_multi_k.py \
        --sv_dir /data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n12000_c0.05_s1.0_ras2_voted \
        --seeds_dir runs/strategic_seeds_0p1pct \
        --k_values 1,3,5,7,10,15,20,25,30,50 \
        --output_dir runs/sv_sparse_prop_0p1pct_strategic \
        --seed 42

Outputs (per case):
    - cases/<case_id>/sparse_sv_labels.json - Input sparse labels
    - cases/<case_id>/propagated_k01_labels.npy - Dense labels for k=1
    - cases/<case_id>/propagated_k03_labels.npy - Dense labels for k=3
    - ... (one per k value)
    - cases/<case_id>/propagation_meta.json - Statistics

    - k_variants/k01/<case_id>_labels.npy - Symlinks for training
    - k_variants/k03/<case_id>_labels.npy
    - ...
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


def compute_sv_centroids(sv_ids: np.ndarray, sv_list: np.ndarray) -> np.ndarray:
    """
    Compute centroids for given supervoxels.

    Args:
        sv_ids: (X, Y, Z) int32, supervoxel IDs
        sv_list: (N,) array of SV IDs to compute centroids for

    Returns:
        centroids: (N, 3) array of (x, y, z) centroids, ordered by sv_list
    """
    centroids = np.zeros((len(sv_list), 3), dtype=float)

    for i, sv_id in enumerate(sv_list):
        mask = (sv_ids == sv_id)
        coords = np.argwhere(mask)
        if len(coords) > 0:
            centroids[i] = coords.mean(axis=0)

    return centroids


def propagate_multi_k(
    sv_ids: np.ndarray,
    sv_labels_sparse: Dict[int, int],
    k_values: List[int],
) -> Dict[int, Dict[int, int]]:
    """
    Propagate SV labels for multiple k values using k-NN.

    Optimization: Build centroids and KD-tree once, query with different k.

    Args:
        sv_ids: (X, Y, Z) int32, supervoxel IDs
        sv_labels_sparse: Dict {sv_id: label}, sparse labels
        k_values: List of k values to test

    Returns:
        results: Dict {k: {sv_id: label}} for each k
    """
    # Get all unique SV IDs
    sv_unique = np.unique(sv_ids)

    # Compute centroids for all SVs
    print("  Computing centroids...")
    centroids = compute_sv_centroids(sv_ids, sv_unique)

    # Split labeled vs unlabeled
    labeled_svs = np.array(list(sv_labels_sparse.keys()))
    labeled_mask = np.isin(sv_unique, labeled_svs)
    unlabeled_svs = sv_unique[~labeled_mask]

    print(f"  Labeled SVs: {len(labeled_svs)}, Unlabeled SVs: {len(unlabeled_svs)}")

    # Build KD-tree for labeled SV centroids
    print("  Building KD-tree...")
    tree = cKDTree(centroids[labeled_mask])

    # Query with max(k) to get all neighbors at once
    k_max = max(k_values)
    k_query = min(k_max, len(labeled_svs))  # Can't query more than available

    print(f"  Querying k_max={k_query} neighbors...")
    distances, indices = tree.query(centroids[~labeled_mask], k=k_query)

    # Handle case when k=1 (returns 1D arrays)
    if k_query == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    # For each k, compute propagated labels
    results = {}

    for k in k_values:
        print(f"  Propagating with k={k}...")

        sv_labels_full = dict(sv_labels_sparse)  # Start with sparse labels

        k_actual = min(k, len(labeled_svs))

        for i, unlabeled_sv in enumerate(unlabeled_svs):
            # Take top k neighbors
            neighbor_indices = indices[i, :k_actual]
            neighbor_distances = distances[i, :k_actual]

            # Get labels of k nearest neighbors
            neighbor_labels = [sv_labels_sparse[int(labeled_svs[idx])] for idx in neighbor_indices]

            # Weighted vote (inverse distance weighting)
            weights = 1.0 / (neighbor_distances + 1e-6)
            votes = Counter()
            for label, weight in zip(neighbor_labels, weights):
                votes[label] += weight

            # Assign majority label
            sv_labels_full[int(unlabeled_sv)] = votes.most_common(1)[0][0]

        results[k] = sv_labels_full

    return results


def sv_labels_to_dense(sv_ids: np.ndarray, sv_labels: Dict[int, int]) -> np.ndarray:
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


def process_case(
    case_id: str,
    sv_dir: Path,
    seeds_dir: Path,
    output_dir: Path,
    k_values: List[int],
) -> Dict:
    """Process one case: load sparse labels, propagate, save all k variants."""

    # Load supervoxel IDs
    sv_ids_path = sv_dir / f"{case_id}_sv_ids.npy"
    if not sv_ids_path.exists():
        print(f"WARNING: SV IDs not found for {case_id}, skipping")
        return None

    sv_ids = np.load(sv_ids_path)

    # Load sparse SV labels
    sparse_labels_path = seeds_dir / f"{case_id}_sv_labels_sparse.json"
    if not sparse_labels_path.exists():
        print(f"WARNING: Sparse labels not found for {case_id}, skipping")
        return None

    with open(sparse_labels_path) as f:
        data = json.load(f)
        sv_labels_sparse = {int(k): int(v) for k, v in data["sv_labels"].items()}

    # Propagate for all k values
    results = propagate_multi_k(sv_ids, sv_labels_sparse, k_values)

    # Save outputs
    case_output_dir = output_dir / "cases" / case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)

    # Save sparse labels (copy for reference)
    with open(case_output_dir / "sparse_sv_labels.json", 'w') as f:
        json.dump({"sv_labels": {str(k): int(v) for k, v in sv_labels_sparse.items()}}, f, indent=2)

    # Save dense labels for each k
    for k, sv_labels_full in results.items():
        dense_labels = sv_labels_to_dense(sv_ids, sv_labels_full)

        # Save dense .npy file
        k_str = f"{k:02d}"
        np.save(case_output_dir / f"propagated_k{k_str}_labels.npy", dense_labels)

    # Compute and save metadata
    meta = {
        "case_id": case_id,
        "n_labeled_svs_input": len(sv_labels_sparse),
        "n_total_svs": int(len(np.unique(sv_ids))),
        "k_values": k_values,
        "sv_ids_shape": list(sv_ids.shape),
    }

    with open(case_output_dir / "propagation_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


def create_training_symlinks(output_dir: Path, cases: List[str], k_values: List[int]):
    """
    Create flat k_variants directories with symlinks for training.

    Structure:
        k_variants/k01/<case_id>_labels.npy -> ../../cases/<case_id>/propagated_k01_labels.npy
        k_variants/k03/<case_id>_labels.npy -> ../../cases/<case_id>/propagated_k03_labels.npy
        ...
    """
    k_variants_dir = output_dir / "k_variants"

    for k in k_values:
        k_str = f"{k:02d}"
        k_dir = k_variants_dir / f"k{k_str}"
        k_dir.mkdir(parents=True, exist_ok=True)

        for case_id in cases:
            # Source: cases/<case_id>/propagated_k??_labels.npy
            src = output_dir / "cases" / case_id / f"propagated_k{k_str}_labels.npy"

            # Target: k_variants/k??/<case_id>_labels.npy
            dst = k_dir / f"{case_id}_labels.npy"

            if src.exists():
                # Create relative symlink
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
    parser = argparse.ArgumentParser(description="Multi-k SV label propagation")
    parser.add_argument("--sv_dir", type=str, required=True,
                       help="Directory containing supervoxel IDs (*_sv_ids.npy)")
    parser.add_argument("--seeds_dir", type=str, required=True,
                       help="Directory with sparse seed labels from sampling step")
    parser.add_argument("--k_values", type=str, default="1,3,5,7,10,15,20,25,30,50",
                       help="Comma-separated k values (default: 1,3,5,7,10,15,20,25,30,50)")
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

    sv_dir = Path(args.sv_dir)
    seeds_dir = Path(args.seeds_dir)
    output_dir = Path(args.output_dir)

    # Find all cases from seeds directory
    sparse_label_files = list(seeds_dir.glob("*_sv_labels_sparse.json"))
    cases = [f.stem.replace("_sv_labels_sparse", "") for f in sparse_label_files]

    print(f"Found {len(cases)} cases with sparse labels")

    # Process all cases
    all_meta = []
    for case_id in tqdm(cases, desc="Processing cases"):
        print(f"\nCase: {case_id}")
        meta = process_case(case_id, sv_dir, seeds_dir, output_dir, k_values)
        if meta:
            all_meta.append(meta)

    # Create training symlinks
    if all_meta:
        create_training_symlinks(output_dir, [m["case_id"] for m in all_meta], k_values)

        # Summary
        avg_labeled = np.mean([m["n_labeled_svs_input"] for m in all_meta])
        avg_total = np.mean([m["n_total_svs"] for m in all_meta])

        summary = {
            "n_cases": len(all_meta),
            "k_values": k_values,
            "avg_labeled_svs_input": float(avg_labeled),
            "avg_total_svs": float(avg_total),
            "avg_coverage_input": float(avg_labeled / avg_total),
            "seed": args.seed,
        }

        with open(output_dir / "propagation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*70)
        print("PROPAGATION COMPLETE!")
        print("="*70)
        print(f"  Processed: {len(all_meta)} cases")
        print(f"  K values: {k_values}")
        print(f"  Avg input coverage: {avg_labeled/avg_total*100:.1f}%")
        print(f"  Output: {output_dir}")
        print(f"  Training dirs: {output_dir}/k_variants/k{{01,03,05,...}}/")
        print("="*70)


if __name__ == "__main__":
    main()
