#!/usr/bin/env python3
"""
Batch graph label propagation for multiple cases.

Uses Zhou-style graph LP to propagate sparse SV labels to all SVs
for all training cases.

Usage:
    python3 scripts/propagate_graph_lp_multi_case.py \
        --sv_dir /data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n12000_c0.05_s1.0_ras2_voted \
        --seeds_dir runs/strategic_sparse_0p1pct_k_multi/strategic_seeds \
        --k 10 \
        --alpha 0.9 \
        --output_dir runs/graph_lp_prop_0p1pct_k10_a0.9 \
        --seed 42

Outputs (per case):
    - cases/<case_id>/propagated_labels.npy - Dense voxel labels
    - cases/<case_id>/propagation_meta.json - Statistics
    - labels/<case_id>_labels.npy - Symlinks for training
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wp5.weaklabel.graph_label_propagation import graph_label_propagation
from scripts.propagate_sv_labels_multi_k import compute_sv_centroids


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


def propagate_case(
    case_id: str,
    sv_dir: Path,
    seeds_dir: Path,
    output_dir: Path,
    k: int,
    alpha: float,
    num_classes: int = 5,
) -> Dict:
    """Process one case: load sparse labels, propagate with Graph LP, save."""

    # Load supervoxel IDs
    sv_ids_path = sv_dir / f"{case_id}_sv_ids.npy"
    if not sv_ids_path.exists():
        print(f"WARNING: SV IDs not found for {case_id}, skipping")
        return None

    sv_ids = np.load(sv_ids_path)
    unique_svs = np.unique(sv_ids)
    N = len(unique_svs)

    # Load sparse SV labels
    sparse_labels_path = seeds_dir / f"{case_id}_sv_labels_sparse.json"
    if not sparse_labels_path.exists():
        print(f"WARNING: Sparse labels not found for {case_id}, skipping")
        return None

    with open(sparse_labels_path) as f:
        data = json.load(f)
        sv_labels_sparse = {int(k): int(v) for k, v in data["sv_labels"].items()}

    n_labeled = len(sv_labels_sparse)

    # Compute SV features (centroids)
    sv_features = compute_sv_centroids(sv_ids, unique_svs)

    # Create sv_labels array with -1 for unlabeled
    sv_labels_array = np.full(N, -1, dtype=np.int64)
    for i, sv_id in enumerate(unique_svs):
        if sv_id in sv_labels_sparse:
            sv_labels_array[i] = sv_labels_sparse[sv_id]

    # Run Graph LP
    pred_sv_labels = graph_label_propagation(
        sv_features,
        sv_labels_array,
        num_classes,
        k=k,
        alpha=alpha,
        sigma=None,  # Use median heuristic
    )

    # Convert to dense voxel labels
    pred_sv_labels_dict = {int(unique_svs[i]): int(pred_sv_labels[i])
                            for i in range(N)}
    dense_labels = sv_labels_to_dense(sv_ids, pred_sv_labels_dict)

    # Save outputs
    case_output_dir = output_dir / "cases" / case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)

    # Save dense labels
    np.save(case_output_dir / "propagated_labels.npy", dense_labels)

    # Save sparse labels (copy for reference)
    with open(case_output_dir / "sparse_sv_labels.json", 'w') as f:
        json.dump({"sv_labels": {str(k): int(v) for k, v in sv_labels_sparse.items()}}, f, indent=2)

    # Save metadata
    meta = {
        "case_id": case_id,
        "n_labeled_svs_input": n_labeled,
        "n_total_svs": int(N),
        "k": k,
        "alpha": alpha,
        "sv_ids_shape": list(sv_ids.shape),
    }

    with open(case_output_dir / "propagation_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


def create_training_symlinks(output_dir: Path, cases: list):
    """
    Create flat labels directory with symlinks for training.

    Structure:
        labels/<case_id>_labels.npy -> ../cases/<case_id>/propagated_labels.npy
    """
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    for case_id in cases:
        # Source: cases/<case_id>/propagated_labels.npy
        src = output_dir / "cases" / case_id / "propagated_labels.npy"

        # Target: labels/<case_id>_labels.npy
        dst = labels_dir / f"{case_id}_labels.npy"

        if src.exists():
            # Create relative symlink
            rel_src = Path("..") / "cases" / case_id / "propagated_labels.npy"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(rel_src)

    print(f"\nCreated training directory: {labels_dir}/")
    n_files = len(list(labels_dir.glob("*.npy")))
    print(f"  {n_files} label files")


def main():
    parser = argparse.ArgumentParser(description="Batch Graph LP propagation")
    parser.add_argument("--sv_dir", type=str, required=True,
                       help="Directory containing supervoxel IDs (*_sv_ids.npy)")
    parser.add_argument("--seeds_dir", type=str, required=True,
                       help="Directory with sparse seed labels from sampling step")
    parser.add_argument("--k", type=int, default=10,
                       help="Number of neighbors for graph (default: 10)")
    parser.add_argument("--alpha", type=float, default=0.9,
                       help="Propagation parameter (default: 0.9)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for propagated labels")
    parser.add_argument("--num_classes", type=int, default=5,
                       help="Number of classes (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    print(f"Graph LP parameters: k={args.k}, alpha={args.alpha}")

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
        meta = propagate_case(case_id, sv_dir, seeds_dir, output_dir, args.k, args.alpha, args.num_classes)
        if meta:
            all_meta.append(meta)

    # Create training symlinks
    if all_meta:
        create_training_symlinks(output_dir, [m["case_id"] for m in all_meta])

        # Summary
        avg_labeled = np.mean([m["n_labeled_svs_input"] for m in all_meta])
        avg_total = np.mean([m["n_total_svs"] for m in all_meta])

        summary = {
            "n_cases": len(all_meta),
            "k": args.k,
            "alpha": args.alpha,
            "avg_labeled_svs_input": float(avg_labeled),
            "avg_total_svs": float(avg_total),
            "avg_coverage_input": float(avg_labeled / avg_total),
            "seed": args.seed,
        }

        with open(output_dir / "propagation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*70)
        print("GRAPH LP PROPAGATION COMPLETE!")
        print("="*70)
        print(f"  Processed: {len(all_meta)} cases")
        print(f"  Parameters: k={args.k}, alpha={args.alpha}")
        print(f"  Avg input coverage: {avg_labeled/avg_total*100:.1f}%")
        print(f"  Output: {output_dir}")
        print(f"  Training dir: {output_dir}/labels/")
        print("="*70)


if __name__ == "__main__":
    main()
