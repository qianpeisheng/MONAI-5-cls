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

from concurrent.futures import ProcessPoolExecutor, as_completed

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


def sv_flags_to_dense(sv_ids: np.ndarray, sv_flags: Dict[int, int]) -> np.ndarray:
    """
    Broadcast SV-level binary flags to voxel-level dense array.

    Args:
        sv_ids: (X, Y, Z) int32, supervoxel IDs
        sv_flags: Dict {sv_id: 0 or 1}

    Returns:
        dense_flags: (X, Y, Z) uint8 array with 1 where sv_flags[sv_id]==1, else 0.
    """
    dense = np.zeros_like(sv_ids, dtype=np.uint8)
    for sv_id, flag in sv_flags.items():
        if flag:
            dense[sv_ids == sv_id] = 1
    return dense


def propagate_case(
    case_id: str,
    sv_dir: Path,
    seeds_dir: Path,
    output_dir: Path,
    k: int,
    alpha: float,
    num_classes: int = 5,
    use_outer_bg_split: bool = False,
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

    # Optional outer background SVs (taken from seeds_meta, if available).
    outer_bg_sv_ids = set()
    if use_outer_bg_split:
        seeds_meta_path = seeds_dir / f"{case_id}_seeds_meta.json"
        if not seeds_meta_path.exists():
            raise RuntimeError(
                f"outer_bg_split requested but seeds meta not found for case {case_id}: "
                f"{seeds_meta_path}"
            )
        with open(seeds_meta_path) as f:
            seeds_meta = json.load(f)
        outer_ids = seeds_meta.get("outer_bg_sv_ids", None)
        if outer_ids is None:
            raise RuntimeError(
                f"seeds_meta for case {case_id} does not contain 'outer_bg_sv_ids'; "
                "re-run sampling with outer background enabled."
            )
        outer_bg_sv_ids = {int(s) for s in outer_ids}

    # Determine ROI vs outer SVs.
    if outer_bg_sv_ids:
        roi_mask = np.array([sv_id not in outer_bg_sv_ids for sv_id in unique_svs], dtype=bool)
    else:
        roi_mask = np.ones_like(unique_svs, dtype=bool)

    roi_svs = unique_svs[roi_mask]
    N_roi = len(roi_svs)

    # Compute SV features (centroids) for all SVs, then restrict to ROI.
    sv_features_all = compute_sv_centroids(sv_ids, unique_svs)
    sv_features = sv_features_all[roi_mask] if N_roi > 0 else sv_features_all[:0]

    # Create sv_labels array for ROI with -1 for unlabeled and a parallel flag
    # array indicating which SVs had GT-derived labels (from sparse seeds).
    sv_labels_array = np.full(N_roi, -1, dtype=np.int64)
    for i, sv_id in enumerate(roi_svs):
        if int(sv_id) in sv_labels_sparse:
            sv_labels_array[i] = sv_labels_sparse[int(sv_id)]

    # Track GT-supported SVs over *all* SVs for source masks.
    sv_has_gt_all = np.array([int(sv_id) in sv_labels_sparse for sv_id in unique_svs], dtype=bool)

    # Run Graph LP only on ROI SVs; outer background SVs are later forced to 0.
    if N_roi > 0:
        pred_sv_labels_roi = graph_label_propagation(
            sv_features,
            sv_labels_array,
            num_classes,
            k=k,
            alpha=alpha,
            sigma=None,  # Use median heuristic
        )
    else:
        pred_sv_labels_roi = np.zeros((0,), dtype=np.int64)

    # Combine ROI predictions with fixed outer background labels.
    pred_sv_labels_dict = {}
    for i, sv_id in enumerate(roi_svs):
        pred_sv_labels_dict[int(sv_id)] = int(pred_sv_labels_roi[i])

    # Any SV not in ROI (outer background) is set to background label 0.
    for sv_id in unique_svs:
        sv_id_int = int(sv_id)
        if sv_id_int not in pred_sv_labels_dict:
            pred_sv_labels_dict[sv_id_int] = 0

    # Convert to dense voxel labels
    dense_labels = sv_labels_to_dense(sv_ids, pred_sv_labels_dict)

    # Build dense voxelwise source mask: 1 for SVs that had GT-derived seeds,
    # 0 for SVs labeled purely via Graph LP.
    sv_source_flags = {int(unique_svs[i]): int(sv_has_gt_all[i]) for i in range(N)}
    dense_source_mask = sv_flags_to_dense(sv_ids, sv_source_flags)

    # Save outputs
    case_output_dir = output_dir / "cases" / case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)

    # Save dense labels
    np.save(case_output_dir / "propagated_labels.npy", dense_labels)
    # Save dense source mask (uint8: 1=seed-supported SV, 0=Graph-only SV)
    np.save(case_output_dir / "source_mask.npy", dense_source_mask.astype(np.uint8))

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
        "n_sv_with_gt": int(sv_has_gt_all.sum()),
        "n_sv_graph_only": int(N - sv_has_gt_all.sum()),
        "voxels_with_gt_seed_support": int(dense_source_mask.sum()),
        "voxels_graph_only": int(dense_source_mask.size - int(dense_source_mask.sum())),
    }
    if outer_bg_sv_ids:
        meta.update(
            {
                "use_outer_bg_split": True,
                "n_outer_bg_svs": int(len(outer_bg_sv_ids)),
                "n_roi_svs": int(N_roi),
            }
        )

    with open(case_output_dir / "propagation_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


def create_training_symlinks(output_dir: Path, cases: list):
    """
    Create flat labels directory with symlinks for training.

    Structure:
        labels/<case_id>_labels.npy -> ../cases/<case_id>/propagated_labels.npy
        source_masks/<case_id>_source.npy -> ../cases/<case_id>/source_mask.npy
    """
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source_masks"
    source_dir.mkdir(parents=True, exist_ok=True)

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

        # Source mask symlink (if present)
        src_mask = output_dir / "cases" / case_id / "source_mask.npy"
        dst_mask = source_dir / f"{case_id}_source.npy"
        if src_mask.exists():
            rel_mask = Path("..") / "cases" / case_id / "source_mask.npy"
            if dst_mask.exists() or dst_mask.is_symlink():
                dst_mask.unlink()
            dst_mask.symlink_to(rel_mask)

    print(f"\nCreated training directory: {labels_dir}/ and source masks: {source_dir}/")
    n_files = len(list(labels_dir.glob("*.npy")))
    n_masks = len(list(source_dir.glob("*.npy")))
    print(f"  {n_files} label files, {n_masks} source masks")


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
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers over cases (default: 1 = serial)")
    parser.add_argument(
        "--use_outer_bg_split",
        action="store_true",
        help=(
            "If set, use outer-background partition from *_seeds_meta.json to "
            "run Graph LP only on ROI SVs and force outer SVs to background."
        ),
    )

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

    # Process all cases (optionally in parallel)
    all_meta = []
    num_workers = max(int(args.num_workers), 1)
    num_workers = min(num_workers, len(cases)) if cases else 0

    if num_workers <= 1:
        for case_id in tqdm(cases, desc="Processing cases"):
            meta = propagate_case(
                case_id=case_id,
                sv_dir=sv_dir,
                seeds_dir=seeds_dir,
                output_dir=output_dir,
                k=args.k,
                alpha=args.alpha,
                num_classes=args.num_classes,
                use_outer_bg_split=args.use_outer_bg_split,
            )
            if meta:
                all_meta.append(meta)
    else:
        print(f"Using {num_workers} workers over {len(cases)} cases")
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {
                ex.submit(
                    propagate_case,
                    case_id,
                    sv_dir,
                    seeds_dir,
                    output_dir,
                    args.k,
                    args.alpha,
                    args.num_classes,
                    args.use_outer_bg_split,
                ): case_id
                for case_id in cases
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing cases"):
                cid = futures[fut]
                try:
                    meta = fut.result()
                except Exception as e:
                    print(f"ERROR processing {cid}: {e}")
                    continue
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
