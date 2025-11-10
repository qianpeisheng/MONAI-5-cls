#!/usr/bin/env python3
"""
Strategic sparse supervoxel seed sampling.

Samples up to 0.1% of voxels strategically to maximize supervoxel coverage:
- Max 1 labeled voxel per supervoxel (no voting needed)
- Prioritizes foreground (classes 1-4)
- Prioritizes borders (high gradient magnitude)
- Prioritizes rare classes (3, 4 get 2x weight)

Usage:
    python3 scripts/sample_strategic_sv_seeds.py \
        --sv_dir /data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n12000_c0.05_s1.0_ras2_voted \
        --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
        --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json \
        --budget_ratio 0.001 \
        --output_dir runs/strategic_seeds_0p1pct \
        --seed 42

Outputs (per case):
    - <case_id>_strategic_seeds.npy - Binary seed mask (X,Y,Z) bool
    - <case_id>_sv_labels_sparse.json - Dict {sv_id: label}
    - <case_id>_seeds_meta.json - Statistics
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import sobel
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd
from tqdm import tqdm


def load_split_cases(split_cfg: str, data_root: Path, split: str = "train") -> list:
    """Load case IDs from split config.

    Split config format:
        {"test_serial_numbers": [9, 12, 15, ...]}

    Train set = all cases except those with serial numbers in test set.
    """
    import re
    import os

    with open(split_cfg) as f:
        cfg = json.load(f)

    test_serials = set(cfg.get("test_serial_numbers", []))

    def serial_from_name(name: str):
        """Extract serial number from case name like 'SN13B0_...'"""
        m = re.match(r"^SN(\d+)", name)
        return int(m.group(1)) if m else None

    # Find all cases from image files (check both data_root and data_root/data)
    cases = []
    search_dirs = [data_root, data_root / "data"]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for name in os.listdir(search_dir):
            if name.endswith("_image.nii"):
                case_id = name[:-10]  # Remove '_image.nii'
                serial = serial_from_name(case_id)

                if serial is None:
                    continue

                # Split based on test_serial_numbers
                if split == "train" and serial not in test_serials:
                    cases.append(case_id)
                elif split == "test" and serial in test_serials:
                    cases.append(case_id)

    return sorted(list(set(cases)))  # Remove duplicates


def compute_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude using Sobel operator.

    Args:
        image: (X, Y, Z) float array

    Returns:
        grad_mag: (X, Y, Z) float array of gradient magnitudes
    """
    grad_x = sobel(image, axis=0)
    grad_y = sobel(image, axis=1)
    grad_z = sobel(image, axis=2)

    return np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)


def sample_strategic_seeds(
    sv_ids: np.ndarray,
    gt_labels: np.ndarray,
    image: np.ndarray,
    budget_ratio: float = 0.001,
    class_weights: Dict[int, float] = None,
    gradient_weight: float = 1.0,
    centroid_weight: float = 0.5,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Strategic seed sampling to maximize supervoxel coverage.

    Algorithm:
        1. Identify all foreground SVs (containing classes 1-4)
        2. For each FG SV, score candidate voxels by:
           - Class priority: rare classes (3, 4) get higher weight
           - Border proximity: high gradient magnitude
           - Representativeness: near SV centroid (tie-breaker)
        3. Select top 1 voxel per SV (highest score)
        4. Rank all SV candidates globally and select top N within budget

    Args:
        sv_ids: (X, Y, Z) int32, supervoxel IDs
        gt_labels: (X, Y, Z) int16, ground truth labels
        image: (X, Y, Z) float, intensity image
        budget_ratio: fraction of total voxels to sample (default 0.001 = 0.1%)
        class_weights: Dict of class→weight (default: {1:1, 2:1, 3:2, 4:2})
        gradient_weight: weight for gradient component (default 1.0)
        centroid_weight: weight for centroid proximity (default 0.5)

    Returns:
        seed_mask: (X, Y, Z) bool, binary mask of sampled voxels
        seed_sv_labels: Dict {sv_id: label}, sparse SV labels (1:1 mapping)
    """
    if class_weights is None:
        class_weights = {1: 1.0, 2: 1.0, 3: 2.0, 4: 2.0}

    # Calculate budget
    total_voxels = sv_ids.size
    max_seeds = int(total_voxels * budget_ratio)

    # Identify foreground supervoxels
    sv_unique = np.unique(sv_ids)
    fg_svs = []

    for sv_id in sv_unique:
        mask = (sv_ids == sv_id)
        labels_in_sv = gt_labels[mask]
        # Check if SV contains any FG class (1-4)
        if np.any((labels_in_sv >= 1) & (labels_in_sv <= 4)):
            fg_svs.append(sv_id)

    print(f"  Found {len(fg_svs)} foreground SVs out of {len(sv_unique)} total")

    # Precompute gradient magnitude
    grad_mag = compute_gradient_magnitude(image)

    # Score candidates within each FG SV
    sv_candidates = []  # List of (sv_id, voxel_coord, score, label)

    for sv_id in tqdm(fg_svs, desc="  Scoring candidates", leave=False):
        mask = (sv_ids == sv_id)
        coords = np.argwhere(mask)

        # Get labels and filter to FG only
        labels = gt_labels[coords[:, 0], coords[:, 1], coords[:, 2]]
        fg_mask = (labels >= 1) & (labels <= 4)
        if fg_mask.sum() == 0:
            continue

        coords_fg = coords[fg_mask]
        labels_fg = labels[fg_mask]

        # Compute scores for each candidate voxel
        scores = np.zeros(len(coords_fg), dtype=float)

        # Component 1: Class priority (rare classes higher)
        for i, label in enumerate(labels_fg):
            scores[i] += class_weights.get(int(label), 0)

        # Component 2: Border proximity (gradient magnitude)
        gradients = grad_mag[coords_fg[:, 0], coords_fg[:, 1], coords_fg[:, 2]]
        if gradients.max() > 0:
            scores += gradient_weight * (gradients / gradients.max())

        # Component 3: Representativeness (distance to centroid, inverted)
        centroid = coords.mean(axis=0)
        distances = np.linalg.norm(coords_fg - centroid, axis=1)
        if distances.max() > 0:
            scores -= centroid_weight * (distances / distances.max())

        # Select best candidate for this SV
        best_idx = scores.argmax()
        best_coord = coords_fg[best_idx]
        best_label = labels_fg[best_idx]
        best_score = scores[best_idx]

        sv_candidates.append((int(sv_id), tuple(best_coord), best_score, int(best_label)))

    # Rank all SV candidates globally and select top N within budget
    sv_candidates.sort(key=lambda x: x[2], reverse=True)  # Sort by score descending
    selected_candidates = sv_candidates[:max_seeds]

    print(f"  Selected {len(selected_candidates)} seeds (budget: {max_seeds})")

    # Create seed mask and SV label dict
    seed_mask = np.zeros_like(sv_ids, dtype=bool)
    seed_sv_labels = {}

    for sv_id, coord, score, label in selected_candidates:
        seed_mask[coord] = True
        seed_sv_labels[sv_id] = label

    return seed_mask, seed_sv_labels


def process_case(
    case_id: str,
    sv_dir: Path,
    data_root: Path,
    output_dir: Path,
    budget_ratio: float,
    class_weights: Dict[int, float],
) -> Dict:
    """Process one case: load data, sample seeds, save outputs."""

    # Load supervoxel IDs
    sv_ids_path = sv_dir / f"{case_id}_sv_ids.npy"
    if not sv_ids_path.exists():
        print(f"WARNING: SV IDs not found for {case_id}, skipping")
        return None

    sv_ids = np.load(sv_ids_path)

    # Load GT labels (check multiple locations)
    label_paths = [
        data_root / case_id / f"{case_id}_label.nii",
        data_root / f"{case_id}_label.nii",
        data_root / "data" / f"{case_id}_label.nii",
    ]
    label_path = None
    for path in label_paths:
        if path.exists():
            label_path = path
            break
    if label_path is None:
        print(f"WARNING: GT label not found for {case_id}, skipping")
        return None

    # Load image (check multiple locations)
    image_paths = [
        data_root / case_id / f"{case_id}_image.nii",
        data_root / f"{case_id}_image.nii",
        data_root / "data" / f"{case_id}_image.nii",
    ]
    image_path = None
    for path in image_paths:
        if path.exists():
            image_path = path
            break
    if image_path is None:
        print(f"WARNING: Image not found for {case_id}, skipping")
        return None

    # Load with MONAI transforms for RAS consistency
    transforms = Compose([
        LoadImaged(keys=["label", "image"]),
        EnsureChannelFirstd(keys=["label", "image"]),
        Orientationd(keys=["label", "image"], axcodes="RAS"),
    ])

    data = transforms({"label": str(label_path), "image": str(image_path)})

    gt_labels = data["label"][0].numpy().astype(np.int16)  # Remove channel dim
    image = data["image"][0].numpy().astype(np.float32)

    # Sample seeds
    seed_mask, seed_sv_labels = sample_strategic_seeds(
        sv_ids, gt_labels, image, budget_ratio, class_weights
    )

    # Compute statistics
    class_counts = Counter(seed_sv_labels.values())
    avg_grad_at_seeds = 0.0
    if seed_mask.sum() > 0:
        grad_mag = compute_gradient_magnitude(image)
        seed_coords = np.argwhere(seed_mask)
        avg_grad_at_seeds = grad_mag[seed_coords[:, 0], seed_coords[:, 1], seed_coords[:, 2]].mean()

    meta = {
        "case_id": case_id,
        "n_seeds": int(seed_mask.sum()),
        "n_labeled_svs": len(seed_sv_labels),
        "n_total_svs": int(len(np.unique(sv_ids))),
        "coverage_fraction": len(seed_sv_labels) / len(np.unique(sv_ids)),
        "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
        "avg_gradient_at_seeds": float(avg_grad_at_seeds),
        "budget_used": budget_ratio,
        "sv_ids_shape": list(sv_ids.shape),
    }

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{case_id}_strategic_seeds.npy", seed_mask)

    with open(output_dir / f"{case_id}_sv_labels_sparse.json", 'w') as f:
        json.dump({"sv_labels": {str(k): int(v) for k, v in seed_sv_labels.items()}}, f, indent=2)

    with open(output_dir / f"{case_id}_seeds_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


def main():
    parser = argparse.ArgumentParser(description="Strategic sparse SV seed sampling")
    parser.add_argument("--sv_dir", type=str, required=True,
                       help="Directory containing supervoxel IDs (*_sv_ids.npy)")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory containing GT labels and images")
    parser.add_argument("--split_cfg", type=str, required=True,
                       help="Split config JSON file")
    parser.add_argument("--split", type=str, default="train",
                       help="Split to process (default: train)")
    parser.add_argument("--budget_ratio", type=float, default=0.001,
                       help="Fraction of voxels to sample (default: 0.001 = 0.1%%)")
    parser.add_argument("--class_weights", type=str, default="1,1,2,2",
                       help="Class weights for 1,2,3,4 (default: 1,1,2,2)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for seeds and metadata")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--num_workers", type=int, default=1,
                       help="Number of parallel workers (default: 1)")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Parse class weights
    weights_list = [float(w) for w in args.class_weights.split(',')]
    if len(weights_list) != 4:
        print("ERROR: --class_weights must have 4 values (for classes 1,2,3,4)")
        sys.exit(1)
    class_weights = {1: weights_list[0], 2: weights_list[1],
                    3: weights_list[2], 4: weights_list[3]}

    # Load cases
    sv_dir = Path(args.sv_dir)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    cases = load_split_cases(args.split_cfg, data_root, args.split)
    print(f"Processing {len(cases)} cases from {args.split} split")
    print(f"Budget: {args.budget_ratio:.4f} ({args.budget_ratio*100:.2f}%)")
    print(f"Class weights: {class_weights}")

    # Process all cases
    all_meta = []
    for case_id in tqdm(cases, desc="Processing cases"):
        print(f"\nCase: {case_id}")
        meta = process_case(case_id, sv_dir, data_root, output_dir, args.budget_ratio, class_weights)
        if meta:
            all_meta.append(meta)

    # Summary statistics
    if all_meta:
        avg_seeds = np.mean([m["n_seeds"] for m in all_meta])
        avg_coverage = np.mean([m["coverage_fraction"] for m in all_meta])
        total_class_dist = Counter()
        for m in all_meta:
            for cls, count in m["class_distribution"].items():
                total_class_dist[cls] += count

        summary = {
            "n_cases": len(all_meta),
            "avg_seeds_per_case": float(avg_seeds),
            "avg_sv_coverage": float(avg_coverage),
            "total_class_distribution": {str(k): int(v) for k, v in total_class_dist.items()},
            "budget_ratio": args.budget_ratio,
            "class_weights": class_weights,
            "seed": args.seed,
        }

        with open(output_dir / "summary_stats.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*70)
        print("SAMPLING COMPLETE!")
        print("="*70)
        print(f"  Processed: {len(all_meta)} cases")
        print(f"  Avg seeds/case: {avg_seeds:.1f}")
        print(f"  Avg SV coverage: {avg_coverage*100:.1f}%")
        print(f"  Class distribution: {dict(total_class_dist)}")
        print(f"  Output: {output_dir}")
        print("="*70)


if __name__ == "__main__":
    main()
