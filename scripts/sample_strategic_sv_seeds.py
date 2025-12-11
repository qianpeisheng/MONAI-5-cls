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

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.ndimage import sobel, distance_transform_edt
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd
from tqdm import tqdm

# Ensure project root is on PYTHONPATH so wp5.* is importable when running as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wp5.weaklabel.outer_bg_utils import choose_outer_bg_distance


def load_split_cases(split_cfg: str, data_root: Path, split: str = "train") -> list:
    """Load case IDs from legacy split config (serial-based).

    Split config format:
        {"test_serial_numbers": [9, 12, 15, ...]}

    Train set = all cases except those with serial numbers in test set.

    This helper is intended for the legacy flat WP5 dataset rooted at
    `/data3/wp5/wp5-code/dataloaders/wp5-dataset`. For the new default dataset
    under `/data3/wp5_4_Dec_data/3ddl-dataset`, prefer passing a MONAI-style
    datalist (see --datalist) instead of mixing split_cfg with the new layout.
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

    # Find all cases from image files (support both legacy flat layout and new BumpDataset layout)
    cases = []
    # Legacy: data_root or data_root/data contains *_image.nii
    # New BumpDataset: images live in data_root/images (we pass data_root=/path/to/.../data)
    search_dirs = [data_root, data_root / "data", data_root / "images"]

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


def partition_supervoxels_by_distance(
    sv_ids: np.ndarray,
    gt_labels: np.ndarray,
    outer_bg_distance: float,
):
    """
    Partition supervoxels into:
      - fg_svs: SVs containing any foreground voxel (labels 1-4, excluding 6).
      - boundary_bg_svs: background SVs within `outer_bg_distance` voxels of FG.
      - outer_bg_svs: background SVs farther than `outer_bg_distance` from FG.

    Args:
        sv_ids: (X, Y, Z) int32, supervoxel IDs.
        gt_labels: (X, Y, Z) int16, ground truth labels.
        outer_bg_distance: distance threshold in voxels.

    Returns:
        (fg_svs, boundary_bg_svs, outer_bg_svs) as sets of SV IDs (ints).
    """
    if outer_bg_distance <= 0:
        raise ValueError("outer_bg_distance must be > 0 for partitioning.")

    # Foreground mask: valid foreground classes 1-4, excluding ignore (6).
    fg_mask = (gt_labels > 0) & (gt_labels != 6)

    unique_svs = np.unique(sv_ids)
    fg_svs = set()
    boundary_bg_svs = set()
    outer_bg_svs = set()

    if not np.any(fg_mask):
        # Degenerate case: no foreground at all – treat all as outer BG.
        outer_bg_svs = {int(sv_id) for sv_id in unique_svs}
        return fg_svs, boundary_bg_svs, outer_bg_svs

    # Distance (in voxels) from each non-foreground voxel to nearest foreground voxel.
    # fg_mask True -> distance 0; ~fg_mask True -> distance to nearest fg voxel.
    dist_to_fg = distance_transform_edt(~fg_mask)

    for sv_id in unique_svs:
        sv_id_int = int(sv_id)
        mask = sv_ids == sv_id

        # Any foreground voxel in this SV?
        if np.any(fg_mask[mask]):
            fg_svs.add(sv_id_int)
            continue

        # Background SV: classify by min distance to foreground.
        sv_dist = dist_to_fg[mask]
        if sv_dist.size == 0:
            outer_bg_svs.add(sv_id_int)
            continue

        min_dist = float(sv_dist.min())
        if min_dist > outer_bg_distance:
            outer_bg_svs.add(sv_id_int)
        else:
            boundary_bg_svs.add(sv_id_int)

    return fg_svs, boundary_bg_svs, outer_bg_svs


def sample_strategic_seeds(
    sv_ids: np.ndarray,
    gt_labels: np.ndarray,
    image: np.ndarray,
    budget_ratio: float = 0.001,
    class_weights: Dict[int, float] = None,
    gradient_weight: float = 1.0,
    centroid_weight: float = 0.5,
    verbose: bool = True,
    outer_bg_distance: float = 0.0,
    boundary_bg_fraction: float = 0.1,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Strategic seed sampling to maximize supervoxel coverage.

    If outer_bg_distance > 0, SVs are partitioned into:
      - foreground SVs (contain any class 1-4 voxel),
      - boundary background SVs (background close to foreground),
      - outer background SVs (background far from foreground).

    Seeds are then allocated only within the ROI SVs (foreground + boundary
    background), with a configurable fraction of the budget reserved for
    boundary background SVs. Outer background SVs are intended to be labeled
    as background later without consuming budget.

    Args:
        sv_ids: (X, Y, Z) int32, supervoxel IDs
        gt_labels: (X, Y, Z) int16, ground truth labels
        image: (X, Y, Z) float, intensity image
        budget_ratio: fraction of total voxels to sample (default 0.001 = 0.1%)
        class_weights: Dict of class→weight (default: {0:0.1, 1:1, 2:1, 3:2, 4:2})
        gradient_weight: weight for gradient component (default 1.0)
        centroid_weight: weight for centroid proximity (default 0.5)
        verbose: enable logging
        outer_bg_distance: if >0, enable outer-background partitioning with this
            distance (in voxels) as the threshold between boundary and outer BG.
        boundary_bg_fraction: fraction of the seed budget to spend on boundary
            background SVs when outer_bg_distance > 0.

    Returns:
        seed_mask: (X, Y, Z) bool, binary mask of sampled voxels
        seed_sv_labels: Dict {sv_id: label}, sparse SV labels (1:1 mapping)
    """
    if class_weights is None:
        class_weights = {0: 0.1, 1: 1.0, 2: 1.0, 3: 2.0, 4: 2.0}

    # Calculate budget
    total_voxels = sv_ids.size
    max_seeds = int(total_voxels * budget_ratio)

    # Get all supervoxels (including background)
    sv_unique = np.unique(sv_ids)
    all_svs = [int(sv_id) for sv_id in sv_unique]

    if verbose:
        print(f"  Processing {len(all_svs)} supervoxels")

    # Precompute gradient magnitude
    grad_mag = compute_gradient_magnitude(image)

    # Legacy behavior: no outer background split, sample over all SVs.
    use_outer_bg = outer_bg_distance is not None and outer_bg_distance > 0.0

    if not use_outer_bg:
        # Score candidates within each supervoxel
        sv_candidates = []  # List of (sv_id, voxel_coord, score, label)

        iterable = tqdm(all_svs, desc="  Scoring candidates", leave=False) if verbose else all_svs
        for sv_id in iterable:
            mask = (sv_ids == sv_id)
            coords = np.argwhere(mask)

            # Get labels for all voxels in this SV
            labels = gt_labels[coords[:, 0], coords[:, 1], coords[:, 2]]
            valid_mask = (labels >= 0) & (labels <= 4)  # Valid classes only
            if valid_mask.sum() == 0:
                continue

            labels_valid = labels[valid_mask]

            # Determine dominant class in this SV
            dominant_class = int(np.bincount(labels_valid).argmax())

            # Only sample voxels of the dominant class (to ensure class representation)
            class_mask = (labels == dominant_class)
            if class_mask.sum() == 0:
                continue

            coords_class = coords[class_mask]
            labels_class = labels[class_mask]

            # Compute scores for candidate voxels of this class
            scores = np.zeros(len(coords_class), dtype=float)

            # Component 1: Class priority weight
            scores += class_weights.get(dominant_class, 0)

            # Component 2: Border proximity (gradient magnitude)
            gradients = grad_mag[coords_class[:, 0], coords_class[:, 1], coords_class[:, 2]]
            if gradients.max() > 0:
                scores += gradient_weight * (gradients / gradients.max())

            # Component 3: Representativeness (distance to centroid, inverted)
            centroid = coords.mean(axis=0)
            distances = np.linalg.norm(coords_class - centroid, axis=1)
            if distances.max() > 0:
                scores -= centroid_weight * (distances / distances.max())

            # Select best candidate for this SV
            best_idx = scores.argmax()
            best_coord = coords_class[best_idx]
            best_label = labels_class[best_idx]
            best_score = scores[best_idx]

            sv_candidates.append((int(sv_id), tuple(best_coord), best_score, int(best_label)))

        # Stratified sampling: allocate budget proportionally to class frequency in GT
        unique_labels, label_counts = np.unique(
            gt_labels[(gt_labels >= 0) & (gt_labels <= 4)], return_counts=True
        )
        class_freq = {int(label): count for label, count in zip(unique_labels, label_counts)}
        total_freq = sum(class_freq.values())

        # Weight frequencies by class_weights so that foreground and rare
        # classes receive more of the seed budget (e.g., class 0 has small
        # weight, classes 3/4 have larger weights).
        weighted_freq = {
            cls: class_freq[cls] * class_weights.get(cls, 0.0) for cls in class_freq
        }
        total_weighted = sum(weighted_freq.values()) or float(total_freq)

        # Group candidates by class
        candidates_by_class = {cls: [] for cls in range(5)}
        for sv_id, coord, score, label in sv_candidates:
            candidates_by_class[label].append((sv_id, coord, score, label))

        # Allocate budget to each class proportionally
        selected_candidates = []
        for cls in range(5):
            if cls not in class_freq or class_freq[cls] == 0:
                continue

            # Calculate target number of seeds for this class, using
            # weighted frequencies to emphasize higher-priority classes.
            if cls in weighted_freq and weighted_freq[cls] > 0 and total_weighted > 0:
                class_ratio = weighted_freq[cls] / total_weighted
            else:
                class_ratio = class_freq[cls] / total_freq if total_freq > 0 else 0.0
            class_budget = int(max_seeds * class_ratio)

            # Select top candidates from this class
            class_candidates = candidates_by_class[cls]
            class_candidates.sort(key=lambda x: x[2], reverse=True)  # Sort by score
            selected_from_class = class_candidates[:class_budget]
            selected_candidates.extend(selected_from_class)

            if verbose and len(selected_from_class) > 0:
                print(f"  Class {cls}: {len(selected_from_class):4d} seeds ({class_ratio*100:.1f}% of budget)")

        if verbose:
            print(f"  Total selected: {len(selected_candidates)} seeds (budget: {max_seeds})")

        # Create seed mask and SV label dict
        seed_mask = np.zeros_like(sv_ids, dtype=bool)
        seed_sv_labels = {}

        for sv_id, coord, score, label in selected_candidates:
            seed_mask[coord] = True
            seed_sv_labels[sv_id] = label

        return seed_mask, seed_sv_labels

    # New behavior: outer background split enabled.
    fg_svs, boundary_bg_svs, outer_bg_svs = partition_supervoxels_by_distance(
        sv_ids=sv_ids,
        gt_labels=gt_labels,
        outer_bg_distance=outer_bg_distance,
    )

    roi_svs = sorted(list(fg_svs | boundary_bg_svs))

    if verbose:
        print(
            f"  Partitioned SVs -> fg: {len(fg_svs)}, "
            f"boundary_bg: {len(boundary_bg_svs)}, outer_bg: {len(outer_bg_svs)}"
        )

    # Score candidates only within ROI SVs.
    fg_candidates = []        # (sv_id, coord, score, label)
    boundary_candidates = []  # (sv_id, coord, score, label)

    iterable = tqdm(roi_svs, desc="  Scoring ROI candidates", leave=False) if verbose else roi_svs
    for sv_id in iterable:
        mask = (sv_ids == sv_id)
        coords = np.argwhere(mask)

        labels = gt_labels[coords[:, 0], coords[:, 1], coords[:, 2]]
        valid_mask = (labels >= 0) & (labels <= 4)
        if valid_mask.sum() == 0:
            continue

        labels_valid = labels[valid_mask]
        dominant_class = int(np.bincount(labels_valid).argmax())

        class_mask = (labels == dominant_class)
        if class_mask.sum() == 0:
            continue

        coords_class = coords[class_mask]
        labels_class = labels[class_mask]

        scores = np.zeros(len(coords_class), dtype=float)

        # Class weighting for foreground SVs, lightweight for boundary BG.
        if sv_id in fg_svs:
            scores += class_weights.get(dominant_class, 0)

        gradients = grad_mag[coords_class[:, 0], coords_class[:, 1], coords_class[:, 2]]
        if gradients.max() > 0:
            scores += gradient_weight * (gradients / gradients.max())

        centroid = coords.mean(axis=0)
        distances = np.linalg.norm(coords_class - centroid, axis=1)
        if distances.max() > 0:
            scores -= centroid_weight * (distances / distances.max())

        best_idx = scores.argmax()
        best_coord = coords_class[best_idx]
        best_label = labels_class[best_idx]
        best_score = scores[best_idx]

        entry = (int(sv_id), tuple(best_coord), best_score, int(best_label))
        if sv_id in boundary_bg_svs:
            boundary_candidates.append(entry)
        else:
            fg_candidates.append(entry)

    # Budget split between foreground and boundary background.
    boundary_bg_fraction = float(np.clip(boundary_bg_fraction, 0.0, 1.0))
    target_boundary = int(max_seeds * boundary_bg_fraction)
    target_fg = max_seeds - target_boundary

    # Clamp to available candidates and reassign any unused budget.
    target_boundary = min(target_boundary, len(boundary_candidates))
    target_fg = min(target_fg, len(fg_candidates))

    remaining = max_seeds - (target_boundary + target_fg)
    if remaining > 0:
        # Prefer adding more FG seeds if available, then boundary BG.
        extra_fg = min(remaining, len(fg_candidates) - target_fg)
        target_fg += extra_fg
        remaining -= extra_fg

        if remaining > 0:
            extra_boundary = min(remaining, len(boundary_candidates) - target_boundary)
            target_boundary += extra_boundary

    if verbose:
        print(
            f"  Seed budget (max {max_seeds}): "
            f"FG={target_fg}, boundary_bg={target_boundary}, "
            f"unused={max_seeds - (target_fg + target_boundary)}"
        )

    # Select foreground seeds with simple class-proportional allocation.
    selected_candidates = []
    if target_fg > 0 and fg_candidates:
        # Compute class distribution over valid GT labels (1-4).
        valid_gt = gt_labels[(gt_labels >= 0) & (gt_labels <= 4)]
        unique_labels, label_counts = np.unique(valid_gt, return_counts=True)
        class_freq = {int(label): count for label, count in zip(unique_labels, label_counts)}
        total_freq = sum(class_freq.values())

        weighted_freq = {
            cls: class_freq[cls] * class_weights.get(cls, 0.0) for cls in class_freq
        }
        total_weighted = sum(weighted_freq.values()) or float(total_freq)

        candidates_by_class = {cls: [] for cls in range(5)}
        for sv_id, coord, score, label in fg_candidates:
            candidates_by_class[label].append((sv_id, coord, score, label))

        for cls in range(5):
            if cls not in class_freq or class_freq[cls] == 0:
                continue
            if cls in weighted_freq and weighted_freq[cls] > 0 and total_weighted > 0:
                class_ratio = weighted_freq[cls] / total_weighted
            else:
                class_ratio = class_freq[cls] / total_freq if total_freq > 0 else 0.0
            class_budget = int(target_fg * class_ratio)
            class_candidates = candidates_by_class[cls]
            class_candidates.sort(key=lambda x: x[2], reverse=True)
            selected_from_class = class_candidates[:class_budget]
            selected_candidates.extend(selected_from_class)

    # If we didn't fill the FG budget (due to rounding or missing classes),
    # top up with remaining best-scoring FG candidates.
    if fg_candidates and len(selected_candidates) < target_fg:
        already = {(sv_id, coord) for sv_id, coord, _, _ in selected_candidates}
        remaining_fg = [c for c in fg_candidates if (c[0], c[1]) not in already]
        remaining_fg.sort(key=lambda x: x[2], reverse=True)
        need = target_fg - len(selected_candidates)
        selected_candidates.extend(remaining_fg[:need])

    # Boundary background seeds: simply take top-scoring candidates.
    if target_boundary > 0 and boundary_candidates:
        boundary_candidates.sort(key=lambda x: x[2], reverse=True)
        selected_candidates.extend(boundary_candidates[:target_boundary])

    if verbose:
        print(f"  Total selected (ROI): {len(selected_candidates)} seeds (budget: {max_seeds})")

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
    outer_bg_distance: float = 0.0,
    boundary_bg_fraction: float = 0.1,
    outer_bg_target_bg_frac: float = 0.0,
    outer_bg_min_distance: float = 1.0,
    outer_bg_max_distance: float = 32.0,
    verbose: bool = True,
) -> Dict:
    """Process one case: load data, sample seeds, save outputs."""

    # Load supervoxel IDs
    sv_ids_path = sv_dir / f"{case_id}_sv_ids.npy"
    if not sv_ids_path.exists():
        print(f"WARNING: SV IDs not found for {case_id}, skipping")
        return None

    sv_ids = np.load(sv_ids_path)

    # Load GT labels (check multiple locations / layouts)
    label_paths = [
        data_root / case_id / f"{case_id}_label.nii",
        data_root / f"{case_id}_label.nii",
        data_root / "data" / f"{case_id}_label.nii",
        data_root / "labels" / f"{case_id}_label.nii",
        data_root / "data" / "labels" / f"{case_id}_label.nii",
    ]
    label_path = None
    for path in label_paths:
        if path.exists():
            label_path = path
            break
    if label_path is None:
        print(f"WARNING: GT label not found for {case_id}, skipping")
        return None

    # Load image (check multiple locations / layouts)
    image_paths = [
        data_root / case_id / f"{case_id}_image.nii",
        data_root / f"{case_id}_image.nii",
        data_root / "data" / f"{case_id}_image.nii",
        data_root / "images" / f"{case_id}_image.nii",
        data_root / "data" / "images" / f"{case_id}_image.nii",
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

    # Optional outer background partitioning (for metadata).
    fg_svs = set()
    boundary_bg_svs = set()
    outer_bg_svs = set()
    effective_outer_bg_distance = 0.0
    outer_bg_bg_fraction = 0.0

    if outer_bg_distance is not None and outer_bg_distance > 0.0:
        # Fixed global threshold (legacy behavior).
        effective_outer_bg_distance = float(outer_bg_distance)
        fg_svs, boundary_bg_svs, outer_bg_svs = partition_supervoxels_by_distance(
            sv_ids=sv_ids,
            gt_labels=gt_labels,
            outer_bg_distance=effective_outer_bg_distance,
        )
    elif outer_bg_target_bg_frac is not None and outer_bg_target_bg_frac > 0.0:
        # Adaptive per-volume threshold based on GT background distances.
        effective_outer_bg_distance, outer_bg_bg_fraction = choose_outer_bg_distance(
            gt_labels=gt_labels,
            target_outer_bg_frac=float(outer_bg_target_bg_frac),
            min_distance=float(outer_bg_min_distance),
            max_distance=float(outer_bg_max_distance),
        )
        if effective_outer_bg_distance > 0.0:
            fg_svs, boundary_bg_svs, outer_bg_svs = partition_supervoxels_by_distance(
                sv_ids=sv_ids,
                gt_labels=gt_labels,
                outer_bg_distance=effective_outer_bg_distance,
            )

    # Sample seeds
    seed_mask, seed_sv_labels = sample_strategic_seeds(
        sv_ids=sv_ids,
        gt_labels=gt_labels,
        image=image,
        budget_ratio=budget_ratio,
        class_weights=class_weights,
        gradient_weight=1.0,
        centroid_weight=0.5,
        verbose=verbose,
        outer_bg_distance=effective_outer_bg_distance,
        boundary_bg_fraction=boundary_bg_fraction,
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
    # Optional region stats for outer-background-aware runs.
    if effective_outer_bg_distance > 0.0:
        meta.update(
            {
                "outer_bg_distance": float(effective_outer_bg_distance),
                "boundary_bg_fraction": float(boundary_bg_fraction),
                "outer_bg_target_bg_frac": float(max(0.0, float(outer_bg_target_bg_frac))),
                "outer_bg_bg_fraction": float(outer_bg_bg_fraction),
                "n_fg_svs": int(len(fg_svs)),
                "n_boundary_bg_svs": int(len(boundary_bg_svs)),
                "n_outer_bg_svs": int(len(outer_bg_svs)),
                "outer_bg_sv_ids": [int(s) for s in sorted(outer_bg_svs)],
            }
        )

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{case_id}_strategic_seeds.npy", seed_mask)

    with open(output_dir / f"{case_id}_sv_labels_sparse.json", 'w') as f:
        json.dump({"sv_labels": {str(k): int(v) for k, v in seed_sv_labels.items()}}, f, indent=2)

    with open(output_dir / f"{case_id}_seeds_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


def _process_case_worker(
    payload: Tuple[str, str, str, str, float, Dict[int, float], float, float, float, float]
) -> Dict:
    """Wrapper for multiprocessing: payload carries only picklable types."""
    (
        case_id,
        sv_dir_str,
        data_root_str,
        output_dir_str,
        budget_ratio,
        class_weights,
        outer_bg_distance,
        boundary_bg_fraction,
        outer_bg_target_bg_frac,
        outer_bg_min_distance,
        outer_bg_max_distance,
    ) = payload
    sv_dir = Path(sv_dir_str)
    data_root = Path(data_root_str)
    output_dir = Path(output_dir_str)
    try:
        return process_case(
            case_id=case_id,
            sv_dir=sv_dir,
            data_root=data_root,
            output_dir=output_dir,
            budget_ratio=budget_ratio,
            class_weights=class_weights,
            outer_bg_distance=outer_bg_distance,
            boundary_bg_fraction=boundary_bg_fraction,
            outer_bg_target_bg_frac=outer_bg_target_bg_frac,
            outer_bg_min_distance=outer_bg_min_distance,
            outer_bg_max_distance=outer_bg_max_distance,
            verbose=False,
        )
    except Exception as e:
        print(f"ERROR processing {case_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Strategic sparse SV seed sampling")
    parser.add_argument("--sv_dir", type=str, required=True,
                       help="Directory containing supervoxel IDs (*_sv_ids.npy)")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory containing GT labels and images")
    parser.add_argument(
        "--split_cfg",
        type=str,
        default="",
        help=(
            "Legacy split config JSON (serial-number based). "
            "For the new WP5 dataset, prefer --datalist instead."
        ),
    )
    parser.add_argument(
        "--datalist",
        type=str,
        default="",
        help=(
            "Optional MONAI-style datalist JSON with records {image,label,id}. "
            "If provided, overrides --split_cfg and uses the datalist ids as cases."
        ),
    )
    parser.add_argument("--split", type=str, default="train",
                       help="Split to process (default: train; ignored when --datalist is set)")
    parser.add_argument("--budget_ratio", type=float, default=0.001,
                       help="Fraction of voxels to sample (default: 0.001 = 0.1%%)")
    parser.add_argument("--class_weights", type=str, default="0.1,1,1,2,2",
                       help="Class weights for 0,1,2,3,4 (default: 0.1,1,1,2,2)")
    parser.add_argument(
        "--outer_bg_distance",
        type=float,
        default=0.0,
        help=(
            "If >0, enable outer background split using this distance (voxels) "
            "to separate boundary vs outer background SVs."
        ),
    )
    parser.add_argument(
        "--outer_bg_target_bg_frac",
        type=float,
        default=0.0,
        help=(
            "If >0 and --outer_bg_distance<=0, enable adaptive outer-background "
            "threshold so that approximately this fraction of GT background voxels "
            "are treated as outer BG for each case."
        ),
    )
    parser.add_argument(
        "--outer_bg_min_distance",
        type=float,
        default=1.0,
        help="Minimum allowed outer-background distance when using adaptive mode (default: 1.0).",
    )
    parser.add_argument(
        "--outer_bg_max_distance",
        type=float,
        default=32.0,
        help="Maximum allowed outer-background distance when using adaptive mode (default: 32.0).",
    )
    parser.add_argument(
        "--boundary_bg_fraction",
        type=float,
        default=0.1,
        help=(
            "Fraction of seed budget to allocate to boundary-background SVs "
            "when outer_bg_distance>0 (default: 0.1)."
        ),
    )
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for seeds and metadata")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--num_workers", type=int, default=16,
                       help="Number of parallel workers (default: 16)")

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Parse class weights
    weights_list = [float(w) for w in args.class_weights.split(',')]
    if len(weights_list) != 5:
        print("ERROR: --class_weights must have 5 values (for classes 0,1,2,3,4)")
        sys.exit(1)
    class_weights = {0: weights_list[0], 1: weights_list[1], 2: weights_list[2],
                    3: weights_list[3], 4: weights_list[4]}

    # Load cases
    sv_dir = Path(args.sv_dir)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    if args.datalist:
        # New dataset path: rely on MONAI-style datalist (e.g., datalist_train_new.json)
        try:
            recs = json.loads(Path(args.datalist).read_text())
        except Exception as e:
            print(f"ERROR: Failed to read datalist '{args.datalist}': {e}")
            sys.exit(1)
        cases = sorted({str(rec.get("id")) for rec in recs if rec.get("id")})
        if not cases:
            print(f"ERROR: No 'id' entries found in datalist '{args.datalist}'")
            sys.exit(1)
        print(f"Using datalist '{args.datalist}' with {len(cases)} cases")
    else:
        if not args.split_cfg:
            print("ERROR: Must provide either --datalist (new dataset) or --split_cfg (legacy dataset).")
            sys.exit(1)
        cases = load_split_cases(args.split_cfg, data_root, args.split)

    print(f"Processing {len(cases)} cases from {args.split} split")
    print(f"Budget: {args.budget_ratio:.4f} ({args.budget_ratio*100:.2f}%)")
    print(f"Class weights: {class_weights}")
    if args.outer_bg_distance > 0.0:
        print(
            f"Outer background enabled (fixed): distance={args.outer_bg_distance}, "
            f"boundary_bg_fraction={args.boundary_bg_fraction}"
        )
    elif args.outer_bg_target_bg_frac > 0.0:
        print(
            "Outer background enabled (adaptive): "
            f"target_bg_frac={args.outer_bg_target_bg_frac}, "
            f"min_distance={args.outer_bg_min_distance}, "
            f"max_distance={args.outer_bg_max_distance}, "
            f"boundary_bg_fraction={args.boundary_bg_fraction}"
        )

    all_meta = []
    num_workers = max(int(args.num_workers), 1)
    num_workers = min(num_workers, len(cases)) if cases else 0

    if num_workers <= 1:
        # Sequential processing (original behavior)
        for case_id in tqdm(cases, desc="Processing cases"):
            meta = process_case(
                case_id=case_id,
                sv_dir=sv_dir,
                data_root=data_root,
                output_dir=output_dir,
                budget_ratio=args.budget_ratio,
                class_weights=class_weights,
                outer_bg_distance=args.outer_bg_distance,
                boundary_bg_fraction=args.boundary_bg_fraction,
                outer_bg_target_bg_frac=args.outer_bg_target_bg_frac,
                outer_bg_min_distance=args.outer_bg_min_distance,
                outer_bg_max_distance=args.outer_bg_max_distance,
                verbose=True,
            )
            if meta:
                all_meta.append(meta)
    else:
        print(f"Using {num_workers} workers over {len(cases)} cases")
        payloads = [
            (
                case_id,
                str(sv_dir),
                str(data_root),
                str(output_dir),
                float(args.budget_ratio),
                class_weights,
                float(args.outer_bg_distance),
                float(args.boundary_bg_fraction),
                float(args.outer_bg_target_bg_frac),
                float(args.outer_bg_min_distance),
                float(args.outer_bg_max_distance),
            )
            for case_id in cases
        ]
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_process_case_worker, pl) for pl in payloads]
            try:
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing cases"):
                    meta = fut.result()
                    if meta:
                        all_meta.append(meta)
            except KeyboardInterrupt:
                print("KeyboardInterrupt received, shutting down workers...")
                ex.shutdown(wait=False, cancel_futures=True)
                raise

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
