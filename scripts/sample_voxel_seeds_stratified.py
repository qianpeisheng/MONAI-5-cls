#!/usr/bin/env python3
"""
Stratified voxel seed sampling for Zhou diffusion training.

Samples 0.1% of voxels with 75% foreground / 25% background stratification.

Usage:
    python3 scripts/sample_voxel_seeds_stratified.py \
        --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
        --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json \
        --budget_ratio 0.001 \
        --fg_ratio 0.75 \
        --output_dir runs/zhou_voxel_seeds_0p1pct \
        --seed 42

Outputs (per case):
    - seeds/<case_id>_seed_labels.npy - Dense array: -1 for unlabeled, 0-4 for labeled
    - seeds/<case_id>_seed_meta.json - Statistics
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd
from tqdm import tqdm


def load_split_cases(split_cfg: str, data_root: Path, split: str = "train") -> list:
    """Load case IDs from split config."""
    import re
    import os

    with open(split_cfg) as f:
        cfg = json.load(f)

    test_serials = set(cfg.get("test_serial_numbers", []))

    def serial_from_name(name: str):
        """Extract serial number from case name like 'SN13B0_...'"""
        m = re.match(r"^SN(\d+)", name)
        return int(m.group(1)) if m else None

    # Find all cases from image files
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

    return sorted(list(set(cases)))


def sample_stratified_seeds(
    gt_labels: np.ndarray,
    budget_ratio: float = 0.001,
    fg_ratio: float = 0.75,
    seed: int = 42,
) -> np.ndarray:
    """
    Sample voxel seeds with foreground/background stratification.

    Args:
        gt_labels: (D, H, W) ground truth labels (0-4)
        budget_ratio: fraction of voxels to sample (default 0.001 = 0.1%)
        fg_ratio: fraction of seeds from foreground (default 0.75 = 75%)
        seed: random seed

    Returns:
        seed_labels: (D, H, W) array with -1 for unlabeled, 0-4 for labeled
    """
    np.random.seed(seed)

    # Calculate budget
    total_voxels = gt_labels.size
    n_seeds_total = int(budget_ratio * total_voxels)
    n_seeds_fg = int(n_seeds_total * fg_ratio)
    n_seeds_bg = n_seeds_total - n_seeds_fg

    # Initialize seed labels to -1 (unlabeled)
    seed_labels = np.full_like(gt_labels, -1, dtype=np.int16)

    # Get foreground and background coordinates
    fg_mask = (gt_labels > 0)
    bg_mask = (gt_labels == 0)

    fg_coords = np.argwhere(fg_mask)
    bg_coords = np.argwhere(bg_mask)

    n_fg_available = len(fg_coords)
    n_bg_available = len(bg_coords)

    # Sample foreground
    if n_fg_available > 0:
        n_fg_sample = min(n_seeds_fg, n_fg_available)
        fg_indices = np.random.choice(n_fg_available, n_fg_sample, replace=False)
        fg_sampled = fg_coords[fg_indices]

        for coord in fg_sampled:
            d, h, w = coord
            seed_labels[d, h, w] = gt_labels[d, h, w]

    # Sample background
    if n_bg_available > 0:
        n_bg_sample = min(n_seeds_bg, n_bg_available)
        bg_indices = np.random.choice(n_bg_available, n_bg_sample, replace=False)
        bg_sampled = bg_coords[bg_indices]

        for coord in bg_sampled:
            d, h, w = coord
            seed_labels[d, h, w] = gt_labels[d, h, w]

    return seed_labels


def process_case(case_id: str, data_root: Path, output_dir: Path, budget_ratio: float, fg_ratio: float, seed: int):
    """Process one case: load GT, sample seeds, save."""

    # Load ground truth
    transforms = Compose([
        LoadImaged(keys=["label"]),
        EnsureChannelFirstd(keys=["label"]),
        Orientationd(keys=["label"], axcodes="RAS"),
    ])

    label_path = data_root / "data" / f"{case_id}_label.nii"
    if not label_path.exists():
        print(f"WARNING: Label not found for {case_id}, skipping")
        return None

    data_dict = transforms({"label": str(label_path)})
    gt_labels = data_dict["label"].squeeze().numpy().astype(np.int16)

    # Sample seeds
    seed_labels = sample_stratified_seeds(gt_labels, budget_ratio, fg_ratio, seed)

    # Compute statistics
    labeled_mask = (seed_labels >= 0)
    n_labeled = np.sum(labeled_mask)
    n_total = gt_labels.size

    class_counts = {}
    for cls in range(5):
        n_cls = np.sum(seed_labels == cls)
        class_counts[f"class_{cls}"] = int(n_cls)

    n_fg = sum(class_counts[f"class_{cls}"] for cls in range(1, 5))
    n_bg = class_counts["class_0"]

    meta = {
        "case_id": case_id,
        "n_seeds": int(n_labeled),
        "n_total_voxels": int(n_total),
        "seed_percentage": float(n_labeled / n_total * 100),
        "n_foreground_seeds": int(n_fg),
        "n_background_seeds": int(n_bg),
        "fg_ratio_actual": float(n_fg / n_labeled) if n_labeled > 0 else 0.0,
        "class_counts": class_counts,
        "volume_shape": list(gt_labels.shape),
    }

    # Save outputs
    seeds_dir = output_dir / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    np.save(seeds_dir / f"{case_id}_seed_labels.npy", seed_labels)

    with open(seeds_dir / f"{case_id}_seed_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


def main():
    parser = argparse.ArgumentParser(description="Stratified voxel seed sampling")
    parser.add_argument("--data_root", type=str,
                       default="/data3/wp5/wp5-code/dataloaders/wp5-dataset",
                       help="Data root directory")
    parser.add_argument("--split_cfg", type=str,
                       default="/data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json",
                       help="Split config JSON")
    parser.add_argument("--budget_ratio", type=float, default=0.001,
                       help="Fraction of voxels to sample (default: 0.001 = 0.1%)")
    parser.add_argument("--fg_ratio", type=float, default=0.75,
                       help="Fraction of seeds from foreground (default: 0.75 = 75%)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("STRATIFIED VOXEL SEED SAMPLING")
    print("="*80)
    print(f"Budget: {args.budget_ratio*100:.3f}% of voxels")
    print(f"FG/BG split: {args.fg_ratio*100:.0f}% / {(1-args.fg_ratio)*100:.0f}%")
    print(f"Random seed: {args.seed}")
    print()

    # Load train cases
    cases = load_split_cases(args.split_cfg, data_root, split="train")
    print(f"Found {len(cases)} training cases")
    print()

    # Process all cases
    all_meta = []
    for case_id in tqdm(cases, desc="Sampling seeds"):
        meta = process_case(case_id, data_root, output_dir, args.budget_ratio, args.fg_ratio, args.seed)
        if meta:
            all_meta.append(meta)

    # Summary
    if all_meta:
        avg_seeds = np.mean([m["n_seeds"] for m in all_meta])
        avg_fg_ratio = np.mean([m["fg_ratio_actual"] for m in all_meta])

        summary = {
            "n_cases": len(all_meta),
            "budget_ratio": args.budget_ratio,
            "fg_ratio_target": args.fg_ratio,
            "avg_seeds_per_case": float(avg_seeds),
            "avg_fg_ratio_actual": float(avg_fg_ratio),
            "seed": args.seed,
        }

        with open(output_dir / "sampling_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*80)
        print("SEED SAMPLING COMPLETE!")
        print("="*80)
        print(f"  Processed: {len(all_meta)} cases")
        print(f"  Avg seeds per case: {avg_seeds:.0f}")
        print(f"  Avg FG ratio: {avg_fg_ratio*100:.1f}%")
        print(f"  Output: {output_dir}")
        print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
