#!/usr/bin/env python3
"""
Evaluate quality of strategic sparse seeds and propagated labels.

Computes:
1. Seed accuracy (seeds vs ground truth)
2. Propagated label quality (propagated vs ground truth)
3. Per-class Dice scores and confusion matrices

Usage:
    python3 scripts/evaluate_sparse_labels.py \
        --seeds_dir runs/strategic_sparse_0p1pct_k15_fixed/strategic_seeds \
        --propagated_dir runs/strategic_sparse_0p1pct_k15_fixed/k_variants/k15 \
        --gt_root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
        --output_dir tmp/evaluation_k15
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd
from tqdm import tqdm


def compute_dice(pred: np.ndarray, gt: np.ndarray, class_id: int) -> float:
    """Compute Dice score for a specific class."""
    pred_mask = (pred == class_id)
    gt_mask = (gt == class_id)

    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return 2.0 * intersection / union


def compute_confusion_matrix(pred: np.ndarray, gt: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """Compute confusion matrix."""
    # Flatten and filter valid labels
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    valid_mask = (gt_flat >= 0) & (gt_flat < num_classes)
    pred_flat = pred_flat[valid_mask]
    gt_flat = gt_flat[valid_mask]

    # Compute confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(num_classes):
        for j in range(num_classes):
            conf_matrix[i, j] = np.sum((gt_flat == i) & (pred_flat == j))

    return conf_matrix


def evaluate_seeds(
    seed_file: Path,
    gt_labels: np.ndarray,
    case_id: str
) -> Dict:
    """Evaluate seed quality against ground truth."""

    # Load seeds
    seeds = np.load(seed_file)

    # Find seeded voxels (non-zero)
    seed_mask = seeds > 0
    n_seeds = np.count_nonzero(seed_mask)

    if n_seeds == 0:
        return {
            'case_id': case_id,
            'n_seeds': 0,
            'accuracy': 0.0,
            'class_0_acc': 0.0,
            'class_1_acc': 0.0,
            'class_2_acc': 0.0,
            'class_3_acc': 0.0,
            'class_4_acc': 0.0,
        }

    # Extract seed labels and corresponding GT
    seed_labels = seeds[seed_mask]
    gt_at_seeds = gt_labels[seed_mask]

    # Overall accuracy
    correct = np.sum(seed_labels == gt_at_seeds)
    accuracy = correct / n_seeds

    # Per-class accuracy
    class_accs = {}
    for cls in range(5):
        cls_mask = (seed_labels == cls)
        if cls_mask.sum() > 0:
            cls_correct = np.sum((seed_labels == cls) & (gt_at_seeds == cls))
            class_accs[f'class_{cls}_acc'] = cls_correct / cls_mask.sum()
        else:
            class_accs[f'class_{cls}_acc'] = np.nan

    # Class distribution
    class_dist = {}
    for cls in range(5):
        class_dist[f'class_{cls}_count'] = np.sum(seed_labels == cls)

    return {
        'case_id': case_id,
        'n_seeds': int(n_seeds),
        'accuracy': float(accuracy),
        **class_accs,
        **class_dist,
    }


def evaluate_propagated(
    propagated_file: Path,
    gt_labels: np.ndarray,
    case_id: str
) -> Dict:
    """Evaluate propagated label quality against ground truth."""

    # Load propagated labels
    propagated = np.load(propagated_file)

    # Compute Dice scores per class
    dice_scores = {}
    for cls in range(5):
        dice_scores[f'dice_class_{cls}'] = compute_dice(propagated, gt_labels, cls)

    # Overall Dice (mean of classes 1-4, excluding background)
    dice_fg = np.mean([dice_scores[f'dice_class_{i}'] for i in range(1, 5)])
    dice_all = np.mean([dice_scores[f'dice_class_{i}'] for i in range(5)])

    # Per-class accuracy
    class_accs = {}
    for cls in range(5):
        cls_mask = (gt_labels == cls)
        if cls_mask.sum() > 0:
            cls_correct = np.sum((propagated == cls) & (gt_labels == cls))
            class_accs[f'acc_class_{cls}'] = cls_correct / cls_mask.sum()
        else:
            class_accs[f'acc_class_{cls}'] = np.nan

    # Coverage (% of voxels with valid labels, including class 0)
    # Note: -1 = unlabeled, 0-4 = valid labels, 6 = ignore
    coverage = np.sum(propagated >= 0) / propagated.size

    # Overall accuracy
    accuracy = np.mean(propagated == gt_labels)

    return {
        'case_id': case_id,
        'dice_fg': float(dice_fg),
        'dice_all': float(dice_all),
        'accuracy': float(accuracy),
        'coverage': float(coverage),
        **dice_scores,
        **class_accs,
    }


def load_gt_labels(case_id: str, gt_root: Path) -> np.ndarray:
    """Load ground truth labels."""

    # Try multiple possible locations
    label_paths = [
        gt_root / case_id / f"{case_id}_label.nii",
        gt_root / f"{case_id}_label.nii",
        gt_root / "data" / f"{case_id}_label.nii",
    ]

    label_path = None
    for path in label_paths:
        if path.exists():
            label_path = path
            break

    if label_path is None:
        raise FileNotFoundError(f"GT label not found for {case_id}")

    # Load with MONAI transforms for RAS consistency
    transforms = Compose([
        LoadImaged(keys=["label"]),
        EnsureChannelFirstd(keys=["label"]),
        Orientationd(keys=["label"], axcodes="RAS"),
    ])

    data = transforms({"label": str(label_path)})
    gt_labels = data["label"][0].numpy().astype(np.int16)

    return gt_labels


def main():
    parser = argparse.ArgumentParser(description="Evaluate sparse label quality")
    parser.add_argument("--seeds_dir", type=str, required=True,
                       help="Directory containing seed files (*_strategic_seeds.npy)")
    parser.add_argument("--propagated_dir", type=str, required=True,
                       help="Directory containing propagated labels (*_labels.npy)")
    parser.add_argument("--gt_root", type=str, required=True,
                       help="Root directory containing ground truth labels")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for evaluation results")

    args = parser.parse_args()

    seeds_dir = Path(args.seeds_dir)
    propagated_dir = Path(args.propagated_dir)
    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all seed files
    seed_files = sorted(seeds_dir.glob("*_strategic_seeds.npy"))
    print(f"Found {len(seed_files)} seed files")

    if len(seed_files) == 0:
        print("ERROR: No seed files found")
        sys.exit(1)

    # Evaluate all cases
    seed_results = []
    propagated_results = []
    seed_conf_matrix = np.zeros((5, 5), dtype=np.int64)
    prop_conf_matrix = np.zeros((5, 5), dtype=np.int64)

    for seed_file in tqdm(seed_files, desc="Evaluating cases"):
        # Extract case ID
        case_id = seed_file.stem.replace("_strategic_seeds", "")

        # Find corresponding propagated file
        propagated_file = propagated_dir / f"{case_id}_labels.npy"

        if not propagated_file.exists():
            print(f"WARNING: Propagated labels not found for {case_id}")
            continue

        try:
            # Load ground truth
            gt_labels = load_gt_labels(case_id, gt_root)

            # Evaluate seeds
            seed_eval = evaluate_seeds(seed_file, gt_labels, case_id)
            seed_results.append(seed_eval)

            # Evaluate propagated labels
            prop_eval = evaluate_propagated(propagated_file, gt_labels, case_id)
            propagated_results.append(prop_eval)

            # Accumulate confusion matrices
            seeds = np.load(seed_file)
            seed_mask = seeds > 0
            if seed_mask.sum() > 0:
                seed_labels = seeds[seed_mask]
                gt_at_seeds = gt_labels[seed_mask]
                for i in range(5):
                    for j in range(5):
                        seed_conf_matrix[i, j] += np.sum((gt_at_seeds == i) & (seed_labels == j))

            propagated = np.load(propagated_file)
            prop_conf_matrix += compute_confusion_matrix(propagated, gt_labels)

        except Exception as e:
            print(f"ERROR processing {case_id}: {e}")
            continue

    print(f"\nSuccessfully evaluated {len(seed_results)} cases")

    # Save per-case results
    seed_df = pd.DataFrame(seed_results)
    prop_df = pd.DataFrame(propagated_results)

    seed_df.to_csv(output_dir / "seed_evaluation_per_case.csv", index=False)
    prop_df.to_csv(output_dir / "propagated_evaluation_per_case.csv", index=False)

    print(f"\nPer-case results saved to {output_dir}/")

    # Compute aggregate statistics
    print("\n" + "="*70)
    print("SEED QUALITY EVALUATION")
    print("="*70)

    print(f"\nOverall Seed Accuracy: {seed_df['accuracy'].mean():.4f} ± {seed_df['accuracy'].std():.4f}")
    print(f"Total seeds: {seed_df['n_seeds'].sum()}")
    print(f"\nPer-class seed accuracy:")
    for cls in range(5):
        col = f'class_{cls}_acc'
        if col in seed_df.columns:
            acc = seed_df[col].mean()
            print(f"  Class {cls}: {acc:.4f}")

    print(f"\nSeed class distribution (total):")
    for cls in range(5):
        col = f'class_{cls}_count'
        if col in seed_df.columns:
            count = seed_df[col].sum()
            pct = count / seed_df['n_seeds'].sum() * 100
            print(f"  Class {cls}: {count:6d} ({pct:5.2f}%)")

    print(f"\nSeed Confusion Matrix (rows=GT, cols=pred):")
    print(seed_conf_matrix)

    print("\n" + "="*70)
    print("PROPAGATED LABEL QUALITY EVALUATION")
    print("="*70)

    print(f"\nOverall Dice (all classes 0-4): {prop_df['dice_all'].mean():.4f} ± {prop_df['dice_all'].std():.4f}")
    print(f"Foreground Dice (classes 1-4):  {prop_df['dice_fg'].mean():.4f} ± {prop_df['dice_fg'].std():.4f}")
    print(f"Overall Accuracy: {prop_df['accuracy'].mean():.4f}")
    print(f"Coverage: {prop_df['coverage'].mean():.4f}")

    print(f"\nPer-class Dice scores:")
    for cls in range(5):
        col = f'dice_class_{cls}'
        dice = prop_df[col].mean()
        std = prop_df[col].std()
        print(f"  Class {cls}: {dice:.4f} ± {std:.4f}")

    print(f"\nPer-class accuracy:")
    for cls in range(5):
        col = f'acc_class_{cls}'
        if col in prop_df.columns:
            acc = prop_df[col].mean()
            print(f"  Class {cls}: {acc:.4f}")

    print(f"\nPropagated Confusion Matrix (rows=GT, cols=pred):")
    print(prop_conf_matrix)

    # Save summary
    summary = {
        "seed_evaluation": {
            "overall_accuracy": float(seed_df['accuracy'].mean()),
            "total_seeds": int(seed_df['n_seeds'].sum()),
            "per_class_accuracy": {
                f"class_{cls}": float(seed_df[f'class_{cls}_acc'].mean())
                for cls in range(5)
                if f'class_{cls}_acc' in seed_df.columns
            },
            "class_distribution": {
                f"class_{cls}": int(seed_df[f'class_{cls}_count'].sum())
                for cls in range(5)
                if f'class_{cls}_count' in seed_df.columns
            },
            "confusion_matrix": seed_conf_matrix.tolist(),
        },
        "propagated_evaluation": {
            "dice_all": float(prop_df['dice_all'].mean()),
            "dice_fg": float(prop_df['dice_fg'].mean()),
            "overall_accuracy": float(prop_df['accuracy'].mean()),
            "coverage": float(prop_df['coverage'].mean()),
            "per_class_dice": {
                f"class_{cls}": float(prop_df[f'dice_class_{cls}'].mean())
                for cls in range(5)
            },
            "per_class_accuracy": {
                f"class_{cls}": float(prop_df[f'acc_class_{cls}'].mean())
                for cls in range(5)
                if f'acc_class_{cls}' in prop_df.columns
            },
            "confusion_matrix": prop_conf_matrix.tolist(),
        }
    }

    with open(output_dir / "evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {output_dir}/evaluation_summary.json")
    print("="*70)

    # Identify best and worst cases
    print("\n" + "="*70)
    print("CASE ANALYSIS")
    print("="*70)

    # Best and worst by propagated Dice
    prop_df_sorted = prop_df.sort_values('dice_fg', ascending=False)
    print("\nTop 5 cases (by foreground Dice):")
    for idx, row in prop_df_sorted.head(5).iterrows():
        print(f"  {row['case_id']}: Dice={row['dice_fg']:.4f}")

    print("\nWorst 5 cases (by foreground Dice):")
    for idx, row in prop_df_sorted.tail(5).iterrows():
        print(f"  {row['case_id']}: Dice={row['dice_fg']:.4f}")

    print("="*70)


if __name__ == "__main__":
    main()
