#!/usr/bin/env python3
"""
Evaluate propagation quality across multiple k values.

Usage:
    python3 scripts/evaluate_k_sweep.py \
        --case_id SN13B0_I17_3D_B1_1B250409 \
        --propagation_dir tmp/test_propagation_sweep \
        --gt_root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
        --k_values 1,3,5,7,10,15,20,30,50,100 \
        --output_file tmp/test_propagation_sweep/k_sweep_results.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd


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
    pred_flat = pred.flatten()
    gt_flat = gt.flatten()

    valid_mask = (gt_flat >= 0) & (gt_flat < num_classes)
    pred_flat = pred_flat[valid_mask]
    gt_flat = gt_flat[valid_mask]

    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(num_classes):
        for j in range(num_classes):
            conf_matrix[i, j] = np.sum((gt_flat == i) & (pred_flat == j))

    return conf_matrix


def evaluate_propagated(
    propagated: np.ndarray,
    gt: np.ndarray,
    k: int
) -> Dict:
    """Evaluate propagated labels against ground truth."""

    # Compute Dice scores per class
    dice_scores = {}
    for cls in range(5):
        dice_scores[cls] = compute_dice(propagated, gt, cls)

    # Overall and foreground Dice
    dice_fg = np.mean([dice_scores[i] for i in range(1, 5)])
    dice_all = np.mean([dice_scores[i] for i in range(5)])

    # Per-class accuracy
    class_accs = {}
    for cls in range(5):
        cls_mask = (gt == cls)
        if cls_mask.sum() > 0:
            cls_correct = np.sum((propagated == cls) & (gt == cls))
            class_accs[cls] = float(cls_correct / cls_mask.sum())
        else:
            class_accs[cls] = np.nan

    # Overall accuracy
    accuracy = float(np.mean(propagated == gt))

    # Coverage (should be 100% for propagated labels)
    coverage = float(np.count_nonzero(propagated >= 0) / propagated.size)

    # Confusion matrix
    conf_matrix = compute_confusion_matrix(propagated, gt)

    return {
        "k": k,
        "dice_all": float(dice_all),
        "dice_fg": float(dice_fg),
        "dice_class_0": float(dice_scores[0]),
        "dice_class_1": float(dice_scores[1]),
        "dice_class_2": float(dice_scores[2]),
        "dice_class_3": float(dice_scores[3]),
        "dice_class_4": float(dice_scores[4]),
        "accuracy": accuracy,
        "coverage": coverage,
        "acc_class_0": class_accs[0],
        "acc_class_1": class_accs[1],
        "acc_class_2": class_accs[2],
        "acc_class_3": class_accs[3],
        "acc_class_4": class_accs[4],
        "confusion_matrix": conf_matrix.tolist(),
    }


def load_gt_labels(case_id: str, gt_root: Path) -> np.ndarray:
    """Load ground truth labels."""

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

    transforms = Compose([
        LoadImaged(keys=["label"]),
        EnsureChannelFirstd(keys=["label"]),
        Orientationd(keys=["label"], axcodes="RAS"),
    ])

    data = transforms({"label": str(label_path)})
    gt_labels = data["label"][0].numpy().astype(np.int16)

    return gt_labels


def main():
    parser = argparse.ArgumentParser(description="Evaluate k sweep results")
    parser.add_argument("--case_id", type=str, required=True,
                       help="Case ID to evaluate")
    parser.add_argument("--propagation_dir", type=str, required=True,
                       help="Directory containing propagated results")
    parser.add_argument("--gt_root", type=str, required=True,
                       help="Root directory containing ground truth labels")
    parser.add_argument("--k_values", type=str, required=True,
                       help="Comma-separated k values")
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output JSON file for results")

    args = parser.parse_args()

    case_id = args.case_id
    propagation_dir = Path(args.propagation_dir)
    gt_root = Path(args.gt_root)
    k_values = [int(k) for k in args.k_values.split(',')]
    output_file = Path(args.output_file)

    print(f"Evaluating case: {case_id}")
    print(f"K values: {k_values}")

    # Load ground truth
    print("\nLoading ground truth...")
    gt_labels = load_gt_labels(case_id, gt_root)
    print(f"GT shape: {gt_labels.shape}")

    # Evaluate each k variant
    results = []

    for k in k_values:
        print(f"\nEvaluating k={k}...")

        k_str = f"{k:02d}"
        propagated_file = propagation_dir / "cases" / case_id / f"propagated_k{k_str}_labels.npy"

        if not propagated_file.exists():
            print(f"  WARNING: File not found: {propagated_file}")
            continue

        propagated = np.load(propagated_file)

        result = evaluate_propagated(propagated, gt_labels, k)
        results.append(result)

        print(f"  Dice (fg): {result['dice_fg']:.4f}")
        print(f"  Dice (all): {result['dice_all']:.4f}")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Coverage: {result['coverage']:.4f}")

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "case_id": case_id,
        "k_values": k_values,
        "results": results,
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_file}")

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY: Dice Scores vs k")
    print(f"{'='*70}")
    print(f"{'k':>5} | {'Dice(fg)':>10} | {'Dice(all)':>10} | {'Accuracy':>10} | {'Coverage':>10}")
    print(f"{'-'*70}")

    for r in results:
        print(f"{r['k']:>5} | {r['dice_fg']:>10.4f} | {r['dice_all']:>10.4f} | "
              f"{r['accuracy']:>10.4f} | {r['coverage']:>10.4f}")

    # Find best k
    best_k = max(results, key=lambda x: x['dice_fg'])
    print(f"\n{'='*70}")
    print(f"BEST k: {best_k['k']} (Foreground Dice: {best_k['dice_fg']:.4f})")
    print(f"{'='*70}")

    # Per-class Dice for best k
    print(f"\nPer-class Dice for k={best_k['k']}:")
    for cls in range(5):
        print(f"  Class {cls}: {best_k[f'dice_class_{cls}']:.4f}")


if __name__ == "__main__":
    main()
