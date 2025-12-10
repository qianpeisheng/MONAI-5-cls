#!/usr/bin/env python3
"""
Comprehensive grid search for Zhou diffusion hyperparameters.

Tests alpha × tolerance combinations to find optimal parameters.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from itertools import product

import numpy as np
import torch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wp5.weaklabel.voxel_diffusion import diffuse_labels_3d


def compute_dice(pred: np.ndarray, gt: np.ndarray, class_id: int) -> float:
    """Compute Dice score for a specific class."""
    pred_mask = (pred == class_id)
    gt_mask = (gt == class_id)

    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return 2.0 * intersection / union


def compute_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int = 5):
    """Compute evaluation metrics."""
    metrics = {}

    # Overall accuracy
    correct = np.sum(pred == gt)
    total = pred.size
    metrics['accuracy'] = float(correct / total)

    # Per-class Dice scores
    dice_scores = {}
    for cls in range(num_classes):
        dice = compute_dice(pred, gt, cls)
        dice_scores[f'class_{cls}'] = float(dice)

    metrics['dice_scores'] = dice_scores
    metrics['mean_dice'] = float(np.mean(list(dice_scores.values())))

    return metrics


def run_single_config(Y, labeled_mask, gt_labels, alpha, tol, connectivity, max_iter, device, num_classes):
    """Run Zhou diffusion with a single configuration."""

    # Run diffusion
    start_time = time.time()
    F = diffuse_labels_3d(
        Y=Y,
        labeled_mask=labeled_mask,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        connectivity=connectivity,
        device=device,
    )
    runtime = time.time() - start_time

    # Extract predictions
    F_cpu = F.cpu()
    pred_labels = torch.argmax(F_cpu, dim=1).squeeze().numpy().astype(np.int64)

    # Compute metrics
    metrics = compute_metrics(pred_labels, gt_labels, num_classes)

    return {
        'alpha': alpha,
        'tol': tol,
        'connectivity': connectivity,
        'max_iter': max_iter,
        'runtime': runtime,
        'mean_dice': metrics['mean_dice'],
        'accuracy': metrics['accuracy'],
        'dice_scores': metrics['dice_scores'],
    }


def main():
    parser = argparse.ArgumentParser(description="Grid search for Zhou diffusion parameters")
    parser.add_argument("--case_id", type=str, default="SN13B0_I17_3D_B1_1B250409", help="Case ID")
    parser.add_argument("--seed_file", type=str,
                        default="tmp/test_seeds/SN13B0_I17_3D_B1_1B250409_0p5pct_seed_labels.npy",
                        help="Path to seed labels .npy file")
    parser.add_argument("--data_root", type=str,
                        default="/data3/wp5/wp5-code/dataloaders/wp5-dataset",
                        help="Root directory for data")
    parser.add_argument("--output_dir", type=str,
                        default="tmp/zhou_grid_search",
                        help="Output directory")
    parser.add_argument("--connectivity", type=int, default=6, help="Neighborhood connectivity")
    parser.add_argument("--max_iter", type=int, default=500, help="Max diffusion iterations")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device (cuda:0, cuda:1, or cpu)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    print("="*80)
    print("ZHOU DIFFUSION GRID SEARCH")
    print("="*80)
    print(f"Case ID: {args.case_id}")
    print(f"Seed file: {args.seed_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print()

    # Check device availability
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load ground truth
    print("Loading data...")
    data_root = Path(args.data_root)
    image_path = data_root / "data" / f"{args.case_id}_image.nii"
    label_path = data_root / "data" / f"{args.case_id}_label.nii"

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ])

    data_dict = transforms({
        "image": str(image_path),
        "label": str(label_path),
    })

    gt_labels = data_dict["label"].squeeze().numpy().astype(np.int64)
    print(f"  Ground truth shape: {gt_labels.shape}")

    # Load seed labels
    seed_labels = np.load(args.seed_file)
    # Handle files with -1 for unlabeled (include class 0 as labeled)
    seed_mask = seed_labels >= 0
    n_seeds = np.count_nonzero(seed_mask)
    print(f"  Number of seeds: {n_seeds:,} ({n_seeds/seed_labels.size*100:.4f}%)")

    # Prepare input
    D, H, W = gt_labels.shape
    Y = torch.zeros((1, args.num_classes, D, H, W), dtype=torch.float32)
    for cls in range(args.num_classes):
        class_mask = (seed_labels == cls)
        Y[0, cls, class_mask] = 1.0

    labeled_mask = torch.from_numpy(seed_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    # Define grid search parameters
    alpha_values = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    tol_values = [1e-3, 1e-4, 1e-5, 1e-6]

    total_configs = len(alpha_values) * len(tol_values)
    print(f"\nGrid search parameters:")
    print(f"  Alpha values: {len(alpha_values)} values from {min(alpha_values)} to {max(alpha_values)}")
    print(f"  Tolerance values: {len(tol_values)} values from {min(tol_values):.0e} to {max(tol_values):.0e}")
    print(f"  Connectivity: {args.connectivity}")
    print(f"  Total configurations: {total_configs}")
    print()

    # Run grid search
    results = []
    best_result = None
    best_dice = 0.0

    print("Running grid search...")
    with tqdm(total=total_configs, desc="Grid search") as pbar:
        for alpha, tol in product(alpha_values, tol_values):
            result = run_single_config(
                Y=Y,
                labeled_mask=labeled_mask,
                gt_labels=gt_labels,
                alpha=alpha,
                tol=tol,
                connectivity=args.connectivity,
                max_iter=args.max_iter,
                device=args.device,
                num_classes=args.num_classes,
            )

            results.append(result)

            # Track best result
            if result['mean_dice'] > best_dice:
                best_dice = result['mean_dice']
                best_result = result

            # Update progress bar with current best
            pbar.set_postfix({
                'best_dice': f"{best_dice:.4f}",
                'current': f"α={alpha:.2f}, tol={tol:.0e}"
            })
            pbar.update(1)

    print(f"\n✓ Grid search completed!")
    print(f"  Total configurations tested: {len(results)}")
    print(f"  Best Dice score: {best_dice:.4f}")
    print(f"  Best configuration: alpha={best_result['alpha']}, tol={best_result['tol']:.0e}")
    print()

    # Save all results
    grid_results = {
        'case_id': args.case_id,
        'n_seeds': int(n_seeds),
        'seed_percentage': float(n_seeds / seed_labels.size * 100),
        'connectivity': args.connectivity,
        'max_iter': args.max_iter,
        'alpha_values': alpha_values,
        'tol_values': tol_values,
        'total_configs': total_configs,
        'results': results,
    }

    grid_file = output_dir / "grid_results.json"
    with open(grid_file, 'w') as f:
        json.dump(grid_results, f, indent=2)
    print(f"Saved grid results to: {grid_file}")

    # Save best configuration
    best_config = {
        'case_id': args.case_id,
        'best_alpha': best_result['alpha'],
        'best_tol': best_result['tol'],
        'connectivity': args.connectivity,
        'max_iter': args.max_iter,
        'mean_dice': best_result['mean_dice'],
        'accuracy': best_result['accuracy'],
        'dice_scores': best_result['dice_scores'],
        'runtime': best_result['runtime'],
    }

    best_file = output_dir / "best_config.json"
    with open(best_file, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"Saved best configuration to: {best_file}")

    # Print top 10 results
    print("\n" + "="*80)
    print("TOP 10 CONFIGURATIONS")
    print("="*80)
    sorted_results = sorted(results, key=lambda x: x['mean_dice'], reverse=True)[:10]
    for i, res in enumerate(sorted_results, 1):
        print(f"{i:2d}. α={res['alpha']:.2f}, tol={res['tol']:.0e} → "
              f"Dice={res['mean_dice']:.4f}, Acc={res['accuracy']:.4f}, "
              f"Time={res['runtime']:.2f}s")

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
