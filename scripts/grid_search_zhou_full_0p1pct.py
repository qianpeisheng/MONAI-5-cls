#!/usr/bin/env python3
"""
Full grid search: alpha × tolerance × connectivity at 0.1% seeds.
"""

import json
import sys
import time
from pathlib import Path
from itertools import product

import numpy as np
import torch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from wp5.weaklabel.voxel_diffusion import diffuse_labels_3d


def compute_dice(pred: np.ndarray, gt: np.ndarray, class_id: int) -> float:
    pred_mask = (pred == class_id)
    gt_mask = (gt == class_id)
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return 2.0 * intersection / union


def compute_metrics(pred: np.ndarray, gt: np.ndarray, num_classes: int = 5):
    metrics = {}
    correct = np.sum(pred == gt)
    total = pred.size
    metrics['accuracy'] = float(correct / total)

    dice_scores = {}
    for cls in range(num_classes):
        dice = compute_dice(pred, gt, cls)
        dice_scores[f'class_{cls}'] = float(dice)

    metrics['dice_scores'] = dice_scores
    metrics['mean_dice'] = float(np.mean(list(dice_scores.values())))
    return metrics


def run_single_config(Y, labeled_mask, gt_labels, alpha, tol, connectivity, max_iter, device, num_classes):
    start_time = time.time()
    F = diffuse_labels_3d(Y=Y, labeled_mask=labeled_mask, alpha=alpha, max_iter=max_iter,
                          tol=tol, connectivity=connectivity, device=device)
    runtime = time.time() - start_time

    F_cpu = F.cpu()
    pred_labels = torch.argmax(F_cpu, dim=1).squeeze().numpy().astype(np.int64)
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
    case_id = "SN13B0_I17_3D_B1_1B250409"
    seed_file = "tmp/test_one_case/SN13B0_I17_3D_B1_1B250409_strategic_seeds.npy"
    data_root = "/data3/wp5/wp5-code/dataloaders/wp5-dataset"
    output_dir = Path("tmp/zhou_grid_search_0p1pct_full")
    device = "cuda:1"
    num_classes = 5
    max_iter = 500

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("FULL ZHOU GRID SEARCH AT 0.1%: ALPHA × TOLERANCE × CONNECTIVITY")
    print("="*80)
    print()

    # Load data
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ])

    data_dict = transforms({
        "image": f"{data_root}/data/{case_id}_image.nii",
        "label": f"{data_root}/data/{case_id}_label.nii",
    })

    gt_labels = data_dict["label"].squeeze().numpy().astype(np.int64)
    seed_labels = np.load(seed_file)
    seed_mask = seed_labels >= 0
    n_seeds = np.count_nonzero(seed_mask)

    print(f"Seeds: {n_seeds:,} ({n_seeds/seed_labels.size*100:.4f}%)")
    print()

    # Prepare input
    D, H, W = gt_labels.shape
    Y = torch.zeros((1, num_classes, D, H, W), dtype=torch.float32)
    for cls in range(num_classes):
        class_mask = (seed_labels == cls)
        Y[0, cls, class_mask] = 1.0

    labeled_mask = torch.from_numpy(seed_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    # Grid parameters
    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    tol_values = [1e-3, 1e-4]  # Only test 2 tolerance values for speed
    connectivity_values = [6, 26]

    total_configs = len(alpha_values) * len(tol_values) * len(connectivity_values)
    print(f"Total configurations: {total_configs}")
    print()

    # Run grid search
    results = []
    best_dice = 0.0

    with tqdm(total=total_configs, desc="Grid search") as pbar:
        for alpha, tol, connectivity in product(alpha_values, tol_values, connectivity_values):
            result = run_single_config(
                Y=Y, labeled_mask=labeled_mask, gt_labels=gt_labels,
                alpha=alpha, tol=tol, connectivity=connectivity,
                max_iter=max_iter, device=device, num_classes=num_classes
            )
            results.append(result)

            if result['mean_dice'] > best_dice:
                best_dice = result['mean_dice']
                best_result = result

            pbar.set_postfix({
                'best_dice': f"{best_dice:.4f}",
                'current': f"α={alpha:.2f}, conn={connectivity}"
            })
            pbar.update(1)

    print(f"\n✓ Best Dice: {best_dice:.4f}")
    print(f"✓ Best config: α={best_result['alpha']}, tol={best_result['tol']:.0e}, conn={best_result['connectivity']}")
    print()

    # Save results
    grid_results = {
        'case_id': case_id,
        'n_seeds': int(n_seeds),
        'results': results,
    }

    with open(output_dir / "grid_results_full.json", 'w') as f:
        json.dump(grid_results, f, indent=2)

    # Print top 20
    sorted_results = sorted(results, key=lambda x: x['mean_dice'], reverse=True)[:20]
    print("TOP 20 CONFIGURATIONS:")
    print("-" * 80)
    for i, res in enumerate(sorted_results, 1):
        print(f"{i:2d}. α={res['alpha']:.2f}, tol={res['tol']:.0e}, conn={res['connectivity']} → "
              f"Dice={res['mean_dice']:.4f}, Acc={res['accuracy']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
