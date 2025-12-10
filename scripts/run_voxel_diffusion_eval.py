#!/usr/bin/env python3
"""
Run Zhou-style voxel diffusion on a volume with sparse seed labels.

This script applies the voxel-level diffusion directly on the 3D volume
(not on supervoxels) and evaluates the results against ground truth.

Usage:
    python scripts/run_voxel_diffusion_eval.py \
        --case_id SN13B0_I17_3D_B1_1B250409 \
        --seed_file tmp/test_seeds/SN13B0_I17_3D_B1_1B250409_0p5pct_seed_labels.npy \
        --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
        --output_dir tmp/test_voxel_diffusion_zhou_0p5pct \
        --alpha 0.99 \
        --max_iter 500 \
        --connectivity 6
"""

import argparse
import json
import sys
import time
from pathlib import Path

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

    # Confusion matrix
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(num_classes):
        for j in range(num_classes):
            conf_matrix[i, j] = np.sum((gt == i) & (pred == j))

    metrics['confusion_matrix'] = conf_matrix.tolist()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run voxel-level Zhou diffusion")
    parser.add_argument("--case_id", type=str, required=True, help="Case ID")
    parser.add_argument("--seed_file", type=str, required=True, help="Path to seed labels .npy file")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory for data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--alpha", type=float, default=0.99, help="Diffusion alpha parameter")
    parser.add_argument("--max_iter", type=int, default=500, help="Max diffusion iterations")
    parser.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance")
    parser.add_argument("--connectivity", type=int, default=6, choices=[6, 26], help="Neighborhood connectivity")
    parser.add_argument("--num_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("VOXEL-LEVEL ZHOU DIFFUSION EVALUATION")
    print("="*80)
    print(f"Case ID: {args.case_id}")
    print(f"Seed file: {args.seed_file}")
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Alpha: {args.alpha}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Tolerance: {args.tol}")
    print(f"Connectivity: {args.connectivity}")
    print(f"Device: {args.device}")
    print()

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Load ground truth
    print("Loading ground truth...")
    data_root = Path(args.data_root)
    image_path = data_root / "data" / f"{args.case_id}_image.nii"
    label_path = data_root / "data" / f"{args.case_id}_label.nii"

    if not image_path.exists():
        print(f"ERROR: Image not found at {image_path}")
        return 1

    if not label_path.exists():
        print(f"ERROR: Label not found at {label_path}")
        return 1

    # Load with MONAI transforms (same as training pipeline)
    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ])

    data_dict = transforms({
        "image": str(image_path),
        "label": str(label_path),
    })

    # Extract numpy arrays
    gt_labels = data_dict["label"].squeeze().numpy().astype(np.int64)
    print(f"  Ground truth shape: {gt_labels.shape}")
    print(f"  Ground truth range: [{gt_labels.min()}, {gt_labels.max()}]")
    print(f"  Ground truth class distribution:")
    for cls in range(args.num_classes):
        count = np.sum(gt_labels == cls)
        print(f"    Class {cls}: {count:,} voxels ({count/gt_labels.size*100:.2f}%)")

    # Load seed labels
    print(f"\nLoading seed labels from {args.seed_file}...")
    seed_file = Path(args.seed_file)
    if not seed_file.exists():
        print(f"ERROR: Seed file not found at {seed_file}")
        return 1

    seed_labels = np.load(seed_file)
    print(f"  Seed labels shape: {seed_labels.shape}")

    # Check shape consistency
    if seed_labels.shape != gt_labels.shape:
        print(f"ERROR: Shape mismatch! Seeds: {seed_labels.shape}, GT: {gt_labels.shape}")
        return 1

    # Analyze seed labels
    # Handle files with -1 for unlabeled (include class 0 as labeled)
    seed_mask = seed_labels >= 0
    n_seeds = np.count_nonzero(seed_mask)
    print(f"  Number of labeled voxels: {n_seeds:,} ({n_seeds/seed_labels.size*100:.4f}%)")
    print(f"  Seed label class distribution:")
    for cls in range(args.num_classes):
        count = np.sum(seed_labels == cls)
        if count > 0:
            print(f"    Class {cls}: {count:,} seeds")

    # Check seed accuracy
    if n_seeds > 0:
        seed_correct = np.sum(seed_labels[seed_mask] == gt_labels[seed_mask])
        seed_accuracy = seed_correct / n_seeds
        print(f"  Seed accuracy: {seed_accuracy:.4f} ({seed_correct}/{n_seeds})")

    # Prepare input for diffusion
    print(f"\nPreparing input for Zhou diffusion...")
    D, H, W = gt_labels.shape

    # Create one-hot encoded Y tensor (1, C, D, H, W)
    Y = torch.zeros((1, args.num_classes, D, H, W), dtype=torch.float32)

    # Fill in seed labels (one-hot encoding)
    for cls in range(args.num_classes):
        class_mask = (seed_labels == cls)
        Y[0, cls, class_mask] = 1.0

    # Create labeled mask (1, 1, D, H, W)
    labeled_mask = torch.from_numpy(seed_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    print(f"  Y shape: {Y.shape}")
    print(f"  Labeled mask shape: {labeled_mask.shape}")
    print(f"  Y memory: {Y.numel() * 4 / (1024**2):.1f} MB")

    # Run Zhou diffusion
    print(f"\nRunning Zhou diffusion...")
    print(f"  This may take a while for large volumes...")
    start_time = time.time()

    F = diffuse_labels_3d(
        Y=Y,
        labeled_mask=labeled_mask,
        alpha=args.alpha,
        max_iter=args.max_iter,
        tol=args.tol,
        connectivity=args.connectivity,
        device=args.device,
    )

    elapsed_time = time.time() - start_time
    print(f"  ✓ Diffusion completed in {elapsed_time:.2f} seconds")

    # Extract predictions
    print(f"\nExtracting predictions...")
    F_cpu = F.cpu()
    pred_labels = torch.argmax(F_cpu, dim=1).squeeze().numpy().astype(np.int64)

    print(f"  Prediction shape: {pred_labels.shape}")
    print(f"  Prediction range: [{pred_labels.min()}, {pred_labels.max()}]")

    # Compute metrics
    print(f"\nEvaluating predictions against ground truth...")
    metrics = compute_metrics(pred_labels, gt_labels, args.num_classes)

    print(f"\nRESULTS:")
    print(f"  Overall accuracy: {metrics['accuracy']:.4f}")
    print(f"  Mean Dice score: {metrics['mean_dice']:.4f}")
    print(f"  Per-class Dice scores:")
    for cls in range(args.num_classes):
        dice = metrics['dice_scores'][f'class_{cls}']
        print(f"    Class {cls}: {dice:.4f}")

    # Save results
    print(f"\nSaving results to {output_dir}...")

    # Save predictions
    pred_file = output_dir / f"{args.case_id}_voxel_diffusion_pred.npy"
    np.save(pred_file, pred_labels)
    print(f"  Saved predictions to {pred_file}")

    # Save scores
    scores_file = output_dir / f"{args.case_id}_voxel_diffusion_scores.npy"
    np.save(scores_file, F_cpu.squeeze().numpy())
    print(f"  Saved scores to {scores_file}")

    # Save metrics
    results = {
        'case_id': args.case_id,
        'method': 'voxel_diffusion_zhou',
        'alpha': args.alpha,
        'max_iter': args.max_iter,
        'tol': args.tol,
        'connectivity': args.connectivity,
        'n_seeds': int(n_seeds),
        'seed_percentage': float(n_seeds / seed_labels.size * 100),
        'seed_accuracy': float(seed_accuracy) if n_seeds > 0 else 0.0,
        'runtime_seconds': float(elapsed_time),
        'volume_shape': list(gt_labels.shape),
        'metrics': metrics,
    }

    results_file = output_dir / f"{args.case_id}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results to {results_file}")

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
