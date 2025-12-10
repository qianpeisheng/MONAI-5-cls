#!/usr/bin/env python3
"""
Batch Zhou diffusion (voxel-level) for multiple training cases.

Applies voxel-level Zhou label propagation to all training cases using optimal parameters.

Usage:
    python3 scripts/propagate_zhou_voxel_multi_case.py \
        --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
        --seeds_dir runs/zhou_voxel_seeds_0p1pct \
        --output_dir runs/zhou_voxel_0p1pct_a0.95_c26 \
        --alpha 0.95 \
        --connectivity 26 \
        --tol 1e-4 \
        --max_iter 500 \
        --device cuda:1

Outputs (per case):
    - cases/<case_id>/propagated_labels.npy - Dense voxel predictions
    - cases/<case_id>/propagation_meta.json - Statistics + metrics
    - labels/<case_id>_labels.npy - Symlinks for training
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

# Add parent to path
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


def propagate_case(
    case_id: str,
    data_root: Path,
    seeds_dir: Path,
    output_dir: Path,
    alpha: float,
    connectivity: int,
    tol: float,
    max_iter: int,
    device: str,
    num_classes: int = 5,
    evaluate: bool = True,
):
    """Process one case: load seeds, run Zhou diffusion, save predictions."""

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
    gt_labels = data_dict["label"].squeeze().numpy().astype(np.int64)

    # Load seed labels
    seed_file = seeds_dir / "seeds" / f"{case_id}_seed_labels.npy"
    if not seed_file.exists():
        print(f"WARNING: Seeds not found for {case_id}, skipping")
        return None

    seed_labels = np.load(seed_file)
    seed_mask = seed_labels >= 0
    n_seeds = np.count_nonzero(seed_mask)

    # Prepare input for Zhou diffusion
    D, H, W = gt_labels.shape
    Y = torch.zeros((1, num_classes, D, H, W), dtype=torch.float32)
    for cls in range(num_classes):
        class_mask = (seed_labels == cls)
        Y[0, cls, class_mask] = 1.0

    labeled_mask = torch.from_numpy(seed_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    # Run Zhou diffusion
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
    pred_labels = torch.argmax(F_cpu, dim=1).squeeze().numpy().astype(np.int16)

    # Compute metrics (if evaluating)
    metrics = None
    if evaluate:
        metrics = compute_metrics(pred_labels, gt_labels, num_classes)

    # Save outputs
    case_output_dir = output_dir / "cases" / case_id
    case_output_dir.mkdir(parents=True, exist_ok=True)

    # Save propagated labels
    np.save(case_output_dir / "propagated_labels.npy", pred_labels)

    # Save metadata
    meta = {
        "case_id": case_id,
        "n_seeds": int(n_seeds),
        "n_total_voxels": int(gt_labels.size),
        "seed_percentage": float(n_seeds / gt_labels.size * 100),
        "alpha": alpha,
        "connectivity": connectivity,
        "tolerance": tol,
        "max_iter": max_iter,
        "runtime_seconds": runtime,
        "volume_shape": list(gt_labels.shape),
    }

    if metrics:
        meta.update({
            "mean_dice": metrics['mean_dice'],
            "accuracy": metrics['accuracy'],
            "dice_scores": metrics['dice_scores'],
        })

    with open(case_output_dir / "propagation_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


def create_training_symlinks(output_dir: Path, cases: list):
    """Create flat labels directory with symlinks for training."""
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    for case_id in cases:
        src = output_dir / "cases" / case_id / "propagated_labels.npy"

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
    parser = argparse.ArgumentParser(description="Batch Zhou diffusion propagation")
    parser.add_argument("--data_root", type=str,
                       default="/data3/wp5/wp5-code/dataloaders/wp5-dataset",
                       help="Data root directory")
    parser.add_argument("--seeds_dir", type=str, required=True,
                       help="Directory with seed labels from sampling step")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for propagated labels")
    parser.add_argument("--alpha", type=float, default=0.95,
                       help="Diffusion parameter (default: 0.95)")
    parser.add_argument("--connectivity", type=int, default=26,
                       help="Neighborhood connectivity (default: 26)")
    parser.add_argument("--tol", type=float, default=1e-4,
                       help="Convergence tolerance (default: 1e-4)")
    parser.add_argument("--max_iter", type=int, default=500,
                       help="Max iterations (default: 500)")
    parser.add_argument("--num_classes", type=int, default=5,
                       help="Number of classes (default: 5)")
    parser.add_argument("--device", type=str, default="cuda:1",
                       help="Device (default: cuda:1)")
    parser.add_argument("--no_eval", action="store_true",
                       help="Skip evaluation against ground truth")

    args = parser.parse_args()

    # Check device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    data_root = Path(args.data_root)
    seeds_dir = Path(args.seeds_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("BATCH ZHOU DIFFUSION (VOXEL-LEVEL)")
    print("="*80)
    print(f"Parameters: α={args.alpha}, connectivity={args.connectivity}, tol={args.tol:.0e}")
    print(f"Device: {args.device}")
    print(f"Seeds dir: {seeds_dir}")
    print(f"Output dir: {output_dir}")
    print()

    # Find all cases from seeds directory
    seed_files = list((seeds_dir / "seeds").glob("*_seed_labels.npy"))
    cases = [f.stem.replace("_seed_labels", "") for f in seed_files]

    print(f"Found {len(cases)} cases with seed labels")
    print()

    # Process all cases
    all_meta = []
    for case_id in tqdm(cases, desc="Processing cases"):
        meta = propagate_case(
            case_id, data_root, seeds_dir, output_dir,
            args.alpha, args.connectivity, args.tol, args.max_iter,
            args.device, args.num_classes, evaluate=not args.no_eval
        )
        if meta:
            all_meta.append(meta)

    # Create training symlinks
    if all_meta:
        create_training_symlinks(output_dir, [m["case_id"] for m in all_meta])

        # Summary
        avg_seeds = np.mean([m["n_seeds"] for m in all_meta])
        avg_runtime = np.mean([m["runtime_seconds"] for m in all_meta])

        summary = {
            "n_cases": len(all_meta),
            "alpha": args.alpha,
            "connectivity": args.connectivity,
            "tolerance": args.tol,
            "max_iter": args.max_iter,
            "avg_seeds_per_case": float(avg_seeds),
            "avg_runtime_seconds": float(avg_runtime),
        }

        if not args.no_eval:
            avg_dice = np.mean([m["mean_dice"] for m in all_meta])
            avg_acc = np.mean([m["accuracy"] for m in all_meta])
            summary["avg_mean_dice"] = float(avg_dice)
            summary["avg_accuracy"] = float(avg_acc)

        with open(output_dir / "propagation_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*80)
        print("ZHOU DIFFUSION COMPLETE!")
        print("="*80)
        print(f"  Processed: {len(all_meta)} cases")
        print(f"  Parameters: α={args.alpha}, conn={args.connectivity}")
        print(f"  Avg runtime: {avg_runtime:.2f}s per case")
        if not args.no_eval:
            print(f"  Avg Dice: {avg_dice:.4f}")
            print(f"  Avg Accuracy: {avg_acc:.4f}")
        print(f"  Output: {output_dir}")
        print(f"  Training dir: {output_dir}/labels/")
        print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
