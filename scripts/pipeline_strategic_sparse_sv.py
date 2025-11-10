#!/usr/bin/env python3
"""
End-to-end pipeline for strategic sparse SV labeling + multi-k propagation.

Orchestrates:
1. Strategic seed sampling (max 1 per SV, FG borders, rare classes)
2. Multi-k label propagation (k-NN weighted voting)
3. Training directory creation (symlinks)

Usage:
    python3 scripts/pipeline_strategic_sparse_sv.py \
        --sv_dir /data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n12000_c0.05_s1.0_ras2_voted \
        --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
        --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json \
        --budget_ratio 0.001 \
        --k_values 1,3,5,7,10,15,20,25,30,50 \
        --output_dir runs/sv_sparse_prop_0p1pct_strategic \
        --seed 42

Outputs:
    - strategic_seeds/ - Sampled seeds and sparse SV labels
    - cases/<case_id>/ - Per-case propagated labels for all k
    - k_variants/k01/, k03/, ... - Training directories (symlinks)
    - summary_stats.json, propagation_summary.json - Statistics
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, desc: str):
    """Run subprocess command with error handling."""
    print(f"\n{'='*70}")
    print(f"{desc}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nERROR: {desc} failed with code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Strategic sparse SV pipeline")
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
    parser.add_argument("--k_values", type=str, default="1,3,5,7,10,15,20,25,30,50",
                       help="Comma-separated k values (default: 1,3,5,7,10,15,20,25,30,50)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for all results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--skip_sampling", action="store_true",
                       help="Skip sampling step (use existing seeds)")
    parser.add_argument("--skip_propagation", action="store_true",
                       help="Skip propagation step (use existing propagated labels)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    seeds_dir = output_dir / "strategic_seeds"

    # Step 1: Strategic seed sampling
    if not args.skip_sampling:
        cmd_sample = [
            "python3", "scripts/sample_strategic_sv_seeds.py",
            "--sv_dir", args.sv_dir,
            "--data_root", args.data_root,
            "--split_cfg", args.split_cfg,
            "--split", args.split,
            "--budget_ratio", str(args.budget_ratio),
            "--class_weights", args.class_weights,
            "--output_dir", str(seeds_dir),
            "--seed", str(args.seed),
        ]

        run_command(cmd_sample, "STEP 1: Strategic Seed Sampling")
    else:
        print(f"\nSkipping sampling (using existing seeds in {seeds_dir})")

    # Check that seeds exist
    if not seeds_dir.exists():
        print(f"ERROR: Seeds directory not found: {seeds_dir}")
        print("Run without --skip_sampling first")
        sys.exit(1)

    # Step 2: Multi-k propagation
    if not args.skip_propagation:
        cmd_propagate = [
            "python3", "scripts/propagate_sv_labels_multi_k.py",
            "--sv_dir", args.sv_dir,
            "--seeds_dir", str(seeds_dir),
            "--k_values", args.k_values,
            "--output_dir", str(output_dir),
            "--seed", str(args.seed),
        ]

        run_command(cmd_propagate, "STEP 2: Multi-k Label Propagation")
    else:
        print(f"\nSkipping propagation (using existing labels in {output_dir})")

    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"  Output directory: {output_dir}")
    print(f"  Seeds: {seeds_dir}/")
    print(f"  Propagated labels: {output_dir}/cases/")
    print(f"  Training dirs: {output_dir}/k_variants/k{{01,03,05,...}}/")
    print("\nNext steps:")
    print(f"  1. Review statistics: cat {output_dir}/summary_stats.json")
    print(f"  2. Review propagation: cat {output_dir}/propagation_summary.json")
    print(f"  3. Train models: bash scripts/train_all_k_variants.sh {output_dir}/k_variants")
    print("="*70)


if __name__ == "__main__":
    main()
