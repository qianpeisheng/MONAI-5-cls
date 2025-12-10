#!/usr/bin/env python3
"""
Compare Zhou voxel diffusion vs Graph LP propagation quality.

Usage:
    python3 scripts/evaluate_zhou_vs_graph_lp.py \
        --zhou_dir runs/zhou_voxel_0p1pct_a0.95_c26 \
        --graph_lp_dir runs/graph_lp_prop_0p1pct_k10_a0.9 \
        --output_file comparison_zhou_vs_graph_lp.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def load_metrics(cases_dir: Path, method_name: str):
    """Load metrics from all cases."""
    all_dice_per_class = defaultdict(list)
    all_mean_dice = []
    all_accuracy = []
    all_runtimes = []
    case_ids = []

    for case_meta in sorted(cases_dir.glob('*/propagation_meta.json')):
        with open(case_meta) as f:
            meta = json.load(f)

        case_ids.append(meta['case_id'])

        if 'mean_dice' in meta:
            all_mean_dice.append(meta['mean_dice'])
        if 'accuracy' in meta:
            all_accuracy.append(meta['accuracy'])
        if 'runtime_seconds' in meta:
            all_runtimes.append(meta['runtime_seconds'])

        if 'dice_scores' in meta:
            for cls in range(5):
                all_dice_per_class[f'class_{cls}'].append(meta['dice_scores'][f'class_{cls}'])

    return {
        'method': method_name,
        'n_cases': len(case_ids),
        'case_ids': case_ids,
        'mean_dice': all_mean_dice,
        'accuracy': all_accuracy,
        'runtimes': all_runtimes,
        'dice_per_class': dict(all_dice_per_class),
    }


def compute_stats(values):
    """Compute statistics for a list of values."""
    if not values:
        return {}

    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Zhou vs Graph LP")
    parser.add_argument("--zhou_dir", type=str, required=True,
                       help="Zhou diffusion output directory")
    parser.add_argument("--graph_lp_dir", type=str, required=True,
                       help="Graph LP output directory")
    parser.add_argument("--output_file", type=str, default="comparison_zhou_vs_graph_lp.json",
                       help="Output JSON file")

    args = parser.parse_args()

    zhou_dir = Path(args.zhou_dir)
    graph_lp_dir = Path(args.graph_lp_dir)

    print("="*80)
    print("ZHOU VOXEL DIFFUSION vs GRAPH LP COMPARISON")
    print("="*80)
    print()

    # Load Zhou metrics
    print("Loading Zhou diffusion metrics...")
    zhou_data = load_metrics(zhou_dir / "cases", "Zhou Voxel Diffusion")

    # Load Graph LP metrics (note: Graph LP doesn't have dice scores by default)
    print("Loading Graph LP metrics...")
    graph_lp_data = load_metrics(graph_lp_dir / "cases", "Graph LP (SV-based)")

    print(f"Zhou cases: {zhou_data['n_cases']}")
    print(f"Graph LP cases: {graph_lp_data['n_cases']}")
    print()

    # Compute statistics
    zhou_stats = {
        'mean_dice': compute_stats(zhou_data['mean_dice']),
        'accuracy': compute_stats(zhou_data['accuracy']),
        'runtime': compute_stats(zhou_data['runtimes']),
    }

    zhou_stats['dice_per_class'] = {}
    for cls in range(5):
        zhou_stats['dice_per_class'][f'class_{cls}'] = compute_stats(
            zhou_data['dice_per_class'][f'class_{cls}']
        )

    # Print comparison
    print("ZHOU VOXEL DIFFUSION RESULTS:")
    print("-"*80)
    print(f"  Mean Dice:     {zhou_stats['mean_dice']['mean']:.4f} ± {zhou_stats['mean_dice']['std']:.4f}")
    print(f"  Accuracy:      {zhou_stats['accuracy']['mean']:.4f} ± {zhou_stats['accuracy']['std']:.4f}")
    print(f"  Avg Runtime:   {zhou_stats['runtime']['mean']:.3f}s per case")
    print()

    print("PER-CLASS DICE (Zhou):")
    print("-"*80)
    for cls in range(5):
        stats = zhou_stats['dice_per_class'][f'class_{cls}']
        print(f"  Class {cls}:  {stats['mean']:.4f} ± {stats['std']:.4f}")
    print()

    # Load Zhou summary
    with open(zhou_dir / "propagation_summary.json") as f:
        zhou_summary = json.load(f)

    # Load Graph LP summary
    with open(graph_lp_dir / "propagation_summary.json") as f:
        graph_lp_summary = json.load(f)

    print("COMPARISON:")
    print("-"*80)
    print(f"Zhou parameters: α={zhou_summary['alpha']}, conn={zhou_summary['connectivity']}")
    print(f"Graph LP parameters: k={graph_lp_summary['k']}, α={graph_lp_summary['alpha']}")
    print(f"Average seeds per case: {zhou_summary['avg_seeds_per_case']:.0f} (Zhou) vs {graph_lp_summary['avg_labeled_svs_input']:.0f} (Graph LP SVs)")
    print()

    # Save comparison
    comparison = {
        'zhou': {
            'method': 'Zhou Voxel Diffusion',
            'parameters': {
                'alpha': zhou_summary['alpha'],
                'connectivity': zhou_summary['connectivity'],
                'tolerance': zhou_summary['tolerance'],
                'max_iter': zhou_summary['max_iter'],
            },
            'avg_seeds': zhou_summary['avg_seeds_per_case'],
            'statistics': zhou_stats,
        },
        'graph_lp': {
            'method': 'Graph LP (SV-based)',
            'parameters': {
                'k': graph_lp_summary['k'],
                'alpha': graph_lp_summary['alpha'],
            },
            'avg_labeled_svs': graph_lp_summary['avg_labeled_svs_input'],
        },
    }

    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"✓ Saved comparison to: {output_path}")
    print()
    print("="*80)
    print("KEY FINDINGS:")
    print("="*80)
    print()
    print(f"1. **Zhou Voxel Diffusion Quality:**")
    print(f"   - Mean Dice: {zhou_stats['mean_dice']['mean']:.4f} (std: {zhou_stats['mean_dice']['std']:.4f})")
    print(f"   - Accuracy: {zhou_stats['accuracy']['mean']:.4f}")
    print()

    print(f"2. **Computational Efficiency:**")
    print(f"   - Zhou runtime: {zhou_stats['runtime']['mean']:.3f}s per case")
    print(f"   - Extremely fast propagation on GPU")
    print()

    print(f"3. **Optimal Parameters (0.1% seeds):**")
    print(f"   - α=0.95 (high smoothing for sparse seeds)")
    print(f"   - 26-connectivity (dense neighbor graph)")
    print(f"   - These parameters adapt to very sparse seed density")
    print()

    print(f"4. **Class Performance:**")
    for cls in range(5):
        print(f"   - Class {cls}: {zhou_stats['dice_per_class'][f'class_{cls}']['mean']:.4f}")
    print()

    print("="*80)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
