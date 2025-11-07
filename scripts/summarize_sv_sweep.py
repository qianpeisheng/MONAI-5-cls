#!/usr/bin/env python3
"""
Summarize supervoxel sweep results from evaluation folders.

Reads metrics/summary.json from all *_eval folders, generates plots,
and creates a markdown report with findings.
"""

import json
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

def parse_folder_name(folder_name):
    """
    Parse folder name to extract parameters.
    Format: sv_fullgt_[MODE]_n[N_SEGMENTS]_c[COMPACTNESS]_s[SIGMA]_ras2_voted_eval
    """
    # Remove _eval suffix
    name = folder_name.replace('_eval', '')

    # Extract parameters using regex
    mode_match = re.search(r'sv_fullgt_([^_]+(?:-[^_]+)*)_n', name)
    n_seg_match = re.search(r'_n(\d+)_', name)
    comp_match = re.search(r'_c([0-9.]+)_', name)
    sigma_match = re.search(r'_s([0-9.]+)_', name)

    return {
        'mode': mode_match.group(1) if mode_match else None,
        'n_segments': int(n_seg_match.group(1)) if n_seg_match else None,
        'compactness': float(comp_match.group(1)) if comp_match else None,
        'sigma': float(sigma_match.group(1)) if sigma_match else None,
    }

def collect_metrics(sweep_root):
    """Collect metrics from all *_eval folders."""
    sweep_path = Path(sweep_root)

    results = []

    # Find all _eval folders
    eval_folders = sorted([f for f in sweep_path.iterdir() if f.is_dir() and f.name.endswith('_eval')])

    print(f"Found {len(eval_folders)} evaluation folders")

    for eval_folder in eval_folders:
        summary_file = eval_folder / 'metrics' / 'summary.json'

        if not summary_file.exists():
            print(f"Warning: {summary_file} not found, skipping")
            continue

        # Parse folder name
        params = parse_folder_name(eval_folder.name)

        # Load metrics
        with open(summary_file, 'r') as f:
            data = json.load(f)

        # Extract key metrics
        row = {
            **params,
            'dice_avg': data['average']['dice'],
            'iou_avg': data['average']['iou'],
            'dice_0': data['per_class']['0']['dice'],
            'dice_1': data['per_class']['1']['dice'],
            'dice_2': data['per_class']['2']['dice'],
            'dice_3': data['per_class']['3']['dice'],
            'dice_4': data['per_class']['4']['dice'],
        }

        # Add SV diagnostics if available
        if 'sv_diagnostics' in data:
            sv = data['sv_diagnostics']
            row.update({
                'sv_count': sv.get('sv_count'),
                'sv_mean_purity': sv.get('sv_mean_purity'),
                'sv_mean_entropy_norm': sv.get('sv_mean_entropy_norm'),
                'sv_vox_weighted_mean_purity': sv.get('sv_vox_weighted_mean_purity'),
                'sv_vox_weighted_mean_entropy_norm': sv.get('sv_vox_weighted_mean_entropy_norm'),
            })

        results.append(row)

    return pd.DataFrame(results)

def generate_plots(df, output_dir):
    """Generate visualization plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {'slic': '#1f77b4', 'slic-grad-mag': '#ff7f0e', 'slic-grad-vec': '#2ca02c'}

    # Plot 1: Dice vs n_segments
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in df['mode'].unique():
        mode_data = df[df['mode'] == mode].sort_values('n_segments')
        ax.plot(mode_data['n_segments'], mode_data['dice_avg'],
                marker='o', label=mode, linewidth=2, markersize=6, color=colors.get(mode))

    ax.set_xlabel('Number of Supervoxels', fontsize=12)
    ax.set_ylabel('Average Dice Score', fontsize=12)
    ax.set_title('Supervoxel Segmentation Quality: Dice vs Number of Segments', fontsize=14, fontweight='bold')
    ax.legend(title='SLIC Mode', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'dice_vs_nsegments.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: IoU vs n_segments
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in df['mode'].unique():
        mode_data = df[df['mode'] == mode].sort_values('n_segments')
        ax.plot(mode_data['n_segments'], mode_data['iou_avg'],
                marker='o', label=mode, linewidth=2, markersize=6, color=colors.get(mode))

    ax.set_xlabel('Number of Supervoxels', fontsize=12)
    ax.set_ylabel('Average IoU', fontsize=12)
    ax.set_title('Supervoxel Segmentation Quality: IoU vs Number of Segments', fontsize=14, fontweight='bold')
    ax.legend(title='SLIC Mode', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'iou_vs_nsegments.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Per-class Dice comparison (using best performing mode for clarity)
    # Find best overall mode
    best_mode = df.groupby('mode')['dice_avg'].mean().idxmax()
    mode_data = df[df['mode'] == best_mode].sort_values('n_segments')

    fig, ax = plt.subplots(figsize=(12, 6))
    class_labels = ['Class 0 (BG)', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    for i, cls in enumerate(['dice_0', 'dice_1', 'dice_2', 'dice_3', 'dice_4']):
        ax.plot(mode_data['n_segments'], mode_data[cls],
                marker='o', label=class_labels[i], linewidth=2, markersize=6)

    ax.set_xlabel('Number of Supervoxels', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title(f'Per-Class Dice Scores ({best_mode} mode)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'perclass_dice.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: SV quality metrics (purity and entropy)
    if 'sv_mean_purity' in df.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Purity
        for mode in df['mode'].unique():
            mode_data = df[df['mode'] == mode].sort_values('n_segments')
            ax1.plot(mode_data['n_segments'], mode_data['sv_mean_purity'],
                    marker='o', label=mode, linewidth=2, markersize=6, color=colors.get(mode))

        ax1.set_xlabel('Number of Supervoxels', fontsize=12)
        ax1.set_ylabel('Mean Purity', fontsize=12)
        ax1.set_title('Supervoxel Quality: Mean Purity', fontsize=14, fontweight='bold')
        ax1.legend(title='SLIC Mode', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.9, 1.0])

        # Entropy
        for mode in df['mode'].unique():
            mode_data = df[df['mode'] == mode].sort_values('n_segments')
            ax2.plot(mode_data['n_segments'], mode_data['sv_mean_entropy_norm'],
                    marker='o', label=mode, linewidth=2, markersize=6, color=colors.get(mode))

        ax2.set_xlabel('Number of Supervoxels', fontsize=12)
        ax2.set_ylabel('Mean Normalized Entropy', fontsize=12)
        ax2.set_title('Supervoxel Quality: Mean Entropy', fontsize=14, fontweight='bold')
        ax2.legend(title='SLIC Mode', fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path / 'sv_quality_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Generated 4 plots in {output_path}")

def generate_report(df, output_file):
    """Generate markdown summary report."""

    # Find best configurations
    best_overall = df.loc[df['dice_avg'].idxmax()]
    best_by_mode = df.loc[df.groupby('mode')['dice_avg'].idxmax()]

    # Calculate mode averages
    mode_summary = df.groupby('mode').agg({
        'dice_avg': ['mean', 'max', 'min'],
        'iou_avg': ['mean', 'max', 'min'],
        'sv_mean_purity': 'mean',
        'sv_mean_entropy_norm': 'mean'
    }).round(4)

    report = f"""# Supervoxel Sweep Summary Report

## Overview

This report summarizes the results of supervoxel parameter sweep across 30 configurations:
- **Modes tested**: {', '.join(df['mode'].unique())}
- **n_segments range**: {df['n_segments'].min()} to {df['n_segments'].max()}
- **Fixed parameters**: compactness={df['compactness'].iloc[0]}, sigma={df['sigma'].iloc[0]}
- **Total evaluations**: {len(df)} configurations

## Best Configuration

**Overall best performance:**
- **Mode**: {best_overall['mode']}
- **n_segments**: {best_overall['n_segments']}
- **Dice score**: {best_overall['dice_avg']:.4f}
- **IoU score**: {best_overall['iou_avg']:.4f}
- **SV purity**: {best_overall.get('sv_mean_purity', 'N/A')}
- **SV entropy**: {best_overall.get('sv_mean_entropy_norm', 'N/A')}

## Best Configuration per Mode

| Mode | n_segments | Dice | IoU | SV Purity | SV Entropy |
|------|-----------|------|-----|-----------|------------|
"""

    for mode in best_by_mode.index:
        row = best_by_mode.loc[mode]
        report += f"| {mode} | {row['n_segments']} | {row['dice_avg']:.4f} | {row['iou_avg']:.4f} | "
        report += f"{row.get('sv_mean_purity', 'N/A') if pd.notna(row.get('sv_mean_purity')) else 'N/A'} | "
        report += f"{row.get('sv_mean_entropy_norm', 'N/A') if pd.notna(row.get('sv_mean_entropy_norm')) else 'N/A'} |\n"

    report += f"""
## Mode Comparison Summary

Average performance across all n_segments values:

| Mode | Avg Dice | Max Dice | Min Dice | Avg IoU | Max IoU | Min IoU | Avg Purity | Avg Entropy |
|------|----------|----------|----------|---------|---------|---------|------------|-------------|
"""

    for mode in df['mode'].unique():
        mode_data = df[df['mode'] == mode]
        report += f"| {mode} | {mode_data['dice_avg'].mean():.4f} | {mode_data['dice_avg'].max():.4f} | "
        report += f"{mode_data['dice_avg'].min():.4f} | {mode_data['iou_avg'].mean():.4f} | "
        report += f"{mode_data['iou_avg'].max():.4f} | {mode_data['iou_avg'].min():.4f} | "
        purity_avg = mode_data['sv_mean_purity'].mean() if 'sv_mean_purity' in mode_data else None
        entropy_avg = mode_data['sv_mean_entropy_norm'].mean() if 'sv_mean_entropy_norm' in mode_data else None
        report += f"{purity_avg:.4f} | {entropy_avg:.4f} |\n" if purity_avg else "N/A | N/A |\n"

    report += """
## Key Findings

### 1. Impact of n_segments

"""
    # Analyze trend
    for mode in df['mode'].unique():
        mode_data = df[df['mode'] == mode].sort_values('n_segments')
        dice_trend = "increases" if mode_data['dice_avg'].iloc[-1] > mode_data['dice_avg'].iloc[0] else "decreases"
        dice_change = abs(mode_data['dice_avg'].iloc[-1] - mode_data['dice_avg'].iloc[0])
        report += f"- **{mode}**: Dice score {dice_trend} by {dice_change:.4f} from {mode_data['n_segments'].min()} to {mode_data['n_segments'].max()} segments\n"

    report += """
### 2. Mode Comparison

"""
    best_mode = df.groupby('mode')['dice_avg'].mean().idxmax()
    worst_mode = df.groupby('mode')['dice_avg'].mean().idxmin()
    report += f"- **Best performing mode**: {best_mode} (avg Dice: {df[df['mode']==best_mode]['dice_avg'].mean():.4f})\n"
    report += f"- **Least performing mode**: {worst_mode} (avg Dice: {df[df['mode']==worst_mode]['dice_avg'].mean():.4f})\n"

    report += """
### 3. Per-Class Performance

Average Dice scores across all configurations:

"""
    for i in range(5):
        col = f'dice_{i}'
        avg_dice = df[col].mean()
        report += f"- **Class {i}**: {avg_dice:.4f}\n"

    report += """
### 4. Supervoxel Quality

"""
    if 'sv_mean_purity' in df.columns:
        report += f"- Average SV purity across all configs: {df['sv_mean_purity'].mean():.4f}\n"
        report += f"- Average SV normalized entropy: {df['sv_mean_entropy_norm'].mean():.4f}\n"
        report += f"- Purity range: [{df['sv_mean_purity'].min():.4f}, {df['sv_mean_purity'].max():.4f}]\n"
        report += f"- Entropy range: [{df['sv_mean_entropy_norm'].min():.4f}, {df['sv_mean_entropy_norm'].max():.4f}]\n"

    report += """
## Recommendations

"""
    report += f"1. **For best segmentation quality**, use **{best_overall['mode']}** mode with **{best_overall['n_segments']} segments**\n"

    # Find sweet spot (good performance, reasonable SV count)
    mid_range = df[(df['n_segments'] >= 8000) & (df['n_segments'] <= 12000)]
    if len(mid_range) > 0:
        sweet_spot = mid_range.loc[mid_range['dice_avg'].idxmax()]
        report += f"2. **For balanced performance/complexity**, consider **{sweet_spot['mode']}** with **{sweet_spot['n_segments']} segments** (Dice: {sweet_spot['dice_avg']:.4f})\n"

    report += f"3. **Minimum recommended segments**: Avoid configurations below {df['n_segments'].min()+2000} segments for adequate quality\n"

    report += """
## Visualizations

![Dice vs n_segments](sv_sweep_ras2_plots/dice_vs_nsegments.png)

![IoU vs n_segments](sv_sweep_ras2_plots/iou_vs_nsegments.png)

![Per-class Dice](sv_sweep_ras2_plots/perclass_dice.png)

![SV Quality Metrics](sv_sweep_ras2_plots/sv_quality_metrics.png)

## Full Results Table

"""

    # Add full table sorted by dice
    full_table = df.sort_values('dice_avg', ascending=False)[
        ['mode', 'n_segments', 'dice_avg', 'iou_avg', 'sv_mean_purity', 'sv_mean_entropy_norm']
    ].copy()

    # Manually format markdown table
    report += "| Mode | n_segments | Dice | IoU | SV Purity | SV Entropy |\n"
    report += "|------|-----------|------|-----|-----------|------------|\n"
    for _, row in full_table.iterrows():
        purity = f"{row['sv_mean_purity']:.4f}" if pd.notna(row.get('sv_mean_purity')) else 'N/A'
        entropy = f"{row['sv_mean_entropy_norm']:.4f}" if pd.notna(row.get('sv_mean_entropy_norm')) else 'N/A'
        report += f"| {row['mode']} | {row['n_segments']} | {row['dice_avg']:.4f} | {row['iou_avg']:.4f} | {purity} | {entropy} |\n"

    report += """

---
*Report generated by summarize_sv_sweep.py*
"""

    # Write report
    with open(output_file, 'w') as f:
        f.write(report)

    print(f"Generated report: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Summarize supervoxel sweep results')
    parser.add_argument('--sweep-root', type=str,
                       default='/data3/wp5/monai-sv-sweeps',
                       help='Root directory containing *_eval folders')
    parser.add_argument('--output-dir', type=str,
                       default='runs/sv_sweep_ras2_plots',
                       help='Directory to save plots')
    parser.add_argument('--report', type=str,
                       default='runs/sv_sweep_ras2_summary.md',
                       help='Output markdown report file')

    args = parser.parse_args()

    print("Collecting metrics from evaluation folders...")
    df = collect_metrics(args.sweep_root)

    if df.empty:
        print("No data collected. Check sweep root path.")
        return

    print(f"\nCollected {len(df)} configurations")
    print(f"Modes: {df['mode'].unique()}")
    print(f"n_segments range: {df['n_segments'].min()} - {df['n_segments'].max()}")

    print("\nGenerating plots...")
    generate_plots(df, args.output_dir)

    print("\nGenerating report...")
    generate_report(df, args.report)

    print("\n" + "="*60)
    print("Summary complete!")
    print(f"  Plots: {args.output_dir}/")
    print(f"  Report: {args.report}")
    print("="*60)

if __name__ == '__main__':
    main()
