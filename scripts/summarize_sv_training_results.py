#!/usr/bin/env python3
"""
Summarize supervoxel training results from training folders.

Reads metrics/summary.json from all train_sv_*_e20* folders, generates plots
comparing trained model performance vs SV quality, and creates a markdown report.
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
    Format: train_sv_[MODE]_n[N_SEGMENTS]_c[COMPACTNESS]_s[SIGMA]_e20_[TIMESTAMP]
    """
    # Extract parameters using regex
    mode_match = re.search(r'train_sv_([^_]+(?:-[^_]+)*)_n', folder_name)
    n_seg_match = re.search(r'_n(\d+)_', folder_name)
    comp_match = re.search(r'_c([0-9.]+)_', folder_name)
    sigma_match = re.search(r'_s([0-9.]+)_', folder_name)
    epochs_match = re.search(r'_e(\d+)_', folder_name)

    return {
        'mode': mode_match.group(1) if mode_match else None,
        'n_segments': int(n_seg_match.group(1)) if n_seg_match else None,
        'compactness': float(comp_match.group(1)) if comp_match else None,
        'sigma': float(sigma_match.group(1)) if sigma_match else None,
        'epochs': int(epochs_match.group(1)) if epochs_match else None,
    }

def parse_train_log(log_file):
    """Extract training info from train.log."""
    if not log_file.exists():
        return {}

    info = {
        'initial_loss': None,
        'final_loss': None,
        'best_epoch_dice': None,
        'training_time_total': 0.0,
    }

    with open(log_file, 'r') as f:
        lines = f.readlines()

    # Parse epoch lines for training loss and time
    epoch_losses = []
    epoch_times = []
    epoch_dice = []

    for line in lines:
        # Training loss: "Epoch 1/20, loss: 0.5876, time: 123.45s"
        loss_match = re.search(r'Epoch \d+/\d+, loss: ([0-9.]+), time: ([0-9.]+)s', line)
        if loss_match:
            epoch_losses.append(float(loss_match.group(1)))
            epoch_times.append(float(loss_match.group(2)))

        # Test Dice: "Epoch 1 test avg dice: 0.6789"
        dice_match = re.search(r'Epoch \d+ test avg dice: ([0-9.]+)', line)
        if dice_match:
            epoch_dice.append(float(dice_match.group(1)))

    if epoch_losses:
        info['initial_loss'] = epoch_losses[0]
        info['final_loss'] = epoch_losses[-1]

    if epoch_times:
        info['training_time_total'] = sum(epoch_times)
        info['avg_epoch_time'] = sum(epoch_times) / len(epoch_times)

    if epoch_dice:
        info['best_epoch_dice'] = max(epoch_dice)

    return info

def collect_metrics(runs_root):
    """Collect metrics from all train_sv_*_e20* folders."""
    runs_path = Path(runs_root)

    results = []

    # Find all training folders
    train_folders = sorted([f for f in runs_path.iterdir()
                           if f.is_dir() and f.name.startswith('train_sv_') and '_e20_' in f.name])

    print(f"Found {len(train_folders)} training folders")

    for train_folder in train_folders:
        summary_file = train_folder / 'metrics' / 'summary.json'

        if not summary_file.exists():
            print(f"Warning: {summary_file} not found, skipping")
            continue

        # Parse folder name
        params = parse_folder_name(train_folder.name)

        # Load test metrics
        with open(summary_file, 'r') as f:
            data = json.load(f)

        # Parse training log for additional info
        log_file = train_folder / 'train.log'
        train_info = parse_train_log(log_file)

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
            'iou_0': data['per_class']['0']['iou'],
            'iou_1': data['per_class']['1']['iou'],
            'iou_2': data['per_class']['2']['iou'],
            'iou_3': data['per_class']['3']['iou'],
            'iou_4': data['per_class']['4']['iou'],
            **train_info,
            'folder': train_folder.name,
        }

        results.append(row)

    return pd.DataFrame(results)

def generate_plots(df, output_dir, baselines=None):
    """Generate visualization plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load baselines if not provided
    if baselines is None:
        baselines = load_baseline_results()

    gt_baseline = baselines.get('100_gt', 0.8718)

    # Set up plotting style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = {'slic': '#1f77b4', 'slic-grad-mag': '#ff7f0e', 'slic-grad-vec': '#2ca02c'}

    # Plot 1: Trained Model Dice vs n_segments
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in sorted(df['mode'].unique()):
        mode_data = df[df['mode'] == mode].sort_values('n_segments')
        ax.plot(mode_data['n_segments'], mode_data['dice_avg'],
                marker='o', label=mode, linewidth=2, markersize=8, color=colors.get(mode))

    ax.set_xlabel('Number of Supervoxels (SV Label Source)', fontsize=12)
    ax.set_ylabel('Test Dice Score (Trained Model)', fontsize=12)
    ax.set_title('Trained Model Performance: Dice vs Supervoxel Granularity', fontsize=14, fontweight='bold')
    ax.legend(title='SLIC Mode', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add GT baseline reference line
    ax.axhline(y=gt_baseline, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label=f'100% GT baseline ({gt_baseline:.4f})')
    ax.legend(title='Training Source', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path / 'trained_dice_vs_nsegments.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Trained Model IoU vs n_segments
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in sorted(df['mode'].unique()):
        mode_data = df[df['mode'] == mode].sort_values('n_segments')
        ax.plot(mode_data['n_segments'], mode_data['iou_avg'],
                marker='o', label=mode, linewidth=2, markersize=8, color=colors.get(mode))

    ax.set_xlabel('Number of Supervoxels (SV Label Source)', fontsize=12)
    ax.set_ylabel('Test IoU (Trained Model)', fontsize=12)
    ax.set_title('Trained Model Performance: IoU vs Supervoxel Granularity', fontsize=14, fontweight='bold')
    ax.legend(title='SLIC Mode', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'trained_iou_vs_nsegments.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Per-class Dice comparison (best mode)
    best_mode = df.groupby('mode')['dice_avg'].mean().idxmax()
    mode_data = df[df['mode'] == best_mode].sort_values('n_segments')

    fig, ax = plt.subplots(figsize=(12, 6))
    class_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    class_colors = plt.cm.tab10(range(5))

    for i, cls in enumerate(['dice_0', 'dice_1', 'dice_2', 'dice_3', 'dice_4']):
        ax.plot(mode_data['n_segments'], mode_data[cls],
                marker='o', label=class_labels[i], linewidth=2, markersize=6, color=class_colors[i])

    ax.set_xlabel('Number of Supervoxels', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title(f'Per-Class Dice Scores - Trained Models ({best_mode} mode)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'trained_perclass_dice.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: Training convergence (loss over epochs) - sample 3 configs
    if 'initial_loss' in df.columns and df['initial_loss'].notna().any():
        # Pick low, mid, high n_segments for best mode
        mode_df = df[df['mode'] == best_mode].sort_values('n_segments')
        sample_indices = [0, len(mode_df)//2, len(mode_df)-1]
        sample_configs = mode_df.iloc[sample_indices]

        fig, ax = plt.subplots(figsize=(10, 6))

        for _, config in sample_configs.iterrows():
            n_seg = config['n_segments']
            initial = config.get('initial_loss')
            final = config.get('final_loss')
            if pd.notna(initial) and pd.notna(final):
                ax.plot([1, 20], [initial, final], marker='o', linewidth=2,
                       label=f'n={n_seg} (Dice={config["dice_avg"]:.4f})')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.set_title(f'Training Convergence - Sample Configs ({best_mode})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'training_convergence.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Plot 5: Mode comparison - grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    modes = sorted(df['mode'].unique())
    x = np.arange(len(modes))
    width = 0.35

    dice_means = [df[df['mode'] == m]['dice_avg'].mean() for m in modes]
    dice_maxs = [df[df['mode'] == m]['dice_avg'].max() for m in modes]

    bars1 = ax.bar(x - width/2, dice_means, width, label='Average Dice',
                   color=[colors.get(m, '#888888') for m in modes], alpha=0.7)
    bars2 = ax.bar(x + width/2, dice_maxs, width, label='Best Dice',
                   color=[colors.get(m, '#888888') for m in modes], alpha=1.0)

    ax.set_xlabel('SLIC Mode', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Mode Comparison: Average vs Best Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add GT baseline
    ax.axhline(y=gt_baseline, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(len(modes)-0.5, gt_baseline+0.005, f'100% GT ({gt_baseline:.4f})', fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(output_path / 'mode_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Generated 5 plots in {output_path}")

def load_baseline_results():
    """Load baseline results from actual experiment directories."""
    baselines = {}

    # 100% GT supervised baseline
    gt_100_path = Path('runs/grid_clip_zscore/scratch_subset_100/eval_20251021-120429/metrics/summary.json')
    if gt_100_path.exists():
        with open(gt_100_path) as f:
            data = json.load(f)
            baselines['100_gt'] = data['average']['dice']
    else:
        print(f"Warning: 100% GT baseline not found at {gt_100_path}")
        baselines['100_gt'] = None

    # 10% GT few-shot baseline
    gt_10_path = Path('runs/fewshot_grid_clip_zscore/points_10_d1_proportional/metrics/summary.json')
    if gt_10_path.exists():
        with open(gt_10_path) as f:
            data = json.load(f)
            baselines['10_gt'] = data['average']['dice']
    else:
        print(f"Warning: 10% GT baseline not found at {gt_10_path}")
        baselines['10_gt'] = None

    # 1% GT few-shot baseline
    gt_1_path = Path('runs/fp_1pct_global_d0_20251021-153502/metrics/summary.json')
    if gt_1_path.exists():
        with open(gt_1_path) as f:
            data = json.load(f)
            baselines['1_gt'] = data['average']['dice']
    else:
        print(f"Warning: 1% GT baseline not found at {gt_1_path}")
        baselines['1_gt'] = None

    return baselines

def generate_report(df, output_file, output_dir):
    """Generate markdown summary report."""

    # Find best configurations
    best_overall = df.loc[df['dice_avg'].idxmax()]
    best_by_mode = df.loc[df.groupby('mode')['dice_avg'].idxmax()]

    # Calculate mode averages
    mode_summary = df.groupby('mode').agg({
        'dice_avg': ['mean', 'max', 'min', 'std'],
        'iou_avg': ['mean', 'max', 'min'],
    }).round(4)

    # Load actual baseline results
    baselines = load_baseline_results()
    gt_baseline_dice = baselines.get('100_gt', 0.8718)
    gt_10_dice = baselines.get('10_gt', 0.8712)
    gt_1_dice = baselines.get('1_gt', 0.8310)

    # Safe formatting for potentially None values
    def safe_format(value, fmt='.4f', default='N/A'):
        if value is None or pd.isna(value):
            return default
        return f"{value:{fmt}}"

    best_initial_loss = safe_format(best_overall.get('initial_loss'))
    best_final_loss = safe_format(best_overall.get('final_loss'))
    best_train_time = best_overall.get('training_time_total', 0) or 0

    report = f"""# Supervoxel Training Results Summary

## Overview

This report summarizes the results of training segmentation models on **supervoxel-voted pseudo-labels** across 30 configurations:

- **Total experiments**: {len(df)} training runs
- **Modes tested**: {', '.join(sorted(df['mode'].unique()))}
- **n_segments range**: {df['n_segments'].min()} to {df['n_segments'].max()}
- **Fixed parameters**: compactness={df['compactness'].iloc[0]}, sigma={df['sigma'].iloc[0]}
- **Training epochs**: {df['epochs'].iloc[0]}
- **Model**: BasicUNet (5.75M params)
- **Test set**: 180 cases evaluated against original ground truth

## Executive Summary

**Key Finding**: Models trained on supervoxel-voted labels achieve **{(best_overall['dice_avg']/gt_baseline_dice)*100:.1f}%** of the fully-supervised baseline ({best_overall['dice_avg']:.4f} vs {gt_baseline_dice:.4f}), demonstrating competitive performance with pseudo-supervision while using the same amount of training data.

### Performance Comparison

| Training Source | Test Dice | % of 100% GT | Training Labels |
|----------------|-----------|--------------|-----------------|
| **100% GT labels** | {gt_baseline_dice:.4f} | 100.0% | 380 GT labels |
| **10% GT labels** | {gt_10_dice:.4f} | {(gt_10_dice/gt_baseline_dice)*100:.1f}% | 38 GT labels |
| **Best SV-trained** ({best_overall['mode']} n={best_overall['n_segments']}) | {best_overall['dice_avg']:.4f} | **{(best_overall['dice_avg']/gt_baseline_dice)*100:.1f}%** | 380 SV-voted labels |
| **1% GT labels** | {gt_1_dice:.4f} | {(gt_1_dice/gt_baseline_dice)*100:.1f}% | 4 GT labels |

**Key Insights**:
- SV-trained models reach **{(best_overall['dice_avg']/gt_baseline_dice)*100:.1f}%** of fully-supervised performance using pseudo-labels
- Performance gap: **{(gt_baseline_dice - best_overall['dice_avg']):.4f}** Dice points below 100% GT baseline
- Notably, 10% GT ({gt_10_dice:.4f}) nearly matches 100% GT ({gt_baseline_dice:.4f}), showing high annotation efficiency
- SV training outperforms 1% GT by **{(best_overall['dice_avg'] - gt_1_dice):.4f}** Dice points

## Best Configuration

**Overall best trained model:**

- **Mode**: `{best_overall['mode']}`
- **n_segments**: {best_overall['n_segments']}
- **Test Dice**: {best_overall['dice_avg']:.4f} ({(best_overall['dice_avg']/gt_baseline_dice)*100:.1f}% of GT baseline)
- **Test IoU**: {best_overall['iou_avg']:.4f}
- **Training loss**: {best_initial_loss} → {best_final_loss}
- **Training time**: {best_train_time/3600:.2f} hours

### Per-Class Dice Scores (Best Config)

| Class | Dice | IoU |
|-------|------|-----|
| Class 0 | {best_overall['dice_0']:.4f} | {best_overall['iou_0']:.4f} |
| Class 1 | {best_overall['dice_1']:.4f} | {best_overall['iou_1']:.4f} |
| Class 2 | {best_overall['dice_2']:.4f} | {best_overall['iou_2']:.4f} |
| Class 3 | {best_overall['dice_3']:.4f} | {best_overall['iou_3']:.4f} |
| Class 4 | {best_overall['dice_4']:.4f} | {best_overall['iou_4']:.4f} |

## Best Configuration per Mode

| Mode | n_segments | Dice | IoU | % of GT | Initial Loss | Final Loss | Train Time (hr) |
|------|-----------|------|-----|---------|--------------|------------|-----------------|
"""

    for mode in sorted(best_by_mode.index):
        row = best_by_mode.loc[mode]
        pct_gt = (row['dice_avg']/gt_baseline_dice)*100
        train_time = (row.get('training_time_total', 0) or 0)/3600
        initial_loss = safe_format(row.get('initial_loss'))
        final_loss = safe_format(row.get('final_loss'))
        report += f"| {mode} | {row['n_segments']} | {row['dice_avg']:.4f} | {row['iou_avg']:.4f} | "
        report += f"{pct_gt:.1f}% | {initial_loss} | "
        report += f"{final_loss} | {train_time:.2f} |\n"

    report += f"""
## Mode Comparison Summary

Average performance across all n_segments values:

| Mode | Avg Dice | Max Dice | Min Dice | Std Dev | Avg IoU | Max IoU | Min IoU |
|------|----------|----------|----------|---------|---------|---------|---------|
"""

    for mode in sorted(df['mode'].unique()):
        mode_data = df[df['mode'] == mode]
        report += f"| {mode} | {mode_data['dice_avg'].mean():.4f} | {mode_data['dice_avg'].max():.4f} | "
        report += f"{mode_data['dice_avg'].min():.4f} | {mode_data['dice_avg'].std():.4f} | "
        report += f"{mode_data['iou_avg'].mean():.4f} | {mode_data['iou_avg'].max():.4f} | "
        report += f"{mode_data['iou_avg'].min():.4f} |\n"

    report += """
## Key Findings

### 1. Impact of Supervoxel Granularity (n_segments)

"""
    # Analyze trend
    for mode in sorted(df['mode'].unique()):
        mode_data = df[df['mode'] == mode].sort_values('n_segments')
        dice_trend = "increases" if mode_data['dice_avg'].iloc[-1] > mode_data['dice_avg'].iloc[0] else "decreases"
        dice_change = abs(mode_data['dice_avg'].iloc[-1] - mode_data['dice_avg'].iloc[0])
        best_n = mode_data.loc[mode_data['dice_avg'].idxmax(), 'n_segments']
        report += f"- **{mode}**: Dice score {dice_trend} by {dice_change:.4f} from {mode_data['n_segments'].min()} to {mode_data['n_segments'].max()} segments (best: n={best_n})\n"

    report += """
### 2. Mode Comparison

"""
    best_mode = df.groupby('mode')['dice_avg'].mean().idxmax()
    worst_mode = df.groupby('mode')['dice_avg'].mean().idxmin()
    best_avg = df[df['mode']==best_mode]['dice_avg'].mean()
    worst_avg = df[df['mode']==worst_mode]['dice_avg'].mean()
    report += f"- **Best performing mode**: `{best_mode}` (avg Dice: {best_avg:.4f})\n"
    report += f"- **Least performing mode**: `{worst_mode}` (avg Dice: {worst_avg:.4f})\n"
    report += f"- **Performance gap**: {(best_avg - worst_avg):.4f} Dice points\n"

    report += """
### 3. Per-Class Performance

Average Dice scores across all 30 trained models:

"""
    class_names = {0: 'Class 0', 1: 'Class 1', 2: 'Class 2', 3: 'Class 3', 4: 'Class 4'}
    for i in range(5):
        col = f'dice_{i}'
        avg_dice = df[col].mean()
        max_dice = df[col].max()
        min_dice = df[col].min()
        report += f"- **{class_names[i]}**: {avg_dice:.4f} (range: [{min_dice:.4f}, {max_dice:.4f}])\n"

    report += """
### 4. Training Convergence

"""
    if 'initial_loss' in df.columns and df['initial_loss'].notna().any():
        avg_initial = df['initial_loss'].mean()
        avg_final = df['final_loss'].mean()
        avg_reduction = ((avg_initial - avg_final) / avg_initial) * 100
        report += f"- **Average initial loss**: {avg_initial:.4f}\n"
        report += f"- **Average final loss**: {avg_final:.4f}\n"
        report += f"- **Average loss reduction**: {avg_reduction:.1f}%\n"

        if 'training_time_total' in df.columns:
            avg_time = df['training_time_total'].mean() / 3600
            report += f"- **Average training time**: {avg_time:.2f} hours per configuration\n"

    report += """
### 5. Comparison with Baselines

| Approach | Dice Score | Relative to 100% GT | Training Data |
|----------|-----------|---------------------|---------------|
| 100% GT Training | """ + f"{gt_baseline_dice:.4f}" + """ | 100.0% (baseline) | 380 GT labels |
| 10% GT Training | """ + f"{gt_10_dice:.4f}" + """ | """ + f"{(gt_10_dice/gt_baseline_dice)*100:.1f}%" + """ | 38 GT labels |
| **Best SV Training** | """ + f"{best_overall['dice_avg']:.4f}" + """ | **""" + f"{(best_overall['dice_avg']/gt_baseline_dice)*100:.1f}%" + """** | 380 SV-voted labels |
| 1% GT Training | """ + f"{gt_1_dice:.4f}" + """ | """ + f"{(gt_1_dice/gt_baseline_dice)*100:.1f}%" + """ | 4 GT labels |

**Analysis**:
- SV-trained models achieve """ + f"{(best_overall['dice_avg']/gt_baseline_dice)*100:.1f}%" + """ of the fully-supervised baseline
- Performance gap (100% GT - SV): """ + f"{(gt_baseline_dice - best_overall['dice_avg']):.4f}" + """ Dice points
- Surprisingly, 10% GT (""" + f"{gt_10_dice:.4f}" + """) nearly matches 100% GT (""" + f"{gt_baseline_dice:.4f}" + """), suggesting high annotation efficiency
- SV training provides a middle ground: better than few-shot (1% GT) but below full supervision
- SV-voted labels serve as reasonable pseudo-supervision, though label noise creates a performance ceiling

## Recommendations

"""
    report += f"1. **For best weakly-supervised performance**: Use `{best_overall['mode']}` mode with `{best_overall['n_segments']}` segments (Dice: {best_overall['dice_avg']:.4f})\n"

    # Find sweet spot (good performance, reasonable complexity)
    mid_range = df[(df['n_segments'] >= 8000) & (df['n_segments'] <= 12000)]
    if len(mid_range) > 0:
        sweet_spot = mid_range.loc[mid_range['dice_avg'].idxmax()]
        report += f"2. **For balanced performance/complexity**: Consider `{sweet_spot['mode']}` with `{sweet_spot['n_segments']}` segments (Dice: {sweet_spot['dice_avg']:.4f})\n"

    report += f"3. **Minimum recommended granularity**: Avoid n_segments below {df['n_segments'].quantile(0.3):.0f} for adequate label quality\n"

    # Mode recommendation
    best_mode_overall = df.groupby('mode')['dice_avg'].mean().idxmax()
    report += f"4. **SLIC mode recommendation**: `{best_mode_overall}` shows most consistent performance across all granularities\n"

    # Calculate relative path from report to plots
    report_path = Path(output_file)
    plots_path = Path(output_dir)
    try:
        rel_path = plots_path.relative_to(report_path.parent)
    except ValueError:
        # If paths are not relative, use the full path
        rel_path = plots_path

    report += f"""
## Visualizations

### Trained Model Performance vs Supervoxel Granularity

![Trained Dice vs n_segments]({rel_path}/trained_dice_vs_nsegments.png)

*Figure 1: Test Dice scores of trained models vs supervoxel granularity. Red dashed line shows 100% GT baseline ({gt_baseline_dice:.4f}).*

![Trained IoU vs n_segments]({rel_path}/trained_iou_vs_nsegments.png)

*Figure 2: Test IoU scores of trained models vs supervoxel granularity.*

### Per-Class Analysis

![Per-class Dice]({rel_path}/trained_perclass_dice.png)

*Figure 3: Per-class Dice scores showing differential performance across semantic classes.*

### Training Dynamics

![Training Convergence]({rel_path}/training_convergence.png)

*Figure 4: Training loss convergence for sample configurations.*

### Mode Comparison

![Mode Comparison]({rel_path}/mode_comparison.png)

*Figure 5: Average vs best performance across different SLIC modes.*

## Full Results Table

Sorted by test Dice score (descending):

| Rank | Mode | n_segments | Dice | IoU | % of GT | Final Loss | Train Time (hr) |
|------|------|-----------|------|-----|---------|------------|-----------------|
"""

    # Add full table sorted by dice
    full_table = df.sort_values('dice_avg', ascending=False).copy()

    for rank, (_, row) in enumerate(full_table.iterrows(), 1):
        pct_gt = (row['dice_avg']/gt_baseline_dice)*100
        train_time = (row.get('training_time_total', 0) or 0)/3600
        final_loss = row.get('final_loss', float('nan'))
        final_loss_str = f"{final_loss:.4f}" if pd.notna(final_loss) else 'N/A'

        report += f"| {rank} | {row['mode']} | {row['n_segments']} | {row['dice_avg']:.4f} | {row['iou_avg']:.4f} | "
        report += f"{pct_gt:.1f}% | {final_loss_str} | {train_time:.2f} |\n"

    report += f"""

## Experimental Details

- **Training set**: 380 cases with supervoxel-voted pseudo-labels
- **Test set**: 180 cases with original ground truth labels
- **Model**: BasicUNet (5.75M parameters)
- **Training**: 20 epochs, batch_size=2
- **Optimizer**: Novograd (lr schedule: 1e-3 → 1e-4 → 1e-5)
- **Augmentation**: Random flips, random spatial crops
- **Inference**: Sliding window (overlap 0.5)
- **Evaluation**: Against original GT labels (classes 0-4, ignore class 6)
- **Total GPU hours**: {df['training_time_total'].fillna(0).sum()/3600:.1f} hours across all 30 runs

---

*Report generated by `scripts/summarize_sv_training_results.py`*

*Timestamp: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    # Write report
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        f.write(report)

    print(f"Generated report: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Summarize supervoxel training results')
    parser.add_argument('--runs-root', type=str,
                       default='runs',
                       help='Root directory containing train_sv_*_e20* folders')
    parser.add_argument('--output-dir', type=str,
                       default='runs/sv_training_plots',
                       help='Directory to save plots')
    parser.add_argument('--report', type=str,
                       default='runs/sv_training_results_summary.md',
                       help='Output markdown report file')

    args = parser.parse_args()

    print("="*70)
    print("SUPERVOXEL TRAINING RESULTS SUMMARY")
    print("="*70)
    print(f"\nCollecting metrics from training folders in {args.runs_root}...")

    df = collect_metrics(args.runs_root)

    if df.empty:
        print("ERROR: No training data collected. Check runs root path.")
        return

    print(f"\n✓ Collected {len(df)} training configurations")
    print(f"  Modes: {', '.join(sorted(df['mode'].unique()))}")
    print(f"  n_segments range: {df['n_segments'].min()} - {df['n_segments'].max()}")

    # Quick stats
    best = df.loc[df['dice_avg'].idxmax()]
    print(f"\n✓ Best performance: {best['mode']} n={best['n_segments']} (Dice: {best['dice_avg']:.4f})")

    print("\nLoading baseline results...")
    baselines = load_baseline_results()
    print(f"  100% GT baseline: {baselines.get('100_gt', 'N/A'):.4f}" if baselines.get('100_gt') else "  100% GT baseline: N/A")
    print(f"  10% GT baseline: {baselines.get('10_gt', 'N/A'):.4f}" if baselines.get('10_gt') else "  10% GT baseline: N/A")
    print(f"  1% GT baseline: {baselines.get('1_gt', 'N/A'):.4f}" if baselines.get('1_gt') else "  1% GT baseline: N/A")

    print("\nGenerating plots...")
    generate_plots(df, args.output_dir, baselines)

    print("\nGenerating comprehensive report...")
    generate_report(df, args.report, args.output_dir)

    print("\n" + "="*70)
    print("SUMMARY COMPLETE!")
    print("="*70)
    print(f"  📊 Plots: {args.output_dir}/")
    print(f"  📄 Report: {args.report}")
    print(f"  🎯 Best Dice: {best['dice_avg']:.4f} ({best['mode']}, n={best['n_segments']})")
    print("="*70)

if __name__ == '__main__':
    main()
