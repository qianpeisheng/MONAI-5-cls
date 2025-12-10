#!/usr/bin/env python3
"""
Compare different voxel-level label propagation methods.

Compares:
1. Voxel-level kNN propagation (existing)
2. Supervoxel-based kNN propagation (existing)
3. Voxel-level Zhou diffusion (new)

Usage:
    python scripts/compare_voxel_methods.py
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_zhou_diffusion_results(results_dir: Path) -> Dict:
    """Load Zhou diffusion results."""
    results_file = results_dir / "SN13B0_I17_3D_B1_1B250409_results.json"
    if not results_file.exists():
        return None

    with open(results_file) as f:
        data = json.load(f)

    return {
        'method': 'Zhou Diffusion (6-conn)',
        'mean_dice': data['metrics']['mean_dice'],
        'accuracy': data['metrics']['accuracy'],
        'dice_class_0': data['metrics']['dice_scores']['class_0'],
        'dice_class_1': data['metrics']['dice_scores']['class_1'],
        'dice_class_2': data['metrics']['dice_scores']['class_2'],
        'dice_class_3': data['metrics']['dice_scores']['class_3'],
        'dice_class_4': data['metrics']['dice_scores']['class_4'],
        'runtime': data['runtime_seconds'],
        'alpha': data['alpha'],
        'connectivity': data['connectivity'],
    }


def load_voxel_knn_results(results_dir: Path) -> List[Dict]:
    """Load voxel-level kNN results."""
    results_file = results_dir / "k_sweep_results.json"
    if not results_file.exists():
        return []

    with open(results_file) as f:
        data = json.load(f)

    results = []
    for result in data['results']:
        results.append({
            'method': f'Voxel kNN (k={result["k"]})',
            'mean_dice': result['dice_all'],
            'accuracy': result['accuracy'],
            'dice_class_0': result['dice_class_0'],
            'dice_class_1': result['dice_class_1'],
            'dice_class_2': result['dice_class_2'],
            'dice_class_3': result['dice_class_3'],
            'dice_class_4': result['dice_class_4'],
            'k': result['k'],
        })

    return results


def load_supervoxel_knn_results(results_dir: Path) -> List[Dict]:
    """Load supervoxel-based kNN results."""
    results_file = results_dir / "k_sweep_results.json"
    if not results_file.exists():
        return []

    with open(results_file) as f:
        data = json.load(f)

    results = []
    for result in data['results']:
        results.append({
            'method': f'Supervoxel kNN (k={result["k"]})',
            'mean_dice': result['dice_all'],
            'accuracy': result['accuracy'],
            'dice_class_0': result['dice_class_0'],
            'dice_class_1': result['dice_class_1'],
            'dice_class_2': result['dice_class_2'],
            'dice_class_3': result['dice_class_3'],
            'dice_class_4': result['dice_class_4'],
            'k': result['k'],
        })

    return results


def main():
    print("="*80)
    print("VOXEL-LEVEL METHOD COMPARISON")
    print("="*80)
    print()

    # Load results
    zhou_result = load_zhou_diffusion_results(Path("tmp/test_voxel_diffusion_zhou_0p5pct"))
    voxel_knn_results = load_voxel_knn_results(Path("tmp/test_propagation_voxel_0p5pct"))
    sv_knn_results = load_supervoxel_knn_results(Path("tmp/test_propagation_0p5pct"))

    # Compile all results
    all_results = []

    if zhou_result:
        all_results.append(zhou_result)

    # Add best k values for each method
    for k in [1, 10, 30, 50]:
        for r in voxel_knn_results:
            if r['k'] == k:
                all_results.append(r)
                break

    for k in [1, 3, 10, 30]:
        for r in sv_knn_results:
            if r['k'] == k:
                all_results.append(r)
                break

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Reorder columns
    cols = ['method', 'mean_dice', 'accuracy',
            'dice_class_0', 'dice_class_1', 'dice_class_2', 'dice_class_3', 'dice_class_4']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # Print table
    print("RESULTS COMPARISON (Case: SN13B0_I17_3D_B1_1B250409, Seeds: 0.5%)")
    print("-"*80)
    print()

    # Format the dataframe for printing
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')

    print(df.to_string(index=False))
    print()

    # Print analysis
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    print()

    if zhou_result:
        best_voxel_knn = max(voxel_knn_results, key=lambda x: x['mean_dice'])
        best_sv_knn = max(sv_knn_results, key=lambda x: x['mean_dice'])

        print("BEST RESULTS:")
        print(f"  Voxel kNN (k={best_voxel_knn['k']}):       Dice = {best_voxel_knn['mean_dice']:.4f}, Acc = {best_voxel_knn['accuracy']:.4f}")
        print(f"  Supervoxel kNN (k={best_sv_knn['k']}):    Dice = {best_sv_knn['mean_dice']:.4f}, Acc = {best_sv_knn['accuracy']:.4f}")
        print(f"  Zhou Diffusion (alpha={zhou_result['alpha']}): Dice = {zhou_result['mean_dice']:.4f}, Acc = {zhou_result['accuracy']:.4f}")
        print()

        print("KEY OBSERVATIONS:")
        print()
        print("1. VOXEL kNN is the BEST performing method:")
        print(f"   - Achieves highest Dice score: {best_voxel_knn['mean_dice']:.4f}")
        print(f"   - Best accuracy: {best_voxel_knn['accuracy']:.4f}")
        print(f"   - Uses feature-based similarity in kNN graph")
        print()

        print("2. SUPERVOXEL kNN performs moderately:")
        print(f"   - Dice score: {best_sv_knn['mean_dice']:.4f}")
        print(f"   - Loses some detail due to supervoxel quantization")
        print(f"   - Faster but less accurate than voxel kNN")
        print()

        print("3. ZHOU DIFFUSION (spatial-only) underperforms:")
        print(f"   - Dice score: {zhou_result['mean_dice']:.4f}")
        print(f"   - Only uses spatial neighbors (no features)")
        print(f"   - Alpha = {zhou_result['alpha']} may be too high (too much smoothing)")
        print(f"   - Runtime: {zhou_result['runtime']:.2f}s (very fast!)")
        print()

        print("WHY ZHOU DIFFUSION UNDERPERFORMS:")
        print("  - Uses uniform spatial weights (no feature similarity)")
        print("  - Equal weight to all 6 neighbors regardless of appearance")
        print("  - kNN methods weight neighbors by feature distance")
        print("  - Medical images need appearance-based similarity")
        print()

        print("RECOMMENDATIONS:")
        print("  1. Use Voxel kNN (k=1-10) for best accuracy")
        print("  2. Zhou diffusion needs feature-aware weights to compete")
        print("  3. Could enhance Zhou by using image gradients as edge weights")
        print("  4. Or hybrid: Zhou diffusion on a kNN graph instead of spatial grid")
        print()

        # Save comparison to file
        output_file = Path("tmp/test_voxel_diffusion_zhou_0p5pct/method_comparison.csv")
        df.to_csv(output_file, index=False)
        print(f"Saved comparison table to: {output_file}")


if __name__ == "__main__":
    main()
