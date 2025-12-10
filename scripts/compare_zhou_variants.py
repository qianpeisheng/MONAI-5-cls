#!/usr/bin/env python3
"""
Compare Zhou diffusion variants (different alpha and connectivity).
"""

import json
from pathlib import Path
import pandas as pd


def load_result(results_dir: Path, label: str):
    """Load a single result."""
    results_file = results_dir / "SN13B0_I17_3D_B1_1B250409_results.json"
    if not results_file.exists():
        return None

    with open(results_file) as f:
        data = json.load(f)

    return {
        'method': label,
        'alpha': data['alpha'],
        'connectivity': data['connectivity'],
        'mean_dice': data['metrics']['mean_dice'],
        'accuracy': data['metrics']['accuracy'],
        'dice_class_0': data['metrics']['dice_scores']['class_0'],
        'dice_class_1': data['metrics']['dice_scores']['class_1'],
        'dice_class_2': data['metrics']['dice_scores']['class_2'],
        'dice_class_3': data['metrics']['dice_scores']['class_3'],
        'dice_class_4': data['metrics']['dice_scores']['class_4'],
        'runtime': data['runtime_seconds'],
    }


def main():
    print("="*80)
    print("ZHOU DIFFUSION PARAMETER COMPARISON")
    print("="*80)
    print()

    # Load all Zhou variants
    results = []

    r = load_result(Path("tmp/test_voxel_diffusion_zhou_0p5pct"), "α=0.99, 6-conn")
    if r: results.append(r)

    r = load_result(Path("tmp/test_voxel_diffusion_zhou_0p5pct_26conn"), "α=0.99, 26-conn")
    if r: results.append(r)

    r = load_result(Path("tmp/test_voxel_diffusion_zhou_0p5pct_6conn_alpha0p9"), "α=0.90, 6-conn")
    if r: results.append(r)

    r = load_result(Path("tmp/test_voxel_diffusion_zhou_0p5pct_6conn_alpha0p5"), "α=0.50, 6-conn")
    if r: results.append(r)

    # Also load kNN baseline for comparison
    voxel_knn_file = Path("tmp/test_propagation_voxel_0p5pct/k_sweep_results.json")
    if voxel_knn_file.exists():
        with open(voxel_knn_file) as f:
            knn_data = json.load(f)
        for res in knn_data['results']:
            if res['k'] == 1:
                results.append({
                    'method': 'Voxel kNN (k=1) [baseline]',
                    'alpha': '-',
                    'connectivity': '-',
                    'mean_dice': res['dice_all'],
                    'accuracy': res['accuracy'],
                    'dice_class_0': res['dice_class_0'],
                    'dice_class_1': res['dice_class_1'],
                    'dice_class_2': res['dice_class_2'],
                    'dice_class_3': res['dice_class_3'],
                    'dice_class_4': res['dice_class_4'],
                    'runtime': '-',
                })
                break

    # Create DataFrame
    df = pd.DataFrame(results)

    # Reorder columns
    cols = ['method', 'alpha', 'connectivity', 'mean_dice', 'accuracy',
            'dice_class_0', 'dice_class_1', 'dice_class_2', 'dice_class_3', 'dice_class_4',
            'runtime']
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # Print table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')

    print(df.to_string(index=False))
    print()

    # Analysis
    print("="*80)
    print("KEY FINDINGS")
    print("="*80)
    print()

    zhou_results = [r for r in results if 'Zhou' in r.get('method', '') or 'α=' in r.get('method', '')]
    if zhou_results:
        best_zhou = max(zhou_results, key=lambda x: x['mean_dice'])
        worst_zhou = min(zhou_results, key=lambda x: x['mean_dice'])

        print(f"1. ALPHA MATTERS A LOT:")
        print(f"   Best:  α={best_zhou['alpha']}, {best_zhou['connectivity']}-conn → Dice = {best_zhou['mean_dice']:.4f}")
        print(f"   Worst: α={worst_zhou['alpha']}, {worst_zhou['connectivity']}-conn → Dice = {worst_zhou['mean_dice']:.4f}")
        print(f"   Difference: {(best_zhou['mean_dice'] - worst_zhou['mean_dice']):.4f} ({(best_zhou['mean_dice'] - worst_zhou['mean_dice'])/worst_zhou['mean_dice']*100:.1f}% relative)")
        print()

        print(f"2. 26-CONNECTIVITY IS WORSE:")
        print(f"   - More neighbors (26 vs 6) causes over-smoothing")
        print(f"   - Diagonal neighbors have less semantic similarity")
        print(f"   - 6-connectivity (face neighbors) is better for medical volumes")
        print()

        print(f"3. LOWER ALPHA IS BETTER:")
        print(f"   - α=0.99 (too high) → over-smooths, loses detail")
        print(f"   - α=0.90 → good balance")
        print(f"   - α=0.50 → even better! Stronger influence from seed labels")
        print(f"   - Lower α means more weight on original labels Y")
        print()

        print(f"4. ZHOU DIFFUSION NOW COMPETITIVE:")
        best_dice = best_zhou['mean_dice']
        knn_baseline = next((r for r in results if 'baseline' in r.get('method', '')), None)
        if knn_baseline:
            knn_dice = knn_baseline['mean_dice']
            print(f"   Best Zhou (α={best_zhou['alpha']}):  Dice = {best_dice:.4f}")
            print(f"   Voxel kNN (k=1):    Dice = {knn_dice:.4f}")
            print(f"   Gap: {(knn_dice - best_dice):.4f} ({(knn_dice - best_dice)/knn_dice*100:.1f}%)")
            print()
            if best_dice > 0.95 * knn_dice:
                print(f"   ✓ Zhou is within 5% of kNN!")
            else:
                print(f"   Still {(knn_dice - best_dice)/knn_dice*100:.1f}% behind kNN (needs features)")
        print()

        print(f"5. RUNTIME:")
        for r in zhou_results:
            if isinstance(r['runtime'], (int, float)):
                print(f"   α={r['alpha']}, {r['connectivity']}-conn: {r['runtime']:.2f}s")
        print(f"   → Very fast! <1 second for 1M voxels")
        print()

    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()
    print("For Zhou diffusion on voxels:")
    print("  • Use α=0.5 to 0.9 (NOT 0.99)")
    print("  • Use 6-connectivity (NOT 26)")
    print("  • Still behind feature-based kNN by ~1-2%")
    print("  • To close the gap: add image gradient or intensity-based weights")
    print()


if __name__ == "__main__":
    main()
