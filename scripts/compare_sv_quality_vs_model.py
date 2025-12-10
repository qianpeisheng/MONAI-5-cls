#!/usr/bin/env python3
"""Compare SV label quality vs trained model performance."""

import re
from pathlib import Path

# Parse SV sweep results
sv_results = {}
summary_file = Path("runs/sv_sweep_ras2_summary.md")
if summary_file.exists():
    for line in summary_file.read_text().splitlines():
        if line.startswith("| slic"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 5:
                mode = parts[1]
                n_seg = parts[2]
                dice = parts[3]
                config = f"{mode}_n{n_seg}_c0.05_s1.0"
                try:
                    sv_results[config] = float(dice)
                except:
                    pass

# Parse training results
model_results = {}
for train_dir in Path("runs").glob("train_sv_*_e20_*/"):
    log_file = train_dir / "train.log"
    if log_file.exists():
        config = train_dir.name.replace("train_sv_", "").split("_e20_")[0]

        # Get final test result
        for line in log_file.read_text().splitlines():
            if "Epoch" in line and "test avg" in line:
                match = re.search(r"overall ([0-9.]+)", line)
                if match:
                    model_results[config] = float(match.group(1))

# Print comparison
print(f"{'Config':<45} | {'SV Dice':>8} | {'Model Dice':>10} | {'Difference':>10}")
print("-" * 85)

results = []
for config in sorted(model_results.keys()):
    if config in sv_results:
        sv_dice = sv_results[config]
        model_dice = model_results[config]
        diff = model_dice - sv_dice
        results.append((config, sv_dice, model_dice, diff))
        print(f"{config:<45} | {sv_dice:8.4f} | {model_dice:10.4f} | {diff:+10.4f}")

if results:
    print("-" * 85)
    avg_sv = sum(r[1] for r in results) / len(results)
    avg_model = sum(r[2] for r in results) / len(results)
    avg_diff = sum(r[3] for r in results) / len(results)
    print(f"{'AVERAGE':<45} | {avg_sv:8.4f} | {avg_model:10.4f} | {avg_diff:+10.4f}")

    print(f"\nKey findings:")
    print(f"  • SV quality range: {min(r[1] for r in results):.4f} - {max(r[1] for r in results):.4f} (Δ={max(r[1] for r in results)-min(r[1] for r in results):.4f})")
    print(f"  • Model performance range: {min(r[2] for r in results):.4f} - {max(r[2] for r in results):.4f} (Δ={max(r[2] for r in results)-min(r[2] for r in results):.4f})")
    print(f"  • Average gap: {avg_diff:+.4f} ({'model better' if avg_diff > 0 else 'SV better'})")
else:
    print("No matching results found")
