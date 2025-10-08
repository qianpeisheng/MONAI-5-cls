#!/usr/bin/env python3
"""
Verify static few-shot supervision budgets saved on disk.

Reads <run_dir>/sup_masks/*_supmask_stats.json and reports summary statistics
for supervised fraction (dilated mask) and seed fraction (seed points only).

Usage examples:
  python3 scripts/verify_sup_masks.py --run_dir runs/fewshot_static_10 --expect_ratio 0.10 --dilate_radius 1 --shape cross
  python3 scripts/verify_sup_masks.py --run_dir runs/fewshot_static_01 --expect_ratio 0.01 --dilate_radius 1 --shape cube
"""

import argparse
import glob
import json
import math
import os
from pathlib import Path


def fmt(x: float) -> str:
    return f"{x:.6f}" if isinstance(x, float) else str(x)


def summarize(values):
    if not values:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var)
    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
    }


def main():
    ap = argparse.ArgumentParser(description="Verify saved supervision mask budgets")
    ap.add_argument("--run_dir", required=True, help="Run directory containing sup_masks/")
    ap.add_argument("--expect_ratio", type=float, default=None, help="Expected supervised fraction (dilated)")
    ap.add_argument("--dilate_radius", type=int, default=1, help="Dilation radius used for few-point masks")
    ap.add_argument("--shape", type=str, default="cube", choices=["cube", "cross"], help="Dilation shape: 'cube' (Chebyshev) or 'cross' (Manhattan)")
    ap.add_argument("--fail_tolerance", type=float, default=0.02, help="Allowed absolute deviation from expected supervised fraction")
    args = ap.parse_args()

    mask_dir = Path(args.run_dir) / "sup_masks"
    stats_glob = str(mask_dir / "*_supmask_stats.json")
    files = sorted(glob.glob(stats_glob))
    if not files:
        print(f"No stats files found under: {stats_glob}")
        raise SystemExit(1)

    sup_fracs = []
    seed_fracs = []
    bad = []
    for f in files:
        try:
            s = json.loads(Path(f).read_text())
        except Exception as e:
            print(f"WARN: failed to read {f}: {e}")
            continue
        sup = float(s.get("sup_fraction", 0.0))
        seed = float(s.get("seed_fraction", 0.0))
        sup_fracs.append(sup)
        seed_fracs.append(seed)

    sup_sum = summarize(sup_fracs)
    seed_sum = summarize(seed_fracs)

    print(f"Volumes: {sup_sum['count']}")
    print("Supervised (dilated) fraction:")
    print("  mean=", fmt(sup_sum['mean']), " std=", fmt(sup_sum['std']), " min=", fmt(sup_sum['min']), " max=", fmt(sup_sum['max']))
    print("Seed (points-only) fraction:")
    print("  mean=", fmt(seed_sum['mean']), " std=", fmt(seed_sum['std']), " min=", fmt(seed_sum['min']), " max=", fmt(seed_sum['max']))

    if args.expect_ratio is not None:
        # Check supervised fraction against expectation
        deviates = [i for i, v in enumerate(sup_fracs) if abs(v - args.expect_ratio) > args.fail_tolerance]
        if deviates:
            print(f"WARNING: {len(deviates)} / {len(sup_fracs)} volumes deviate more than {args.fail_tolerance} from expected {args.expect_ratio}")
        else:
            print(f"OK: all volumes within ±{args.fail_tolerance} of expected {args.expect_ratio}")
        # Seed budget equals expect_ratio by policy
        print(f"Expected seed fraction: {fmt(args.expect_ratio)} (global budget)")
        # Approximate expected sup fraction upper bound (ignoring edges/overlaps)
        if args.shape == 'cube':
            kv = (2 * args.dilate_radius + 1) ** 3
        else:
            kv = 1 + 6 * args.dilate_radius
        exp_sup = args.expect_ratio * kv
        print(f"Approx sup fraction upper bound (~): {fmt(exp_sup)} (ratio * neighborhood_size)")


if __name__ == "__main__":
    main()
