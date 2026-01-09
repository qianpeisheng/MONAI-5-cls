#!/usr/bin/env python3
"""
Pipeline: Graph Label Propagation over supervoxels + evaluation vs GT.

Steps:
  1) Run Graph LP on supervoxels to generate dense voxel labels
  2) Evaluate propagated labels against ground truth using eval_sv_voted_wp5.py

Example (new WP5 dataset, train split, k=10, alpha=0.9):

  python3 scripts/pipeline_graph_lp_sv.py \
    --sv_dir runs/sv_fullgt_slic_n12000_new_ras \
    --seeds_dir runs/strategic_sparse_0p1pct_new/strategic_seeds \
    --datalist datalist_train_new.json \
    --output_dir runs/graph_lp_prop_0p1pct_k10_a0.9_new \
    --k 10 --alpha 0.9 --seed 42 \
    --eval_num_workers 16 --eval_progress --eval_log_to_file --eval_heavy
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], desc: str) -> None:
    """Run a subprocess command with basic error handling."""
    print("\n" + "=" * 70)
    print(desc)
    print("=" * 70)
    print("Command:", " ".join(cmd), "\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: {desc} failed with code {result.returncode}")
        sys.exit(result.returncode)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Graph LP over SVs + evaluation vs GT (WP5-style)."
    )
    p.add_argument(
        "--sv_dir",
        type=str,
        required=True,
        help="Directory with supervoxel IDs (*_sv_ids.npy), e.g. runs/sv_fullgt_slic_n12000_new_ras",
    )
    p.add_argument(
        "--seeds_dir",
        type=str,
        required=True,
        help="Directory with sparse SV labels (*_sv_labels_sparse.json) from strategic sampling.",
    )
    p.add_argument(
        "--datalist",
        type=str,
        default="datalist_train.json",
        help="MONAI-style datalist JSON with records {image,label,id}.",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="",
        help="Optional dataset root for eval (not needed if datalist has absolute label paths).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base output directory for Graph LP results (labels) and evaluation.",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="Graph LP neighbor count k (default: 10).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Graph LP propagation strength alpha (default: 0.9).",
    )
    p.add_argument(
        "--num_classes",
        type=int,
        default=5,
        help="Number of classes for Graph LP (default: 5 for WP5 classes 0..4).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Graph LP (default: 42).",
    )
    p.add_argument(
        "--ignore_class",
        type=int,
        default=6,
        help="Ignore label for evaluation (default: 6).",
    )
    p.add_argument(
        "--lp_num_workers",
        type=int,
        default=16,
        help="Number of parallel workers for Graph LP over cases (default: 16).",
    )
    p.add_argument(
        "--eval_num_workers",
        type=int,
        default=16,
        help="Number of parallel workers for evaluation (default: 16).",
    )
    p.add_argument(
        "--eval_heavy",
        action="store_true",
        help="If set, compute HD/ASD in evaluation.",
    )
    p.add_argument(
        "--eval_hd_percentile",
        type=float,
        default=95.0,
        help="Hausdorff percentile for evaluation (default: 95.0).",
    )
    p.add_argument(
        "--eval_progress",
        action="store_true",
        help="If set, show progress bar during evaluation.",
    )
    p.add_argument(
        "--eval_log_to_file",
        action="store_true",
        help="If set, tee evaluation logs to eval.log in the eval output dir.",
    )
    p.add_argument(
        "--use_outer_bg_split",
        action="store_true",
        help=(
            "If set, propagate_graph_lp_multi_case will run LP only on ROI SVs "
            "and force outer-background SVs to label 0 (requires seeds_meta with outer_bg_sv_ids)."
        ),
    )
    p.add_argument(
        "--descriptor_type",
        type=str,
        default="none",
        choices=["none", "moments", "quantiles16", "hist32"],
        help="Optional intensity descriptor for graph weights (default: none = coords-only).",
    )
    p.add_argument(
        "--descriptor_cache_dir",
        type=str,
        default="",
        help="Optional cache directory for computed SV descriptors.",
    )
    p.add_argument(
        "--use_cosine",
        action="store_true",
        help="Use cosine distance for moments/quantiles descriptors (default: L2).",
    )
    p.add_argument(
        "--sigma_phi",
        type=str,
        default="median",
        help="Descriptor kernel sigma: float or 'median' (default: median).",
    )
    p.add_argument(
        "--quantiles_include_mad",
        action="store_true",
        help="If set, append MAD to quantiles16 descriptor (default: False).",
    )
    p.add_argument(
        "--hist_bins",
        type=int,
        default=32,
        help="Histogram bins for hist32 descriptor (default: 32).",
    )
    p.add_argument(
        "--hist_range",
        type=float,
        nargs=2,
        default=[-3.0, 3.0],
        metavar=("VMIN", "VMAX"),
        help="Histogram intensity range [vmin vmax] on normalized image (default: -3 3).",
    )
    p.add_argument(
        "--moments_trim_ratio",
        type=float,
        default=0.1,
        help="Trim ratio for moments trimmed mean (default: 0.1 = 10%% each tail).",
    )
    p.add_argument(
        "--sample_edges_for_sigma",
        type=int,
        default=50_000,
        help="Max neighbor edges to sample when sigma_phi='median' (default: 50000).",
    )

    args = p.parse_args()

    sv_dir = Path(args.sv_dir)
    seeds_dir = Path(args.seeds_dir)
    base_out = Path(args.output_dir)
    lp_out = base_out  # Graph LP writes here directly
    eval_out = base_out / "eval"

    # Step 1: Graph Label Propagation over SVs
    cmd_lp = [
        "python3",
        "scripts/propagate_graph_lp_multi_case.py",
        "--sv_dir",
        str(sv_dir),
        "--seeds_dir",
        str(seeds_dir),
        "--k",
        str(args.k),
        "--alpha",
        str(args.alpha),
        "--output_dir",
        str(lp_out),
        "--num_classes",
        str(args.num_classes),
        "--seed",
        str(args.seed),
        "--num_workers",
        str(args.lp_num_workers),
        "--descriptor_type",
        str(args.descriptor_type),
    ]
    if args.datalist:
        cmd_lp += ["--datalist", str(args.datalist)]
    if args.data_root:
        cmd_lp += ["--data_root", str(args.data_root)]
    if args.descriptor_cache_dir:
        cmd_lp += ["--descriptor_cache_dir", str(args.descriptor_cache_dir)]
    if args.use_cosine:
        cmd_lp.append("--use_cosine")
    if args.sigma_phi:
        cmd_lp += ["--sigma_phi", str(args.sigma_phi)]
    if args.quantiles_include_mad:
        cmd_lp.append("--quantiles_include_mad")
    cmd_lp += ["--hist_bins", str(args.hist_bins)]
    cmd_lp += ["--hist_range", str(args.hist_range[0]), str(args.hist_range[1])]
    cmd_lp += ["--moments_trim_ratio", str(args.moments_trim_ratio)]
    cmd_lp += ["--sample_edges_for_sigma", str(args.sample_edges_for_sigma)]
    if args.use_outer_bg_split:
        cmd_lp.append("--use_outer_bg_split")
    _run(cmd_lp, "STEP 1: Graph LP propagation over supervoxels")

    # Step 2: Evaluate propagated labels vs GT
    eval_out.mkdir(parents=True, exist_ok=True)

    cmd_eval = [
        "python3",
        "scripts/eval_sv_voted_wp5.py",
        "--sv-dir",
        str(lp_out / "labels"),
        "--sv-ids-dir",
        str(sv_dir),
        "--datalist",
        str(args.datalist),
        "--output_dir",
        str(eval_out),
        "--ignore-class",
        str(args.ignore_class),
        "--num_workers",
        str(args.eval_num_workers),
    ]

    if args.data_root:
        cmd_eval += ["--data-root", str(args.data_root)]
    if args.eval_heavy:
        cmd_eval += ["--heavy", "--hd_percentile", str(args.eval_hd_percentile)]
    if args.eval_progress:
        cmd_eval.append("--progress")
    if args.eval_log_to_file:
        cmd_eval.append("--log_to_file")

    _run(cmd_eval, "STEP 2: Evaluate Graph LP-propagated labels vs GT")

    print("\n" + "=" * 70)
    print("GRAPH LP + EVAL PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Graph LP output dir: {lp_out}")
    print(f"  Training labels (for models): {lp_out}/labels/")
    print(f"  Evaluation output dir: {eval_out}")
    print("  - Summary metrics: eval/metrics/summary.json")
    print("  - Per-case metrics: eval/metrics/per_case.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
