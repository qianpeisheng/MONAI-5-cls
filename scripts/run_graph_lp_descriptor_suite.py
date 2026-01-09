#!/usr/bin/env python3
"""
Run Graph LP pseudo-label generation with multiple graph descriptor settings
and evaluate each result vs GT.

This script is an orchestration helper. It does not change the LP solver; it
only varies the graph affinity construction via --descriptor_type.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


def _run(cmd: list[str], desc: str) -> None:
    print("\n" + "=" * 70)
    print(desc)
    print("=" * 70)
    print("Command:", " ".join(cmd), "\n")
    subprocess.run(cmd, check=True)


def _variant_cmd(
    *,
    sv_dir: Path,
    seeds_dir: Path,
    datalist: Path,
    data_root: str,
    out_dir: Path,
    k: int,
    alpha: float,
    num_classes: int,
    seed: int,
    num_workers: int,
    descriptor_type: str,
    descriptor_cache_dir: str,
    use_cosine: bool,
    sigma_phi: str,
    quantiles_include_mad: bool,
    hist_bins: int,
    hist_range: tuple[float, float],
    moments_trim_ratio: float,
    sample_edges_for_sigma: int,
) -> list[str]:
    cmd = [
        "python3",
        "scripts/propagate_graph_lp_multi_case.py",
        "--sv_dir",
        str(sv_dir),
        "--seeds_dir",
        str(seeds_dir),
        "--datalist",
        str(datalist),
        "--data_root",
        str(data_root),
        "--output_dir",
        str(out_dir),
        "--k",
        str(k),
        "--alpha",
        str(alpha),
        "--num_classes",
        str(num_classes),
        "--seed",
        str(seed),
        "--num_workers",
        str(num_workers),
        "--descriptor_type",
        str(descriptor_type),
        "--sigma_phi",
        str(sigma_phi),
        "--hist_bins",
        str(hist_bins),
        "--hist_range",
        str(hist_range[0]),
        str(hist_range[1]),
        "--moments_trim_ratio",
        str(moments_trim_ratio),
        "--sample_edges_for_sigma",
        str(sample_edges_for_sigma),
    ]
    if descriptor_cache_dir:
        cmd += ["--descriptor_cache_dir", str(descriptor_cache_dir)]
    if use_cosine:
        cmd.append("--use_cosine")
    if quantiles_include_mad:
        cmd.append("--quantiles_include_mad")
    return cmd


def _eval_cmd(
    *,
    sv_dir_labels: Path,
    sv_ids_dir: Path,
    datalist: Path,
    data_root: str,
    out_dir: Path,
    ignore_class: int,
    num_workers: int,
    progress: bool,
    log_to_file: bool,
) -> list[str]:
    cmd = [
        "python3",
        "scripts/eval_sv_voted_wp5.py",
        "--sv-dir",
        str(sv_dir_labels),
        "--sv-ids-dir",
        str(sv_ids_dir),
        "--datalist",
        str(datalist),
        "--output_dir",
        str(out_dir),
        "--ignore-class",
        str(ignore_class),
        "--num_workers",
        str(num_workers),
    ]
    if data_root:
        cmd += ["--data-root", str(data_root)]
    if progress:
        cmd.append("--progress")
    if log_to_file:
        cmd.append("--log_to_file")
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description="Run Graph LP with intensity descriptors + evaluate vs GT.")
    p.add_argument("--sv_dir", type=str, required=True, help="Directory with *_sv_ids.npy for cases.")
    p.add_argument("--seeds_dir", type=str, required=True, help="Directory with *_sv_labels_sparse.json for cases.")
    p.add_argument("--datalist", type=str, required=True, help="MONAI-style datalist JSON with {image,label,id}.")
    p.add_argument("--data_root", type=str, default="", help="Optional root for resolving datalist paths.")
    p.add_argument("--output_root", type=str, default="runs/graph_lp_descriptor_suite", help="Base output folder.")

    p.add_argument("--k", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--num_classes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lp_num_workers", type=int, default=16)
    p.add_argument("--eval_num_workers", type=int, default=16)
    p.add_argument("--ignore_class", type=int, default=6)
    p.add_argument("--descriptor_cache_dir", type=str, default="", help="Optional descriptor cache directory.")

    p.add_argument("--sigma_phi", type=str, default="median", help="Descriptor sigma: float or 'median'.")
    p.add_argument("--use_cosine", action="store_true", help="Use cosine distance (moments/quantiles).")
    p.add_argument("--quantiles_include_mad", action="store_true", help="Append MAD to quantiles16 descriptor.")
    p.add_argument("--hist_bins", type=int, default=32)
    p.add_argument("--hist_range", type=float, nargs=2, default=[-3.0, 3.0], metavar=("VMIN", "VMAX"))
    p.add_argument("--moments_trim_ratio", type=float, default=0.1)
    p.add_argument("--sample_edges_for_sigma", type=int, default=50_000)

    p.add_argument("--skip_eval", action="store_true", help="Only generate pseudo labels; skip evaluation.")
    args = p.parse_args()

    sv_dir = Path(args.sv_dir)
    seeds_dir = Path(args.seeds_dir)
    datalist = Path(args.datalist)
    out_root = Path(args.output_root)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_out = out_root / f"graph_lp_suite_{ts}"
    base_out.mkdir(parents=True, exist_ok=True)

    variants = [
        ("baseline_coords_only", "none"),
        ("moments", "moments"),
        ("quantiles16", "quantiles16"),
        ("hist32", "hist32"),
    ]

    summary = {"created": ts, "variants": []}

    for name, dtype in variants:
        run_dir = base_out / name
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd_lp = _variant_cmd(
            sv_dir=sv_dir,
            seeds_dir=seeds_dir,
            datalist=datalist,
            data_root=str(args.data_root),
            out_dir=run_dir,
            k=int(args.k),
            alpha=float(args.alpha),
            num_classes=int(args.num_classes),
            seed=int(args.seed),
            num_workers=int(args.lp_num_workers),
            descriptor_type=dtype,
            descriptor_cache_dir=str(args.descriptor_cache_dir),
            use_cosine=bool(args.use_cosine),
            sigma_phi=str(args.sigma_phi),
            quantiles_include_mad=bool(args.quantiles_include_mad),
            hist_bins=int(args.hist_bins),
            hist_range=(float(args.hist_range[0]), float(args.hist_range[1])),
            moments_trim_ratio=float(args.moments_trim_ratio),
            sample_edges_for_sigma=int(args.sample_edges_for_sigma),
        )
        _run(cmd_lp, f"Graph LP propagation: {name}")

        eval_summary = None
        if not args.skip_eval:
            eval_dir = run_dir / "eval"
            eval_dir.mkdir(parents=True, exist_ok=True)
            cmd_eval = _eval_cmd(
                sv_dir_labels=run_dir / "labels",
                sv_ids_dir=sv_dir,
                datalist=datalist,
                data_root=str(args.data_root),
                out_dir=eval_dir,
                ignore_class=int(args.ignore_class),
                num_workers=int(args.eval_num_workers),
                progress=True,
                log_to_file=True,
            )
            _run(cmd_eval, f"Evaluate pseudo labels vs GT: {name}")
            summary_path = eval_dir / "metrics" / "summary.json"
            if summary_path.exists():
                eval_summary = str(summary_path)

        summary["variants"].append(
            {
                "name": name,
                "descriptor_type": dtype,
                "run_dir": str(run_dir),
                "eval_summary_json": eval_summary,
            }
        )

    (base_out / "suite_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSuite complete.")
    print(f"Output: {base_out}")
    print(f"Summary: {base_out / 'suite_summary.json'}")


if __name__ == "__main__":
    main()

