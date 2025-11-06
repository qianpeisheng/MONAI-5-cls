#!/usr/bin/env python3
"""
Parameter sweep for WP5 supervoxel generation and evaluation.

Loops over n_segments and generation modes (SLIC intensity / geometry-aware),
runs supervoxel generation, then evaluates the voted labels against GT using
scripts/eval_sv_voted_wp5.py. Results are written per-run under --out-root,
with each run having its own folder and an appended entry in sweep_summary.csv.

Example (A1 + A2 over 1k..20k, step 1k):
  python3 scripts/sweep_supervoxels_wp5.py --data-root /data3/wp5/wp5-code/dataloaders/wp5-dataset --datalist datalist_train.json --out-root runs/sv_sweep_ras2 --modes slic,slic-grad-mag --n-seg-min 1000 --n-seg-max 20000 --n-seg-step 1000 --compactness 0.02 --sigma 0.5 --ignore-class 6 --eval-workers 8
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Sweep WP5 supervoxels and evaluate")
    p.add_argument("--data-root", required=True, help="WP5 dataset root containing data/ with NIfTI pairs")
    p.add_argument("--datalist", default="datalist_train.json", help="JSON list with records {image,label,id}")
    p.add_argument("--out-root", required=True, help="Root directory for all sweep runs")
    p.add_argument("--modes", default="slic", help="Comma-separated modes: slic,slic-grad-mag,slic-grad-vec")
    p.add_argument("--n-seg-min", type=int, default=1000)
    p.add_argument("--n-seg-max", type=int, default=20000)
    p.add_argument("--n-seg-step", type=int, default=1000)
    p.add_argument("--compactness", type=float, default=0.02)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--compactness-list", type=str, default="", help="Comma-separated list of compactness values to sweep (overrides --compactness)")
    p.add_argument("--sigma-list", type=str, default="", help="Comma-separated list of sigma values to sweep (overrides --sigma)")
    p.add_argument("--grad-sigma", type=float, default=1.0, help="Smoothing sigma for gradient features (voxels)")
    p.add_argument("--ref-header", choices=["image", "label"], default="label")
    p.add_argument("--ignore-class", type=int, default=6)
    p.add_argument("--eval-workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--link-in-runs", type=str, default="", help="Optional path (e.g., runs/sv_sweep_ras2) where a symlink to --out-root will be created/updated")
    p.add_argument("--base-suffix", type=str, default="_ras2", help="Base suffix appended to run names")
    p.add_argument("--append-suffix", type=str, default="_voted", help="Additional suffix appended to run names (e.g., _voted)")
    return p.parse_args()


def _mk_run_name(mode: str, nseg: int, c: float, s: float, base_suffix: str, append_suffix: str) -> str:
    return f"sv_fullgt_{mode}_n{nseg}_c{c}_s{s}{base_suffix}{append_suffix}"


def _append_summary(summary_csv: Path, row: dict) -> None:
    exists = summary_csv.exists()
    with open(summary_csv, "a", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "n_segments",
                "compactness",
                "sigma",
                "grad_sigma",
                "run_dir",
                "eval_dir",
                "avg_dice",
                "avg_iou",
            ],
        )
        if not exists:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "sweep_summary.csv"

    # Optionally create/update a symlink under repo runs/ to the out_root on /data3
    if args.link_in_runs:
        link_path = Path(args.link_in_runs)
        link_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if link_path.exists() or link_path.is_symlink():
                if link_path.is_symlink() or link_path.resolve() == out_root:
                    link_path.unlink(missing_ok=True)  # type: ignore[arg-type]
                else:
                    print(f"[warn] link target exists and is not a symlink: {link_path}; skipping link update", file=sys.stderr)
            # Create new symlink
            link_path.symlink_to(out_root, target_is_directory=True)
            print(f"Linked {link_path} -> {out_root}")
        except Exception as e:
            print(f"[warn] failed to create symlink {link_path} -> {out_root}: {e}", file=sys.stderr)

    modes: List[str] = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes:
        print("No modes specified.", file=sys.stderr)
        sys.exit(1)

    n_values = list(range(int(args.n_seg_min), int(args.n_seg_max) + 1, int(args.n_seg_step)))
    gen_script = Path("scripts/gen_supervoxels_wp5.py").resolve()
    eval_script = Path("scripts/eval_sv_voted_wp5.py").resolve()
    if not gen_script.exists() or not eval_script.exists():
        print("Required scripts not found under scripts/", file=sys.stderr)
        sys.exit(1)

    # Build sweep grids
    if args.compactness_list:
        c_values = [float(x) for x in args.compactness_list.split(",") if x.strip()]
    else:
        c_values = [float(args.compactness)]
    if args.sigma_list:
        s_values = [float(x) for x in args.sigma_list.split(",") if x.strip()]
    else:
        s_values = [float(args.sigma)]

    for mode in modes:
        for nseg in n_values:
            for c in c_values:
                for s in s_values:
                    run_name = _mk_run_name(mode, nseg, c, s, args.base_suffix, args.append_suffix)
            run_dir = out_root / run_name
            eval_dir = out_root / f"{run_name}_eval"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Skip if evaluation already exists
            summary_json = eval_dir / "metrics" / "summary.json"
            if summary_json.exists():
                print(f"[skip] {run_name}: existing {summary_json}")
                continue

            # 1) Generation
            gen_cmd = [
                sys.executable,
                str(gen_script),
                "--data-root",
                str(args.data_root),
                "--out-dir",
                str(run_dir),
                "--datalist",
                str(args.datalist),
                "--ref-header",
                str(args.ref_header),
                "--n-segments",
                str(nseg),
                "--compactness",
                str(c),
                "--sigma",
                str(s),
                "--mode",
                str(mode),
                "--grad-sigma",
                str(args.grad_sigma),
                "--assign-all-from-gt",
                "--ignore-class",
                str(args.ignore_class),
            ]
            print("RUN:", " ".join(gen_cmd))
            if not args.dry_run:
                with open(run_dir / "gen.log", "a", buffering=1) as lf:
                    proc = subprocess.run(gen_cmd, stdout=lf, stderr=subprocess.STDOUT)
                    if proc.returncode != 0:
                        print(f"[error] Generation failed for {run_name}", file=sys.stderr)
                        sys.exit(proc.returncode)

            # 2) Evaluation
            eval_cmd = [
                sys.executable,
                str(eval_script),
                "--sv-dir",
                str(run_dir),
                "--sv-ids-dir",
                str(run_dir),
                "--datalist",
                str(args.datalist),
                "--data-root",
                str(args.data_root),
                "--output_dir",
                str(eval_dir),
                "--ignore-class",
                str(args.ignore_class),
                "--num_workers",
                str(args.eval_workers),
                "--progress",
                "--log_to_file",
            ]
            print("RUN:", " ".join(eval_cmd))
            if not args.dry_run:
                proc = subprocess.run(eval_cmd)
                if proc.returncode != 0:
                    print(f"[error] Evaluation failed for {run_name}", file=sys.stderr)
                    sys.exit(proc.returncode)

            # 3) Append to sweep summary
            avg_dice = float("nan")
            avg_iou = float("nan")
            if summary_json.exists():
                try:
                    j = json.loads(summary_json.read_text())
                    avg = j.get("average", {})
                    avg_dice = float(avg.get("dice", float("nan")))
                    avg_iou = float(avg.get("iou", float("nan")))
                except Exception:
                    pass
            _append_summary(
                summary_csv,
                {
                    "mode": mode,
                    "n_segments": nseg,
                    "compactness": c,
                    "sigma": s,
                    "grad_sigma": args.grad_sigma,
                    "run_dir": str(run_dir),
                    "eval_dir": str(eval_dir),
                    "avg_dice": f"{avg_dice:.6f}" if avg_dice == avg_dice else "nan",
                    "avg_iou": f"{avg_iou:.6f}" if avg_iou == avg_iou else "nan",
                },
            )
            print(f"[done] {run_name}: avg_dice={avg_dice} avg_iou={avg_iou}")


if __name__ == "__main__":
    main()
