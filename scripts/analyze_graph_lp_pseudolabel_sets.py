#!/usr/bin/env python3
"""
Analyze and compare Graph-LP pseudo-label sets.

This is a helper script for investigating why some pseudo-label sets that score
better vs train GT may (or may not) translate into better trained-model Dice.

It supports:
  - Quick summaries from existing `eval/metrics/per_case.csv` (if present)
  - Deep analysis requiring GT + `source_masks/<id>_source.npy`:
      * seeded vs graph-only region metrics
      * agreement vs disagreement quality
      * simple majority-vote ensembles + confidence analysis

Run the preset used in the report:
  python3 scripts/analyze_graph_lp_pseudolabel_sets.py --preset wp5_graphlp_intensity_vs_coords
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PRESET_WP5_GRAPHLP_INTENSITY_VS_COORDS = {
    "datalist": "datalist_train_new.json",
    "ignore_label": 6,
    "label_dirs": {
        "C": "runs/graph_lp_prop_0p1pct_k10_a0.9_new_n12000",
        "O": "runs/graph_lp_prop_0p1pct_k10_a0.9_new_n12000_outerbg_adaptive",
        "M": "runs/graph_lp_3desc_eval/graph_lp_3desc_20260109-162446_k10_a0.9_sigPhimedian/moments",
        "Q": "runs/graph_lp_3desc_eval/graph_lp_3desc_20260109-162446_k10_a0.9_sigPhimedian/quantiles16",
        "H": "runs/graph_lp_3desc_eval/graph_lp_3desc_20260109-162446_k10_a0.9_sigPhimedian/hist32",
    },
    "source_mask_dir": "runs/graph_lp_3desc_eval/graph_lp_3desc_20260109-162446_k10_a0.9_sigPhimedian/quantiles16/source_masks",
    "compare": ("C", "Q"),
    "ensemble": ["C,M,Q", "C,O,M,Q"],
    "tie_break": "Q",
}


def _parse_name_path_pairs(items: Sequence[str]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Expected NAME=PATH, got: {it!r}")
        name, path = it.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name:
            raise ValueError(f"Empty NAME in {it!r}")
        if name in out:
            raise ValueError(f"Duplicate NAME: {name}")
        out[name] = Path(path)
    return out


def _resolve_label_dir(path: Path) -> Path:
    # Allow passing either a run dir containing labels/ or the labels/ dir itself.
    if (path / "labels").is_dir():
        return path / "labels"
    return path


def _resolve_per_case_csv(run_or_labels: Path) -> Optional[Path]:
    # If user passes run dir, prefer run/eval/metrics/per_case.csv
    cand = run_or_labels / "eval" / "metrics" / "per_case.csv"
    if cand.exists():
        return cand
    # If user passes labels/ dir, look at parent
    if run_or_labels.name == "labels":
        cand2 = run_or_labels.parent / "eval" / "metrics" / "per_case.csv"
        if cand2.exists():
            return cand2
    return None


def _load_datalist_id_to_label(datalist: Path) -> Dict[str, str]:
    records = json.loads(datalist.read_text())
    out = {}
    for r in records:
        cid = r.get("id")
        lp = r.get("label")
        if cid and lp:
            out[str(cid)] = str(lp)
    return out


def _load_nii_ras_int(path: Path) -> np.ndarray:
    try:
        import nibabel as nib  # type: ignore
    except Exception as e:
        raise RuntimeError(f"nibabel is required to read NIfTI: {e}")
    img = nib.load(str(path))
    try:
        img = nib.as_closest_canonical(img)
    except Exception:
        pass
    arr = np.asarray(img.get_fdata())
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr.astype(np.int64)


def _center_pad_or_crop(vol: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    out = vol
    # crop
    for axis in range(3):
        diff = out.shape[axis] - target_shape[axis]
        if diff > 0:
            start = diff // 2
            end = start + target_shape[axis]
            slicer = [slice(None)] * out.ndim
            slicer[axis] = slice(start, end)
            out = out[tuple(slicer)]
    # pad
    pad_width = []
    for axis in range(3):
        diff = target_shape[axis] - out.shape[axis]
        if diff > 0:
            left = diff // 2
            right = diff - left
            pad_width.append((left, right))
        else:
            pad_width.append((0, 0))
    return np.pad(out, pad_width, mode="constant", constant_values=0)


def _dice_from_counts(pred_count: int, gt_count: int, inter: int) -> float:
    if pred_count + gt_count == 0:
        return 1.0
    return float(2.0 * inter / (pred_count + gt_count))


def _avg_dice_from_arrays(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Tuple[float, Tuple[float, float, float, float, float]]:
    """Returns (avg_dice_0_4, per_class_dice_tuple). both-empty => 1.0."""
    p = pred[mask].astype(np.int64, copy=False)
    g = gt[mask].astype(np.int64, copy=False)
    # counts
    pc = np.bincount(p, minlength=5)[:5]
    gc = np.bincount(g, minlength=5)[:5]
    pair = p * 5 + g
    mat = np.bincount(pair, minlength=25).reshape(5, 5)
    inter = np.diag(mat).astype(np.int64)
    dices = [_dice_from_counts(int(pc[i]), int(gc[i]), int(inter[i])) for i in range(5)]
    return float(np.mean(dices)), (float(dices[0]), float(dices[1]), float(dices[2]), float(dices[3]), float(dices[4]))


def _majority_vote(labels: Sequence[np.ndarray], tie_break: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      vote: int64 labels (0..4) with argmax ties replaced by tie_break
      maxc: uint8 max vote count per voxel
    """
    stack = np.stack(labels, axis=0)
    counts = np.zeros((5,) + stack.shape[1:], dtype=np.uint8)
    for c in range(5):
        counts[c] = np.sum(stack == c, axis=0)
    maxc = counts.max(axis=0)
    ties = (counts == maxc).sum(axis=0) > 1
    vote = counts.argmax(axis=0).astype(np.int64)
    if np.any(ties):
        vote[ties] = tie_break[ties]
    return vote, maxc.astype(np.uint8)


@dataclass
class Acc:
    n: int = 0
    sum_avg_dice: float = 0.0
    sum_per_class_dice: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))
    sum_vox_acc: float = 0.0
    sum_cov: float = 0.0

    def add(self, avg_dice: float, per_class: Sequence[float], vox_acc: float, cov: float) -> None:
        self.n += 1
        self.sum_avg_dice += float(avg_dice)
        self.sum_per_class_dice += np.asarray(per_class, dtype=np.float64)
        self.sum_vox_acc += float(vox_acc)
        self.sum_cov += float(cov)

    def summary(self) -> Tuple[float, List[float], float, float]:
        if self.n == 0:
            return float("nan"), [float("nan")] * 5, float("nan"), float("nan")
        return (
            self.sum_avg_dice / self.n,
            (self.sum_per_class_dice / self.n).tolist(),
            self.sum_vox_acc / self.n,
            self.sum_cov / self.n,
        )


def _print_per_case_csv_stats(name: str, csv_path: Path) -> None:
    import pandas as pd  # type: ignore

    df = pd.read_csv(csv_path)
    dice_cols = [f"dice_{i}" for i in range(5)]
    for c in dice_cols:
        if c not in df.columns:
            raise RuntimeError(f"{csv_path} missing column {c!r}")
    df["dice_mean"] = df[dice_cols].mean(axis=1)
    q = df["dice_mean"].quantile([0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]).to_dict()
    worst = df.nsmallest(5, "dice_mean")[["id", "dice_mean"]]
    print(f"[{name}] {csv_path}")
    print(f"  mean={df['dice_mean'].mean():.6f} std={df['dice_mean'].std():.6f}")
    print(f"  min={q[0]:.6f} p01={q[0.01]:.6f} p05={q[0.05]:.6f} p50={q[0.5]:.6f} p95={q[0.95]:.6f} max={q[1]:.6f}")
    print("  worst5:")
    for _, r in worst.iterrows():
        print(f"    {r['id']}: {r['dice_mean']:.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", type=str, default="", help="wp5_graphlp_intensity_vs_coords")
    ap.add_argument("--datalist", type=str, default="")
    ap.add_argument("--ignore_label", type=int, default=6)
    ap.add_argument("--label_dir", action="append", default=[], help="NAME=PATH (run dir or labels dir). Repeatable.")
    ap.add_argument("--source_mask_dir", type=str, default="", help="Directory with <id>_source.npy (seed-supported mask).")
    ap.add_argument("--compare", nargs=2, default=[], metavar=("A", "B"), help="Compare A vs B (B-A).")
    ap.add_argument("--ensemble", action="append", default=[], help="Comma-separated names to majority-vote (repeatable).")
    ap.add_argument("--tie_break", type=str, default="", help="Label name used for tie-break for ensembles.")
    ap.add_argument("--deep", action="store_true", help="Compute GT + source-mask deep stats (slower).")
    ap.add_argument("--max_cases", type=int, default=0, help="Limit cases for quick runs (0=all).")
    args = ap.parse_args()

    if args.preset:
        if args.preset != "wp5_graphlp_intensity_vs_coords":
            raise SystemExit(f"Unknown preset: {args.preset}")
        args.datalist = PRESET_WP5_GRAPHLP_INTENSITY_VS_COORDS["datalist"]
        args.ignore_label = PRESET_WP5_GRAPHLP_INTENSITY_VS_COORDS["ignore_label"]
        args.label_dir = [f"{k}={v}" for k, v in PRESET_WP5_GRAPHLP_INTENSITY_VS_COORDS["label_dirs"].items()]
        args.source_mask_dir = PRESET_WP5_GRAPHLP_INTENSITY_VS_COORDS["source_mask_dir"]
        args.compare = list(PRESET_WP5_GRAPHLP_INTENSITY_VS_COORDS["compare"])
        args.ensemble = list(PRESET_WP5_GRAPHLP_INTENSITY_VS_COORDS["ensemble"])
        args.tie_break = PRESET_WP5_GRAPHLP_INTENSITY_VS_COORDS["tie_break"]
        args.deep = True

    if not args.datalist:
        raise SystemExit("Missing --datalist (or use --preset).")

    label_dirs_raw = _parse_name_path_pairs(args.label_dir)
    label_dirs = {k: _resolve_label_dir(v) for k, v in label_dirs_raw.items()}
    if len(label_dirs) < 2:
        raise SystemExit("Provide at least two --label_dir entries.")

    datalist = Path(args.datalist)
    id_to_gt_label = _load_datalist_id_to_label(datalist)
    case_ids = sorted(id_to_gt_label.keys())
    if args.max_cases and args.max_cases > 0:
        case_ids = case_ids[: int(args.max_cases)]

    print(f"Cases: {len(case_ids)} (from {datalist})")
    print("Label dirs:")
    for name, path in label_dirs.items():
        print(f"  {name}: {path}")

    print("\n=== per_case.csv summaries (if available) ===")
    name_to_csv: Dict[str, Path] = {}
    for name, raw in label_dirs_raw.items():
        csvp = _resolve_per_case_csv(raw)
        if csvp is None:
            continue
        name_to_csv[name] = csvp
        _print_per_case_csv_stats(name, csvp)

    if args.compare and args.compare[0] in name_to_csv and args.compare[1] in name_to_csv:
        import pandas as pd  # type: ignore

        a, b = args.compare
        da = pd.read_csv(name_to_csv[a])
        db = pd.read_csv(name_to_csv[b])
        da["dice_mean_a"] = da[[f"dice_{i}" for i in range(5)]].mean(axis=1)
        db["dice_mean_b"] = db[[f"dice_{i}" for i in range(5)]].mean(axis=1)
        m = da[["id", "dice_mean_a"]].merge(db[["id", "dice_mean_b"]], on="id", how="inner")
        m["delta"] = m["dice_mean_b"] - m["dice_mean_a"]
        print(f"\n=== per-case delta from CSV: {b} - {a} ===")
        print(f"  mean={m['delta'].mean():.6f} median={m['delta'].median():.6f}")
        print("  Top + improvements:")
        for _, r in m.sort_values("delta", ascending=False).head(10).iterrows():
            print(f"    {r['id']}: {r['delta']:+.6f}")
        print("  Top - regressions:")
        for _, r in m.sort_values("delta", ascending=True).head(10).iterrows():
            print(f"    {r['id']}: {r['delta']:+.6f}")

    if not args.deep:
        return

    if not args.source_mask_dir:
        raise SystemExit("--deep requires --source_mask_dir (or use --preset).")
    src_dir = Path(args.source_mask_dir)
    ignore_label = int(args.ignore_label)

    regions = ["all", "graph_only", "seeded"]
    acc: Dict[str, Dict[str, Acc]] = {r: {name: Acc() for name in label_dirs} for r in regions}

    # Agreement stats for compare pair
    a_name, b_name = (args.compare[0], args.compare[1]) if args.compare else ("", "")
    agree_fracs: List[float] = []
    b_dice_agree: List[float] = []
    b_dice_disagree: List[float] = []

    # Ensemble accumulators
    ensembles: List[List[str]] = []
    for spec in args.ensemble:
        keys = [k.strip() for k in spec.split(",") if k.strip()]
        if keys:
            ensembles.append(keys)
    tie_break_name = args.tie_break.strip()
    ens_acc: Dict[str, Acc] = {",".join(keys): Acc() for keys in ensembles}
    ens_conf_cov: Dict[str, Dict[int, List[float]]] = {",".join(keys): {i: [] for i in range(1, len(keys) + 1)} for keys in ensembles}
    ens_conf_dice: Dict[str, Dict[int, List[float]]] = {",".join(keys): {i: [] for i in range(1, len(keys) + 1)} for keys in ensembles}

    print("\n=== Deep analysis (GT + source masks) ===")
    print(f"ignore_label={ignore_label}")
    print(f"source_mask_dir={src_dir}")

    for idx, cid in enumerate(case_ids):
        gt = _load_nii_ras_int(Path(id_to_gt_label[cid]))
        base = gt != ignore_label
        base_n = int(base.sum())
        if base_n == 0:
            continue

        src = np.load(str(src_dir / f"{cid}_source.npy")).astype(bool)
        if src.shape != gt.shape:
            src = _center_pad_or_crop(src.astype(np.uint8), tuple(gt.shape)).astype(bool)  # type: ignore[arg-type]

        graph = base & (~src)
        seeded = base & (src)

        masks = {
            "all": base,
            "graph_only": graph,
            "seeded": seeded,
        }

        # Load label arrays once per label set
        preds: Dict[str, np.ndarray] = {}
        for name, ldir in label_dirs.items():
            arr = np.load(str(ldir / f"{cid}_labels.npy")).astype(np.int64)
            if arr.shape != gt.shape:
                arr = _center_pad_or_crop(arr, tuple(gt.shape))  # type: ignore[arg-type]
            preds[name] = arr

        # Region metrics per label set
        for region, m in masks.items():
            cov = float(m.sum()) / float(base_n)
            if int(m.sum()) == 0:
                # coverage contributes; metrics undefined -> set 0
                for name in label_dirs:
                    acc[region][name].add(avg_dice=0.0, per_class=[0.0] * 5, vox_acc=0.0, cov=cov)
                continue

            gt_m = gt[m]
            for name in label_dirs:
                pr = preds[name]
                pr_m = pr[m]
                vox_acc = float(np.mean(pr_m == gt_m))
                avg_dice, per = _avg_dice_from_arrays(pr, gt, m)
                acc[region][name].add(avg_dice=avg_dice, per_class=per, vox_acc=vox_acc, cov=cov)

        # Agreement analysis on graph-only voxels for compare pair (if requested)
        if a_name and b_name and a_name in preds and b_name in preds and int(graph.sum()) > 0:
            pa = preds[a_name]
            pb = preds[b_name]
            agree = graph & (pa == pb)
            disagree = graph & (pa != pb)
            agree_fracs.append(float(agree.sum()) / float(graph.sum()))
            b_dice_agree.append(_avg_dice_from_arrays(pb, gt, agree)[0] if int(agree.sum()) > 0 else 0.0)
            b_dice_disagree.append(_avg_dice_from_arrays(pb, gt, disagree)[0] if int(disagree.sum()) > 0 else 0.0)

        # Ensembles
        for keys in ensembles:
            key = ",".join(keys)
            if tie_break_name not in preds:
                continue
            lbls = [preds[k] for k in keys]
            vote, maxc = _majority_vote(lbls, tie_break=preds[tie_break_name])
            vox_acc = float(np.mean(vote[base] == gt[base]))
            avg_dice, per = _avg_dice_from_arrays(vote, gt, base)
            ens_acc[key].add(avg_dice=avg_dice, per_class=per, vox_acc=vox_acc, cov=1.0)

            if int(graph.sum()) > 0:
                gcount = int(graph.sum())
                for mc in range(1, len(keys) + 1):
                    mk = graph & (maxc == mc)
                    ens_conf_cov[key][mc].append(float(mk.sum()) / float(gcount))
                    ens_conf_dice[key][mc].append(_avg_dice_from_arrays(vote, gt, mk)[0] if int(mk.sum()) > 0 else 0.0)

        if (idx + 1) % 200 == 0:
            print(f"  processed {idx+1}/{len(case_ids)}")

    # Print region summaries
    for region in regions:
        print(f"\n[REGION {region}]")
        for name in label_dirs:
            avg_dice, per, vox_acc, cov = acc[region][name].summary()
            pcs = ", ".join(f"{x:.4f}" for x in per)
            print(f"  {name}: avg_dice={avg_dice:.6f} per_class=[{pcs}] vox_acc={vox_acc:.6f} cov={cov:.6f}")

    # Print agreement summaries
    if agree_fracs:
        af = np.asarray(agree_fracs, dtype=np.float64)
        da = np.asarray(b_dice_agree, dtype=np.float64)
        dd = np.asarray(b_dice_disagree, dtype=np.float64)
        print(f"\n=== Agreement on graph-only voxels ({a_name} vs {b_name}) ===")
        print(
            f"  agree_frac: mean={af.mean():.6f} median={np.median(af):.6f} "
            f"p10={np.quantile(af,0.1):.6f} p90={np.quantile(af,0.9):.6f}"
        )
        print(f"  {b_name} dice on agree region: mean={da.mean():.6f} median={np.median(da):.6f}")
        print(f"  {b_name} dice on disagree region: mean={dd.mean():.6f} median={np.median(dd):.6f}")

    # Print ensemble summaries
    for keys in ensembles:
        key = ",".join(keys)
        avg_dice, per, vox_acc, _ = ens_acc[key].summary()
        pcs = ", ".join(f"{x:.4f}" for x in per)
        print(f"\n=== Ensemble vote({key}) tie_break={tie_break_name} ===")
        print(f"  avg_dice={avg_dice:.6f} per_class=[{pcs}] vox_acc={vox_acc:.6f}")
        if ens_conf_cov[key][1]:
            print("  confidence on graph-only voxels (max vote count -> cov_mean / dice_mean):")
            for mc in range(len(keys), 0, -1):
                covm = float(np.mean(ens_conf_cov[key][mc]))
                dicem = float(np.mean(ens_conf_dice[key][mc]))
                print(f"    maxc={mc}: cov_mean={covm:.4f} dice_mean={dicem:.4f}")


if __name__ == "__main__":
    main()
