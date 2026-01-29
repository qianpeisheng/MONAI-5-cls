#!/usr/bin/env python3
"""
Build an ensemble (majority-vote) pseudo-label set from multiple Graph-LP label dirs.

Outputs are written in a Graph-LP-like structure:
  <out_dir>/
    labels/<id>_labels.npy
    agreement/<id>_agree_count.npy      (per-voxel max vote count; GT tier encoded as gt_sentinel)
    source_masks/<id>_source.npy        (optional; copied/pass-through seed-support mask)
    confidence/<id>_maxc.npy            (optional alias of agreement/<id>_agree_count.npy; legacy)
    propagation_summary.json            (describes how this ensemble was built)

Notes:
  - This script does NOT require running Graph LP again; it just combines saved labels.
  - Reliability thresholding should be done in the training code (via agree_count weighting),
    not in this builder. The builder records raw agreement counts.

Example (vote 4 runs, tie-break by Q, and mark voxels with maxc>=3 as reliable):
  python3 scripts/build_graph_lp_ensemble_labels.py --datalist datalist_train_new.json --out_dir runs/graph_lp_ensemble_vote_C_O_M_Q_tieQ --label_dir C=runs/graph_lp_prop_0p1pct_k10_a0.9_new_n12000 --label_dir O=runs/graph_lp_prop_0p1pct_k10_a0.9_new_n12000_outerbg_adaptive --label_dir M=runs/graph_lp_3desc_eval/graph_lp_3desc_20260109-162446_k10_a0.9_sigPhimedian/moments --label_dir Q=runs/graph_lp_3desc_eval/graph_lp_3desc_20260109-162446_k10_a0.9_sigPhimedian/quantiles16 --tie_break Q --seed_source_mask_dir runs/graph_lp_3desc_eval/graph_lp_3desc_20260109-162446_k10_a0.9_sigPhimedian/quantiles16/source_masks
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


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
    if (path / "labels").is_dir():
        return path / "labels"
    return path


def _load_case_ids(datalist: Path) -> List[str]:
    records = json.loads(datalist.read_text())
    ids = [str(r.get("id")) for r in records if r.get("id")]
    return sorted(set(ids))


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


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datalist", type=str, required=True, help="Datalist JSON used to enumerate case IDs")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--label_dir", action="append", required=True, help="NAME=PATH (run dir or labels dir). Repeatable.")
    ap.add_argument("--tie_break", type=str, required=True, help="Label NAME used for tie-breaking")
    ap.add_argument("--seed_source_mask_dir", type=str, default="", help="Optional dir with <id>_source.npy to OR into output source masks")
    ap.add_argument("--gt_sentinel", type=int, default=255, help="agree_count value reserved for GT-supported voxels (default: 255)")
    ap.add_argument("--write_source_masks", action="store_true", help="Write out_dir/source_masks/<id>_source.npy")
    ap.add_argument(
        "--write_confidence_maps",
        action="store_true",
        help="[Legacy] Also write out_dir/confidence/<id>_maxc.npy as an alias of agreement/<id>_agree_count.npy.",
    )
    args = ap.parse_args(argv)

    datalist = Path(args.datalist)
    out_dir = Path(args.out_dir)
    label_dirs_raw = _parse_name_path_pairs(args.label_dir)
    label_dirs = {k: _resolve_label_dir(v) for k, v in label_dirs_raw.items()}

    if args.tie_break not in label_dirs:
        raise SystemExit(f"--tie_break {args.tie_break!r} not found among label dirs: {sorted(label_dirs.keys())}")

    case_ids = _load_case_ids(datalist)
    if not case_ids:
        raise SystemExit(f"No ids found in {datalist}")

    labels_out = out_dir / "labels"
    labels_out.mkdir(parents=True, exist_ok=True)
    src_out = out_dir / "source_masks"
    agree_out = out_dir / "agreement"
    agree_out.mkdir(parents=True, exist_ok=True)
    conf_out = out_dir / "confidence"

    if args.write_confidence_maps:
        conf_out.mkdir(parents=True, exist_ok=True)
    if args.write_source_masks:
        src_out.mkdir(parents=True, exist_ok=True)

    seed_src_dir = Path(args.seed_source_mask_dir) if args.seed_source_mask_dir else None
    gt_sentinel = int(args.gt_sentinel)

    # Build summary for provenance
    K = len(label_dirs)
    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "datalist": str(datalist),
        "label_dirs": {k: str(v) for k, v in label_dirs_raw.items()},
        "K": int(K),
        "tie_break": args.tie_break,
        "seed_source_mask_dir": str(seed_src_dir) if seed_src_dir is not None else "",
        "gt_sentinel": int(gt_sentinel),
        "n_cases": len(case_ids),
    }
    (out_dir / "propagation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Cases: {len(case_ids)}")
    print(f"Out: {out_dir}")
    print(f"Labels: {labels_out}")
    print(f"Agreement: {agree_out}")
    if args.write_source_masks:
        print(f"Source masks: {src_out}")
    if args.write_confidence_maps:
        print(f"Confidence: {conf_out}")

    for i, cid in enumerate(case_ids):
        # Load all label arrays for this case
        arrays: Dict[str, np.ndarray] = {}
        target_shape = None
        for name, ldir in label_dirs.items():
            p = ldir / f"{cid}_labels.npy"
            if not p.exists():
                raise FileNotFoundError(f"Missing label: {p}")
            a = np.load(str(p))
            if a.ndim == 4 and a.shape[0] == 1:
                a = a[0]
            a = a.astype(np.int64, copy=False)
            if target_shape is None:
                target_shape = tuple(a.shape)
            else:
                if tuple(a.shape) != target_shape:
                    a = _center_pad_or_crop(a, target_shape)  # type: ignore[arg-type]
            arrays[name] = a

        assert target_shape is not None
        tie = arrays[args.tie_break]
        vote, maxc = _majority_vote([arrays[k] for k in label_dirs.keys()], tie_break=tie)

        np.save(str(labels_out / f"{cid}_labels.npy"), vote.astype(np.int16, copy=False))

        # Encode GT-supported voxels (if provided) as a special high-count sentinel.
        agree_count = maxc.astype(np.uint8, copy=False)
        if seed_src_dir is not None:
            sp = seed_src_dir / f"{cid}_source.npy"
            if not sp.exists():
                raise FileNotFoundError(f"Missing seed source mask: {sp}")
            s = np.load(str(sp))
            if s.ndim == 4 and s.shape[0] == 1:
                s = s[0]
            if tuple(s.shape) != target_shape:
                s = _center_pad_or_crop(s.astype(np.uint8), target_shape)  # type: ignore[arg-type]
            agree_count = agree_count.copy()
            agree_count[s.astype(bool)] = np.uint8(gt_sentinel)

        np.save(str(agree_out / f"{cid}_agree_count.npy"), agree_count.astype(np.uint8, copy=False))

        # Optional legacy alias for older scripts.
        if args.write_confidence_maps:
            np.save(str(conf_out / f"{cid}_maxc.npy"), agree_count.astype(np.uint8, copy=False))

        if args.write_source_masks:
            src = np.zeros(target_shape, dtype=np.uint8)
            if seed_src_dir is not None:
                sp = seed_src_dir / f"{cid}_source.npy"
                if not sp.exists():
                    raise FileNotFoundError(f"Missing seed source mask: {sp}")
                s = np.load(str(sp))
                if s.ndim == 4 and s.shape[0] == 1:
                    s = s[0]
                if tuple(s.shape) != target_shape:
                    s = _center_pad_or_crop(s.astype(np.uint8), target_shape)  # type: ignore[arg-type]
                src |= (s.astype(bool)).astype(np.uint8)
            np.save(str(src_out / f"{cid}_source.npy"), src.astype(np.uint8, copy=False))

        if (i + 1) % 200 == 0:
            print(f"  processed {i+1}/{len(case_ids)}")

    print("DONE")


if __name__ == "__main__":
    main()
