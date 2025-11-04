#!/usr/bin/env python3
"""
Select informative 2D slices from 3D WP5 training volumes across all three axes.

Key properties
- Counts total candidate slices as sum over axes for each volume: X + Y + Z (RAS oriented).
- Budget as percentage of ALL training slices (configurable, e.g., 0.01 for 1%).
- Scores slices using GT label priors by default: boundary density + foreground fraction + class rarity.
- Optionally enforces simple non-max suppression (NMS) per volume/axis and per-volume caps.
- Saves:
  - selected_slices.json: list of {id, axis, index, score}
  - per-volume candidate stats under candidates/<id>_slice_stats.npz
  - optional static sup masks (1,X,Y,Z boolean) under sup_masks/<id>_supmask.npy

Usage
  python scripts/select_informative_slices.py \
    --train_list datalist_train.json \
    --out_dir runs/slice_selection_20251009 \
    --percent 0.01 --per_volume_cap 12 --nms 3 --save_sup_masks

Notes
- Axis names: 'x' (sagittal), 'y' (coronal), 'z' (axial). Indices are 0..(len-1).
- Sup masks mark the entire selected planes as supervised (True); other voxels are False.
- For training, enable few-slices static mode by passing --fewshot_mode few_slices --fewshot_static --save_sup_masks in train_finetune_wp5.py and point save dir to the produced sup_masks directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _load_label_volume(path: str) -> np.ndarray:
    """Load label volume as channel-first (1,X,Y,Z) int64, RAS oriented.
    Uses MONAI dict transforms to ensure consistent orientation with training.
    """
    from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd

    data = {"label": path}
    t = Compose([
        LoadImaged(keys=["label"]),
        EnsureChannelFirstd(keys=["label"]),
        Orientationd(keys=["label"], axcodes="RAS"),
    ])
    d = t(data)
    arr = d["label"].astype(np.int64)
    return arr  # (1,X,Y,Z)


def _boundary_2d(mask2d: np.ndarray) -> np.ndarray:
    """Compute a simple 2D boundary map (4-neighborhood morphological gradient)."""
    # mask2d: HxW boolean
    m = mask2d
    up = np.zeros_like(m); up[1:, :] = m[:-1, :]
    down = np.zeros_like(m); down[:-1, :] = m[1:, :]
    left = np.zeros_like(m); left[:, 1:] = m[:, :-1]
    right = np.zeros_like(m); right[:, :-1] = m[:, 1:]
    neigh_any = up | down | left | right
    boundary = m & (~(up & down & left & right))
    # Fallback: also include edge of connected components where neighborhood changes
    boundary |= (m & (~neigh_any))
    return boundary


def _compute_slice_stats(lbl: np.ndarray) -> Dict[str, dict]:
    """Compute per-slice stats for all axes using labels.

    Returns a dict:
      {
        'x': {'len': X, 'fg_frac': [..], 'bnd_frac': [..], 'classes': [[0/1,..], ...]},
        'y': {...},
        'z': {...},
      }
    where each list has length equal to axis length.
    """
    assert lbl.ndim == 4 and lbl.shape[0] == 1
    _, X, Y, Z = lbl.shape
    out = {}
    fg3d = (lbl != 0) & (lbl != 6)

    # Pre-compute class presence maps (per class 1..4)
    class_maps = {c: (lbl == c) for c in [1, 2, 3, 4]}

    def stats_along_axis(axis: int, name: str) -> dict:
        if axis == 0:  # X
            length = X
            fg_frac = []
            bnd_frac = []
            classes = []
            for i in range(X):
                sl = fg3d[0, i, :, :]
                if sl.size == 0:
                    fg_frac.append(0.0)
                    bnd_frac.append(0.0)
                    classes.append([0, 0, 0, 0])
                    continue
                fg_f = float(sl.mean())
                bndry = _boundary_2d(sl)
                b_f = float(bndry.mean())
                pres = [int(class_maps[c][0, i, :, :].any()) for c in [1, 2, 3, 4]]
                fg_frac.append(fg_f)
                bnd_frac.append(b_f)
                classes.append(pres)
        elif axis == 1:  # Y
            length = Y
            fg_frac = []
            bnd_frac = []
            classes = []
            for i in range(Y):
                sl = fg3d[0, :, i, :]
                fg_f = float(sl.mean()) if sl.size > 0 else 0.0
                bndry = _boundary_2d(sl)
                b_f = float(bndry.mean())
                pres = [int(class_maps[c][0, :, i, :].any()) for c in [1, 2, 3, 4]]
                fg_frac.append(fg_f)
                bnd_frac.append(b_f)
                classes.append(pres)
        else:  # Z
            length = Z
            fg_frac = []
            bnd_frac = []
            classes = []
            for i in range(Z):
                sl = fg3d[0, :, :, i]
                fg_f = float(sl.mean()) if sl.size > 0 else 0.0
                bndry = _boundary_2d(sl)
                b_f = float(bndry.mean())
                pres = [int(class_maps[c][0, :, :, i].any()) for c in [1, 2, 3, 4]]
                fg_frac.append(fg_f)
                bnd_frac.append(b_f)
                classes.append(pres)
        # background-present list (0 if slice has no background voxel; 1 otherwise)
        bg_present = []
        if axis == 0:
            for i in range(X):
                bg_present.append(int((lbl[0, i, :, :] == 0).any()))
        elif axis == 1:
            for i in range(Y):
                bg_present.append(int((lbl[0, :, i, :] == 0).any()))
        else:
            for i in range(Z):
                bg_present.append(int((lbl[0, :, :, i] == 0).any()))
        return {"len": int(length), "fg_frac": fg_frac, "bnd_frac": bnd_frac, "classes": classes, "bg": bg_present}

    out["x"] = stats_along_axis(0, "x")
    out["y"] = stats_along_axis(1, "y")
    out["z"] = stats_along_axis(2, "z")
    return out


def select_slices(
    train_list: List[Dict],
    out_dir: Path,
    percent: float,
    per_volume_cap: int | None,
    nms: int,
    w_boundary: float,
    w_fg: float,
    w_rarity: float,
    save_sup_masks: bool,
    target_class: int = 0,
    min_target_share: float = 0.0,
    target_boost: float = 0.0,
    selector: str = "score",
    seed: int = 42,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_dir = out_dir / "candidates"
    sup_dir = out_dir / "sup_masks"
    cand_dir.mkdir(parents=True, exist_ok=True)
    if save_sup_masks:
        sup_dir.mkdir(parents=True, exist_ok=True)

    # First pass: class frequencies and shapes
    shapes = {}
    class_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    per_case_stats = {}
    for rec in train_list:
        lbl = _load_label_volume(rec["label"])  # (1,X,Y,Z)
        _, X, Y, Z = lbl.shape
        shapes[rec["id"]] = (X, Y, Z)
        # Update class counts (ignore 0 and 6)
        for c in [1, 2, 3, 4]:
            class_counts[c] += int((lbl == c).sum())
        # Slice stats per axis
        st = _compute_slice_stats(lbl)
        per_case_stats[rec["id"]] = st
        # save candidates cache
        np.savez_compressed(
            cand_dir / f"{rec['id'].replace('/', '_')}_slice_stats.npz",
            x_len=st["x"]["len"], x_fg=np.asarray(st["x"]["fg_frac"], dtype=np.float32), x_bnd=np.asarray(st["x"]["bnd_frac"], dtype=np.float32), x_cls=np.asarray(st["x"]["classes"], dtype=np.int16),
            y_len=st["y"]["len"], y_fg=np.asarray(st["y"]["fg_frac"], dtype=np.float32), y_bnd=np.asarray(st["y"]["bnd_frac"], dtype=np.float32), y_cls=np.asarray(st["y"]["classes"], dtype=np.int16),
            z_len=st["z"]["len"], z_fg=np.asarray(st["z"]["fg_frac"], dtype=np.float32), z_bnd=np.asarray(st["z"]["bnd_frac"], dtype=np.float32), z_cls=np.asarray(st["z"]["classes"], dtype=np.int16),
        )

    # Class rarity weights (inverse sqrt frequency)
    total_fg = sum(class_counts.values())
    rarity = {c: (0.0 if class_counts[c] == 0 else float(1.0 / np.sqrt(class_counts[c] / max(1.0, total_fg)))) for c in [1, 2, 3, 4]}

    # Build candidates across all volumes and axes
    # Each entry: (id, axis, idx, score, has_target)
    candidates: List[Tuple[str, str, int, float, bool]] = []
    total_slices = 0
    for rec in train_list:
        cid = rec["id"]
        st = per_case_stats[cid]
        for axis in ["x", "y", "z"]:
            L = st[axis]["len"]
            total_slices += L
            if selector == "all_classes_random":
                cls_list = st[axis]["classes"]  # list of [1/0,...] for classes 1..4 (foreground only)
                for i in range(L):
                    pres = cls_list[i]
                    # Require all foreground classes 1..4 to be present at least once on the slice
                    if all(int(v) == 1 for v in pres):
                        candidates.append((cid, axis, i, 1.0, True))
            else:
                fg_list = st[axis]["fg_frac"]
                bnd_list = st[axis]["bnd_frac"]
                cls_list = st[axis]["classes"]
                for i in range(L):
                    fg_f = fg_list[i]
                    b_f = bnd_list[i]
                    pres = cls_list[i]
                    rarity_score = 0.0
                    for k, present in enumerate(pres, start=1):
                        if present:
                            rarity_score += rarity[k]
                    # Normalize rarity score by max possible (sum of all four classes)
                    max_r = sum(rarity.values()) if rarity.values() else 1.0
                    rarity_norm = rarity_score / max_r if max_r > 0 else 0.0
                    base = w_boundary * b_f + w_fg * fg_f + w_rarity * rarity_norm
                    has_target = False
                    if int(target_class) in (1, 2, 3, 4):
                        # classes list is [c1,c2,c3,c4]
                        idx_tc = int(target_class) - 1
                        if 0 <= idx_tc < len(pres):
                            has_target = bool(pres[idx_tc])
                            if has_target and target_boost > 1e-12:
                                base += float(target_boost)
                    score = float(base)
                    candidates.append((cid, axis, i, score, has_target))

    budget = int(np.ceil(max(1e-6, percent) * total_slices))
    budget = max(1, budget)

    selected = []
    picked_map: Dict[Tuple[str, str], List[int]] = {}
    per_vol_count: Dict[str, int] = {}

    # Optional phase 1: ensure a minimum share of target-class slices
    target_needed = int(np.ceil(float(min_target_share) * budget)) if min_target_share > 1e-12 else 0
    target_taken = 0

    if selector != "all_classes_random" and target_needed > 0 and int(target_class) in (1, 2, 3, 4):
        for cid, axis, idx, score, has_target in candidates:
            if target_taken >= target_needed:
                break
            if not has_target:
                continue
            if len(selected) >= budget:
                break
            if per_volume_cap is not None and per_vol_count.get(cid, 0) >= per_volume_cap:
                continue
            key = (cid, axis)
            taken = picked_map.get(key, [])
            if any(abs(idx - t) <= nms for t in taken):
                continue
            selected.append({"id": cid, "axis": axis, "index": int(idx), "score": float(score)})
            per_vol_count[cid] = per_vol_count.get(cid, 0) + 1
            taken.append(idx)
            picked_map[key] = taken
            target_taken += 1

    if selector == "all_classes_random":
        # Randomly sample from candidate set
        import random as _random
        _random.seed(int(seed))
        # Build flat list of unique (cid, axis, idx)
        uniq = [(cid, axis, int(idx)) for (cid, axis, idx, _s, _t) in candidates]
        # Optionally enforce per_volume_cap if provided
        if per_volume_cap is not None:
            # Shuffle and take while respecting per_volume_cap
            _random.shuffle(uniq)
            for cid, axis, idx in uniq:
                if len(selected) >= budget:
                    break
                if per_vol_count.get(cid, 0) >= per_volume_cap:
                    continue
                selected.append({"id": cid, "axis": axis, "index": int(idx), "score": 1.0})
                per_vol_count[cid] = per_vol_count.get(cid, 0) + 1
        else:
            _random.shuffle(uniq)
            for cid, axis, idx in uniq[: min(budget, len(uniq))]:
                selected.append({"id": cid, "axis": axis, "index": int(idx), "score": 1.0})
    else:
        # Sort candidates by score desc
        candidates.sort(key=lambda t: t[3], reverse=True)
        # Phase 2: fill remainder from all candidates (scored selection)
        selected_set = {(e["id"], e["axis"], int(e["index"])) for e in selected}
        for cid, axis, idx, score, _has_target in candidates:
            if len(selected) >= budget:
                break
            if (cid, axis, int(idx)) in selected_set:
                continue
            if per_volume_cap is not None and per_vol_count.get(cid, 0) >= per_volume_cap:
                continue
            key = (cid, axis)
            taken = picked_map.get(key, [])
            if any(abs(idx - t) <= nms for t in taken):
                continue
            selected.append({"id": cid, "axis": axis, "index": int(idx), "score": float(score)})
            per_vol_count[cid] = per_vol_count.get(cid, 0) + 1
            taken.append(idx)
            picked_map[key] = taken
            selected_set.add((cid, axis, int(idx)))

    # Persist selection and summary
    sel_path = out_dir / "selected_slices.json"
    sel_path.write_text(json.dumps(selected, indent=2))

    # Target selection stats
    target_selected = 0
    if int(target_class) in (1, 2, 3, 4):
        # Build quick lookup for per-case axis slice presence
        sel_map: Dict[str, Dict[str, set]] = {}
        for e in selected:
            sel_map.setdefault(e["id"], {}).setdefault(e["axis"], set()).add(int(e["index"]))
        for rec in train_list:
            cid = rec["id"]
            st = per_case_stats[cid]
            for axis in ["x", "y", "z"]:
                L = st[axis]["len"]
                pres = st[axis]["classes"]
                chosen = sel_map.get(cid, {}).get(axis, set())
                for i in chosen:
                    if 0 <= i < L:
                        # classes list is [1..4]
                        idx_tc = int(target_class) - 1
                        if 0 <= idx_tc < len(pres[i]) and int(pres[i][idx_tc]) == 1:
                            target_selected += 1

    summary = {
        "total_volumes": len(train_list),
        "total_slices_all_axes": int(total_slices),
        "percent": float(percent),
        "budget": int(budget),
        "selected": len(selected),
        "selector": selector,
        "seed": int(seed),
        "candidate_slices": int(len(candidates)),
        "classes_required": [0, 1, 2, 3, 4] if selector == "all_classes_random" else None,
        "per_volume_cap": int(per_volume_cap) if per_volume_cap is not None else None,
        "nms": int(nms),
        "weights": {"boundary": w_boundary, "fg": w_fg, "rarity": w_rarity},
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "rarity_weights": {str(k): float(v) for k, v in rarity.items()},
        "target_class": int(target_class),
        "min_target_share": float(min_target_share),
        "target_boost": float(target_boost),
        "target_selected": int(target_selected),
        "target_selected_share": (float(target_selected) / float(len(selected)) if len(selected) > 0 else 0.0),
    }
    (out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2))

    # Optional: export static sup masks per case
    if save_sup_masks:
        # Build a map id -> per-axis selected indices set
        by_id: Dict[str, Dict[str, set]] = {}
        for e in selected:
            by_id.setdefault(e["id"], {}).setdefault(e["axis"], set()).add(int(e["index"]))
        for rec in train_list:
            cid = rec["id"]
            lbl = _load_label_volume(rec["label"])  # (1,X,Y,Z)
            _, X, Y, Z = lbl.shape
            sup = np.zeros_like(lbl, dtype=bool)
            axes_sel = by_id.get(cid, {})
            if 'x' in axes_sel:
                for i in axes_sel['x']:
                    if 0 <= i < X:
                        sup[0, i, :, :] = True
            if 'y' in axes_sel:
                for i in axes_sel['y']:
                    if 0 <= i < Y:
                        sup[0, :, i, :] = True
            if 'z' in axes_sel:
                for i in axes_sel['z']:
                    if 0 <= i < Z:
                        sup[0, :, :, i] = True
            safe = cid.replace('/', '_')
            np.save(sup_dir / f"{safe}_supmask.npy", sup)

    return {"selected_path": str(sel_path), "summary": summary}


def main():
    ap = argparse.ArgumentParser(description="Select informative slices across all axes for few-shot labeling.")
    ap.add_argument("--train_list", type=str, required=True, help="Path to datalist_train.json")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for selection and caches")
    ap.add_argument("--percent", type=float, default=0.01, help="Fraction of ALL candidate slices to select (e.g., 0.01 for 1%)")
    ap.add_argument("--per_volume_cap", type=int, default=12, help="Max selected slices per volume across axes; -1 to disable")
    ap.add_argument("--nms", type=int, default=3, help="NMS window in slices per volume/axis (suppress within ±nms)")
    ap.add_argument("--w_boundary", type=float, default=0.5, help="Weight for boundary density in slice score")
    ap.add_argument("--w_fg", type=float, default=0.3, help="Weight for foreground fraction in slice score")
    ap.add_argument("--w_rarity", type=float, default=0.2, help="Weight for class rarity coverage in slice score")
    ap.add_argument("--save_sup_masks", action="store_true", help="Also save static sup masks per case under out_dir/sup_masks")
    ap.add_argument("--selector", type=str, default="score", choices=["score", "all_classes_random"], help="Slice selection strategy: 'score' (original) or 'all_classes_random' (require 0..4 present; random sample to budget)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for 'all_classes_random' selector")
    # Class emphasis options (default off)
    ap.add_argument("--target_class", type=int, default=0, help="Emphasize this class (1-4). 0 disables.")
    ap.add_argument("--min_target_share", type=float, default=0.0, help="Reserve at least this fraction of budget for target-class slices")
    ap.add_argument("--target_boost", type=float, default=0.0, help="Add this bonus to score if slice contains target class")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    train_list = json.loads(Path(args.train_list).read_text())
    per_volume_cap = None if args.per_volume_cap is None or args.per_volume_cap < 0 else int(args.per_volume_cap)

    res = select_slices(
        train_list=train_list,
        out_dir=out_dir,
        percent=float(args.percent),
        per_volume_cap=per_volume_cap,
        nms=int(args.nms),
        w_boundary=float(args.w_boundary),
        w_fg=float(args.w_fg),
        w_rarity=float(args.w_rarity),
        save_sup_masks=bool(args.save_sup_masks),
        target_class=int(args.target_class),
        min_target_share=float(args.min_target_share),
        target_boost=float(args.target_boost),
        selector=str(args.selector),
        seed=int(args.seed),
    )
    print(json.dumps(res["summary"], indent=2))
    print(f"Selected slices saved to: {res['selected_path']}")


if __name__ == "__main__":
    main()
