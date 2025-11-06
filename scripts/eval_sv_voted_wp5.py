#!/usr/bin/env python3
"""
Evaluate supervoxel-voted per-voxel labels against GT on WP5-like data.

Outputs per-case Dice/IoU (0..4, ignore GT==6; both-empty=1.0) and optional HD/ASD,
plus voting-specific diagnostics when supervoxel IDs are provided: per-SV purity,
normalized Shannon entropy (over classes 0..4), and entropy-error correlations.

Usage (train split, your folders):
  python3 scripts/eval_sv_voted_wp5.py \
    --sv-dir /home/peisheng/MONAI/runs/sv_fullgt_5k_ras2_voted \
    --sv-ids-dir /home/peisheng/MONAI/runs/sv_fullgt_5k_ras2 \
    --datalist datalist_train.json \
    --data-root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
    --output_dir /home/peisheng/MONAI/runs/sv_fullgt_5k_ras2_voted_eval \
    --ignore-class 6 --heavy --hd_percentile 95 --log_to_file

Notes
- SV-voted labels are expected at <sv-dir>/<id>_labels.npy (RAS, same canonical orientation used during training/viewing).
- If --sv-ids-dir is provided and contains <id>_sv_ids.npy, supervoxel diagnostics are computed.
- Evaluation semantics match train_finetune_wp5.compute_metrics: classes 0..4, ignore GT==6, both-empty=1.0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import concurrent.futures as _cf
import os as _os

import numpy as np
import torch

# Ensure project root is importable
import sys as _sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))

import train_finetune_wp5 as tfw  # reuse compute_metrics semantics


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Evaluate SV-voted labels vs GT on WP5-like data")
    p.add_argument("--sv-dir", type=str, required=True, help="Directory with <id>_labels.npy from voted supervoxels")
    p.add_argument("--sv-ids-dir", type=str, default="", help="Optional directory with <id>_sv_ids.npy for SV diagnostics")
    p.add_argument("--datalist", type=str, default="datalist_train.json", help="JSON with records of {image,label,id}")
    p.add_argument("--data-root", type=str, default="", help="Dataset root (not required if datalist has absolute paths)")
    p.add_argument("--output_dir", type=str, required=True, help="Folder to write metrics and logs (will be created)")
    p.add_argument("--ignore-class", type=int, default=6, help="Label to ignore for metrics (WP5: 6)")
    p.add_argument("--heavy", action="store_true", help="Also compute HD/ASD (slower)")
    p.add_argument("--hd_percentile", type=float, default=95.0, help="Hausdorff percentile: 95.0 for HD95, 100.0 for full HD")
    p.add_argument("--max_cases", type=int, default=-1, help="Limit number of cases for smoke tests")
    p.add_argument("--log_to_file", action="store_true", help="Tee stdout/stderr to <output_dir>/eval.log")
    p.add_argument("--log_file_name", type=str, default="eval.log", help="Name of log file inside output_dir")
    p.add_argument("--num_workers", type=int, default=1, help="Parallel workers over cases (1=serial)")
    p.add_argument("--progress", action="store_true", help="Show a progress bar over cases")
    return p


def _init_eval_logging(out_dir: Path, enable: bool = True, filename: str = "eval.log") -> None:
    if not enable:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    logp = out_dir / filename
    fh = open(logp, "a", buffering=1, encoding="utf-8")

    class Tee:
        def __init__(self, stream, mirror):
            self.stream = stream
            self.mirror = mirror
        def write(self, s):
            self.stream.write(s)
            self.mirror.write(s)
            return len(s)
        def flush(self):
            self.stream.flush()
            self.mirror.flush()

    import sys
    sys.stdout = Tee(sys.__stdout__, fh)  # type: ignore[assignment]
    sys.stderr = Tee(sys.__stderr__, fh)  # type: ignore[assignment]


def _load_json_list(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Datalist not found: {path}")
    return json.loads(path.read_text())


def _discover_sv_ids_available(sv_dir: Path, suffix: str) -> set[str]:
    ids: set[str] = set()
    for p in sv_dir.glob(f"*{suffix}"):
        name = p.name
        if name.endswith(suffix):
            ids.add(name[: -len(suffix)])
    return ids


def _load_nii_ras_int(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
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
    def _crop_to(v, ts):
        out = v
        for axis in range(3):
            diff = out.shape[axis] - ts[axis]
            if diff > 0:
                start = diff // 2
                end = start + ts[axis]
                slicer = [slice(None)] * out.ndim
                slicer[axis] = slice(start, end)
                out = out[tuple(slicer)]
        return out
    def _pad_to(v, ts):
        pad_width = []
        for axis in range(3):
            diff = ts[axis] - v.shape[axis]
            if diff > 0:
                left = diff // 2
                right = diff - left
                pad_width.append((left, right))
            else:
                pad_width.append((0, 0))
        return np.pad(v, pad_width, mode='constant', constant_values=0)
    out = _crop_to(vol, target_shape)
    out = _pad_to(out, target_shape)
    return out


def _to_5d(t: np.ndarray) -> torch.Tensor:
    # shape to (B=1,C=1,X,Y,Z)
    if t.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={tuple(t.shape)}")
    return torch.from_numpy(t[None, None, ...])


def _both_empty_policy_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    # Helper for sanity unit; not used for primary metrics (we reuse tfw.compute_metrics)
    inter = float(np.logical_and(pred_mask, gt_mask).sum())
    psum = float(pred_mask.sum())
    gsum = float(gt_mask.sum())
    union = float(np.logical_or(pred_mask, gt_mask).sum())
    both_empty = (psum + gsum) == 0.0
    dice = (2.0 * inter / (psum + gsum)) if not both_empty else 1.0
    iou = (inter / union) if (union > 0) else (1.0 if both_empty else 0.0)
    return dice, iou


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    # Return Pearson r over finite pairs; NaN if insufficient variance
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        return float("nan")
    vx = np.var(x)
    vy = np.var(y)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _update_running_corr(acc: Dict[str, float], x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> None:
    # Accumulate sums for correlation; unweighted if w is None, else weighted
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if w is not None:
        w = w[m].astype(np.float64)
        if w.size == 0 or float(w.sum()) <= 0:
            return
        acc["Sw"] = acc.get("Sw", 0.0) + float(w.sum())
        acc["Sx"] = acc.get("Sx", 0.0) + float((w * x).sum())
        acc["Sy"] = acc.get("Sy", 0.0) + float((w * y).sum())
        acc["Sxx"] = acc.get("Sxx", 0.0) + float((w * x * x).sum())
        acc["Syy"] = acc.get("Syy", 0.0) + float((w * y * y).sum())
        acc["Sxy"] = acc.get("Sxy", 0.0) + float((w * x * y).sum())
    else:
        n = float(x.size)
        acc["Sw"] = acc.get("Sw", 0.0) + n
        acc["Sx"] = acc.get("Sx", 0.0) + float(x.sum())
        acc["Sy"] = acc.get("Sy", 0.0) + float(y.sum())
        acc["Sxx"] = acc.get("Sxx", 0.0) + float((x * x).sum())
        acc["Syy"] = acc.get("Syy", 0.0) + float((y * y).sum())
        acc["Sxy"] = acc.get("Sxy", 0.0) + float((x * y).sum())


def _finalize_running_corr(acc: Dict[str, float]) -> float:
    Sw = acc.get("Sw", 0.0)
    if Sw <= 1.0:
        return float("nan")
    Sx, Sy = acc.get("Sx", 0.0), acc.get("Sy", 0.0)
    Sxx, Syy, Sxy = acc.get("Sxx", 0.0), acc.get("Syy", 0.0), acc.get("Sxy", 0.0)
    # Weighted Pearson correlation (falls back to unweighted when Sw==N and w=1)
    num = (Sw * Sxy) - (Sx * Sy)
    denx = (Sw * Sxx) - (Sx * Sx)
    deny = (Sw * Syy) - (Sy * Sy)
    if denx <= 0 or deny <= 0:
        return float("nan")
    return float(num / math.sqrt(denx * deny))


def _sv_diagnostics(
    sv_ids: np.ndarray,
    sv_labeled: np.ndarray,
    gt: np.ndarray,
    ignore_class: Optional[int] = 6,
) -> Dict[str, float]:
    # Flatten
    if sv_ids.ndim == 4 and sv_ids.shape[0] == 1:
        sv_ids = sv_ids[0]
    s = sv_ids.ravel().astype(np.int64)
    y = gt.ravel().astype(np.int64)
    y_pred = sv_labeled.ravel().astype(np.int64)

    # Mask ignored voxels
    mask = (y != int(ignore_class)) if (ignore_class is not None) else np.ones_like(y, dtype=bool)
    s_m = s[mask]
    y_m = y[mask]
    yp_m = y_pred[mask]

    # Map SV ids to 0..K-1
    sv_unique, sv_idx = np.unique(s_m, return_inverse=True)
    K = int(sv_unique.size)
    if K == 0:
        return {
            "sv_count": 0,
            "sv_vox_count": 0,
            "sv_mean_purity": float("nan"),
            "sv_mean_entropy_norm": float("nan"),
            "sv_vox_weighted_mean_purity": float("nan"),
            "sv_vox_weighted_mean_entropy_norm": float("nan"),
            "sv_corr_entropy_mismatch": float("nan"),
            "sv_corr_entropy_mismatch_weighted": float("nan"),
            "sv_frac_svs_entropy_ge_0.3": float("nan"),
            "sv_frac_svs_entropy_ge_0.5": float("nan"),
            "sv_frac_vox_entropy_ge_0.3": float("nan"),
            "sv_frac_vox_entropy_ge_0.5": float("nan"),
        }

    # Per-SV counts by class 0..4
    classes = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    R = int(classes.size)
    counts = np.zeros((K, R), dtype=np.int64)
    for j, cls in enumerate(classes.tolist()):
        sel = (y_m == cls)
        if not np.any(sel):
            continue
        counts[:, j] = np.bincount(sv_idx[sel], minlength=K)

    # Per-SV totals (non-ignored voxels)
    totals = counts.sum(axis=1).astype(np.int64)  # (K,)
    valid = totals > 0
    # Purity and normalized entropy over valid SVs
    purity = np.full(K, np.nan, dtype=np.float64)
    entropy_norm = np.full(K, np.nan, dtype=np.float64)
    if np.any(valid):
        p = counts[valid].astype(np.float64) / totals[valid][:, None]
        # Numerical safety
        p = np.clip(p, 1e-12, 1.0)
        H = -np.sum(p * np.log(p), axis=1)  # natural log
        H_max = math.log(R)
        entropy_norm[valid] = H / (H_max if H_max > 0 else 1.0)
        purity[valid] = p.max(axis=1)

    # Per-SV mismatch rate (fraction of voxels where assigned SV label != GT)
    mism = (yp_m != y_m).astype(np.int64)
    mism_counts = np.bincount(sv_idx, weights=mism, minlength=K).astype(np.int64)
    mismatch_rate = np.full(K, np.nan, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        mismatch_rate[valid] = (mism_counts[valid] / totals[valid].astype(np.float64))

    # Aggregate statistics
    w = totals.astype(np.float64)
    w_sum = float(w[valid].sum()) if np.any(valid) else 0.0
    mean_purity = float(np.nanmean(purity)) if np.any(valid) else float("nan")
    mean_entropy_n = float(np.nanmean(entropy_norm)) if np.any(valid) else float("nan")
    w_mean_purity = float(np.nansum(purity * w) / w_sum) if w_sum > 0 else float("nan")
    w_mean_entropy_n = float(np.nansum(entropy_norm * w) / w_sum) if w_sum > 0 else float("nan")

    # Correlations
    corr_acc_u: Dict[str, float] = {}
    corr_acc_w: Dict[str, float] = {}
    _update_running_corr(corr_acc_u, entropy_norm[valid], mismatch_rate[valid], w=None)
    _update_running_corr(corr_acc_w, entropy_norm[valid], mismatch_rate[valid], w=w[valid])
    corr_u = _finalize_running_corr(corr_acc_u)
    corr_w = _finalize_running_corr(corr_acc_w)

    # Threshold fractions
    def frac_ge(arr: np.ndarray, thr: float) -> Tuple[float, float]:
        if not np.any(valid):
            return float("nan"), float("nan")
        m = (arr >= thr) & valid
        sv_frac = float(np.nansum(m.astype(np.float64)) / float(valid.sum()))
        vox_frac = float(np.nansum(w[m]) / w_sum) if w_sum > 0 else float("nan")
        return sv_frac, vox_frac

    sv_f_e03, vox_f_e03 = frac_ge(entropy_norm, 0.3)
    sv_f_e05, vox_f_e05 = frac_ge(entropy_norm, 0.5)

    return {
        "sv_count": int(K),
        "sv_vox_count": int(int(w.sum())),
        "sv_mean_purity": mean_purity,
        "sv_mean_entropy_norm": mean_entropy_n,
        "sv_vox_weighted_mean_purity": w_mean_purity,
        "sv_vox_weighted_mean_entropy_norm": w_mean_entropy_n,
        "sv_corr_entropy_mismatch": float(corr_u),
        "sv_corr_entropy_mismatch_weighted": float(corr_w),
        "sv_frac_svs_entropy_ge_0.3": sv_f_e03,
        "sv_frac_svs_entropy_ge_0.5": sv_f_e05,
        "sv_frac_vox_entropy_ge_0.3": vox_f_e03,
        "sv_frac_vox_entropy_ge_0.5": vox_f_e05,
    }


def _process_case_worker(payload: Tuple[str, str, str, str, int, bool, float]) -> Tuple[str, Dict[str, Dict[str, float]], float, Optional[Dict[str, float]]]:
    """Top-level worker function so it is picklable for ProcessPool.
    payload: (cid, gt_label_path_str, sv_dir_str, sv_ids_dir_str, ignore_class, heavy, hd_percentile)
    Returns: (cid, per_class_metrics_dict, voxel_acc, sv_diag_or_None)
    """
    cid, gt_label_path_str, sv_dir_str, sv_ids_dir_str, ignore_class, heavy, hd_percentile = payload
    try:
        gt_path = Path(gt_label_path_str)
        sv_lbl_path = Path(sv_dir_str) / f"{cid}_labels.npy"
        if not (gt_path.exists() and sv_lbl_path.exists()):
            return cid, {}, float("nan"), None
        gt = _load_nii_ras_int(gt_path)
        sv_lbl = np.load(str(sv_lbl_path)).astype(np.int64)
        if sv_lbl.ndim == 4 and sv_lbl.shape[0] == 1:
            sv_lbl = sv_lbl[0]
        if sv_lbl.shape != gt.shape:
            sv_lbl = _center_pad_or_crop(sv_lbl, gt.shape)
        per_class = tfw.compute_metrics(
            pred=_to_5d(sv_lbl),
            gt=_to_5d(gt),
            heavy=bool(heavy),
            hd_percentile=float(hd_percentile),
        )
        mask = (gt != int(ignore_class)) if (ignore_class is not None and ignore_class >= 0) else np.ones_like(gt, dtype=bool)
        total_cmp = int(mask.sum())
        vox_acc = float(((sv_lbl == gt) & mask).sum()) / float(total_cmp) if total_cmp > 0 else 0.0

        diag = None
        if sv_ids_dir_str:
            sv_ids_path = Path(sv_ids_dir_str) / f"{cid}_sv_ids.npy"
            if sv_ids_path.exists():
                sv_ids = np.load(str(sv_ids_path))
                if sv_ids.ndim == 4 and sv_ids.shape[0] == 1:
                    sv_ids = sv_ids[0]
                if sv_ids.shape != gt.shape:
                    sv_ids = _center_pad_or_crop(sv_ids, gt.shape)
                diag = _sv_diagnostics(sv_ids=sv_ids, sv_labeled=sv_lbl, gt=gt, ignore_class=int(ignore_class) if ignore_class >= 0 else None)

        return cid, {str(k): v for k, v in per_class.items()}, vox_acc, diag
    except Exception:
        # Ensure a robust return even if a single case fails
        return cid, {}, float("nan"), None


def main() -> None:
    args = get_parser().parse_args()
    out_dir = Path(args.output_dir)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    _init_eval_logging(out_dir=out_dir, enable=bool(args.log_to_file), filename=str(args.log_file_name))

    datalist = Path(args.datalist)
    sv_dir = Path(args.sv_dir)
    sv_ids_dir = Path(args.sv_ids_dir) if args.sv_ids_dir else None
    data_root = Path(args.data_root) if args.data_root else None

    # Build case list (train only)
    records = _load_json_list(datalist)
    ids_from_dl = [str(r.get("id")) for r in records if r.get("id")]
    ids_avail = _discover_sv_ids_available(sv_dir, suffix="_labels.npy")
    case_ids = sorted([i for i in ids_from_dl if i in ids_avail])
    if args.max_cases is not None and args.max_cases > 0:
        case_ids = case_ids[: int(args.max_cases)]
    if not case_ids:
        raise RuntimeError("No overlapping cases between datalist and sv-dir.")

    print(f"Cases to evaluate: {len(case_ids)} (from {datalist})")

    # Prepare accumulators
    classes = [0, 1, 2, 3, 4]
    sums = {c: {"dice": 0.0, "iou": 0.0, "hd": 0.0, "asd": 0.0, "n": 0} for c in classes}
    per_case_rows: List[List] = []

    # Dataset-level SV diag accumulators (unweighted across SVs and voxel-weighted)
    sv_totals = {
        "sv_count": 0,
        "sv_vox_count": 0,
        "sum_purity": 0.0,
        "sum_entropy_n": 0.0,
        "sum_purity_w": 0.0,
        "sum_entropy_n_w": 0.0,
        "sum_w": 0.0,
        "sv_ge_0.3": 0.0,
        "sv_ge_0.5": 0.0,
        "vox_ge_0.3": 0.0,
        "vox_ge_0.5": 0.0,
    }
    corr_u_acc: Dict[str, float] = {}
    corr_w_acc: Dict[str, float] = {}

    def _process_case(cid: str) -> Tuple[str, Dict[str, Dict[str, float]], float, Optional[Dict[str, float]]]:
        rec = next(r for r in records if r.get("id") == cid)
        gt_path = Path(rec.get("label"))
        sv_lbl_path = sv_dir / f"{cid}_labels.npy"
        if not (gt_path.exists() and sv_lbl_path.exists()):
            return cid, {}, float("nan"), None
        gt = _load_nii_ras_int(gt_path)
        sv_lbl = np.load(str(sv_lbl_path)).astype(np.int64)
        if sv_lbl.ndim == 4 and sv_lbl.shape[0] == 1:
            sv_lbl = sv_lbl[0]
        if sv_lbl.shape != gt.shape:
            sv_lbl = _center_pad_or_crop(sv_lbl, gt.shape)
        per_class = tfw.compute_metrics(
            pred=_to_5d(sv_lbl),
            gt=_to_5d(gt),
            heavy=bool(args.heavy),
            hd_percentile=float(args.hd_percentile),
        )
        mask = (gt != int(args.ignore_class)) if (args.ignore_class is not None) else np.ones_like(gt, dtype=bool)
        total_cmp = int(mask.sum())
        vox_acc = float(((sv_lbl == gt) & mask).sum()) / float(total_cmp) if total_cmp > 0 else 0.0
        diag = None
        if sv_ids_dir is not None:
            sv_ids_path = sv_ids_dir / f"{cid}_sv_ids.npy"
            if sv_ids_path.exists():
                sv_ids = np.load(str(sv_ids_path))
                if sv_ids.ndim == 4 and sv_ids.shape[0] == 1:
                    sv_ids = sv_ids[0]
                if sv_ids.shape != gt.shape:
                    sv_ids = _center_pad_or_crop(sv_ids, gt.shape)
                diag = _sv_diagnostics(sv_ids=sv_ids, sv_labeled=sv_lbl, gt=gt, ignore_class=int(args.ignore_class))
        return cid, {str(k): v for k, v in per_class.items()}, vox_acc, diag

    # Build per-case payloads (id -> label path string) to avoid shipping full records to workers
    id_to_label: Dict[str, str] = {str(r["id"]): str(r["label"]) for r in records if r.get("id") and r.get("label")}

    def _mk_payload(cid: str):
        return (
            cid,
            id_to_label.get(cid, ""),
            str(sv_dir),
            str(sv_ids_dir) if sv_ids_dir is not None else "",
            int(args.ignore_class) if args.ignore_class is not None else -1,
            bool(args.heavy),
            float(args.hd_percentile),
        )

    payloads = [_mk_payload(cid) for cid in case_ids]

    # Parallel map over cases
    results: List[Tuple[str, Dict[str, Dict[str, float]], float, Optional[Dict[str, float]]]] = []
    if int(args.num_workers) > 1:
        maxw = max(1, int(args.num_workers))
        with _cf.ProcessPoolExecutor(max_workers=maxw) as ex:
            futs = [ex.submit(_process_case_worker, pl) for pl in payloads]
            if args.progress:
                try:
                    from tqdm.auto import tqdm  # type: ignore
                    for f in tqdm(_cf.as_completed(futs), total=len(futs)):
                        results.append(f.result())
                except Exception:
                    for f in _cf.as_completed(futs):
                        results.append(f.result())
            else:
                for f in _cf.as_completed(futs):
                    results.append(f.result())
    else:
        if args.progress:
            try:
                from tqdm.auto import tqdm  # type: ignore
                for pl in tqdm(payloads):
                    results.append(_process_case_worker(pl))
            except Exception:
                for pl in payloads:
                    results.append(_process_case_worker(pl))
        else:
            for pl in payloads:
                results.append(_process_case_worker(pl))

    # Aggregate over results
    for cid, per_class, vox_acc, diag in results:
        if not per_class:
            print(f"skip {cid}: missing files")
            continue
        row = [cid]
        for c in classes:
            key = str(c)
            sums[c]["dice"] += per_class[key]["dice"]
            sums[c]["iou"] += per_class[key]["iou"]
            if args.heavy and per_class[key]["hd"] is not None:
                sums[c]["hd"] += float(per_class[key]["hd"])  # type: ignore[arg-type]
            if args.heavy and per_class[key]["asd"] is not None:
                sums[c]["asd"] += float(per_class[key]["asd"])  # type: ignore[arg-type]
            sums[c]["n"] += 1
            row.append(per_class[key]["dice"])
        for c in classes:
            key = str(c)
            row.append(per_class[key]["iou"])
        if args.heavy:
            for c in classes:
                key = str(c)
                row.append(per_class[key]["hd"] if per_class[key]["hd"] is not None else float("nan"))
            for c in classes:
                key = str(c)
                row.append(per_class[key]["asd"] if per_class[key]["asd"] is not None else float("nan"))
        row.append(vox_acc)
        per_case_rows.append(row)

        if diag is not None:
            K = int(diag["sv_count"])
            sv_totals["sv_count"] += K
            sv_totals["sv_vox_count"] += int(diag["sv_vox_count"])
            if K > 0 and math.isfinite(diag["sv_mean_purity"]):
                sv_totals["sum_purity"] += float(diag["sv_mean_purity"]) * K
            if K > 0 and math.isfinite(diag["sv_mean_entropy_norm"]):
                sv_totals["sum_entropy_n"] += float(diag["sv_mean_entropy_norm"]) * K
            if int(diag["sv_vox_count"]) > 0 and math.isfinite(diag["sv_vox_weighted_mean_purity"]):
                sv_totals["sum_purity_w"] += float(diag["sv_vox_weighted_mean_purity"]) * float(diag["sv_vox_count"])  # weight
            if int(diag["sv_vox_count"]) > 0 and math.isfinite(diag["sv_vox_weighted_mean_entropy_norm"]):
                sv_totals["sum_entropy_n_w"] += float(diag["sv_vox_weighted_mean_entropy_norm"]) * float(diag["sv_vox_count"])  # weight
            sv_totals["sum_w"] += float(diag["sv_vox_count"]) if int(diag["sv_vox_count"]) > 0 else 0.0
            if K > 0 and math.isfinite(diag["sv_frac_svs_entropy_ge_0.3"]):
                sv_totals["sv_ge_0.3"] += float(diag["sv_frac_svs_entropy_ge_0.3"]) * K
            if K > 0 and math.isfinite(diag["sv_frac_svs_entropy_ge_0.5"]):
                sv_totals["sv_ge_0.5"] += float(diag["sv_frac_svs_entropy_ge_0.5"]) * K
            if int(diag["sv_vox_count"]) > 0 and math.isfinite(diag["sv_frac_vox_entropy_ge_0.3"]):
                sv_totals["vox_ge_0.3"] += float(diag["sv_frac_vox_entropy_ge_0.3"]) * float(diag["sv_vox_count"])  # type: ignore[operator]
            if int(diag["sv_vox_count"]) > 0 and math.isfinite(diag["sv_frac_vox_entropy_ge_0.5"]):
                sv_totals["vox_ge_0.5"] += float(diag["sv_frac_vox_entropy_ge_0.5"]) * float(diag["sv_vox_count"])  # type: ignore[operator]

    # Write per-case CSV
    csv_path = out_dir / "metrics" / "per_case.csv"
    with csv_path.open("w", newline="") as fp:
        w = csv.writer(fp)
        header = ["id"] + [f"dice_{c}" for c in classes] + [f"iou_{c}" for c in classes]
        if args.heavy:
            header += [f"hd_{c}" for c in classes] + [f"asd_{c}" for c in classes]
        header += ["voxel_acc"]
        w.writerow(header)
        for row in per_case_rows:
            w.writerow(row)
    print(f"Wrote per-case metrics: {csv_path}")

    # Dataset-level summary
    per_class = {}
    for c in classes:
        entry = {
            "dice": sums[c]["dice"] / max(sums[c]["n"], 1),
            "iou": sums[c]["iou"] / max(sums[c]["n"], 1),
        }
        if args.heavy:
            entry["hd"] = sums[c]["hd"] / max(sums[c]["n"], 1)
            entry["asd"] = sums[c]["asd"] / max(sums[c]["n"], 1)
        else:
            entry["hd"] = None
            entry["asd"] = None
        per_class[str(c)] = entry

    avg = {
        "dice": float(np.mean([per_class[str(c)]["dice"] for c in classes])),
        "iou": float(np.mean([per_class[str(c)]["iou"] for c in classes])),
        "hd": float(np.mean([per_class[str(c)]["hd"] for c in classes if per_class[str(c)]["hd"] is not None])) if args.heavy else None,
        "asd": float(np.mean([per_class[str(c)]["asd"] for c in classes if per_class[str(c)]["asd"] is not None])) if args.heavy else None,
    }

    sv_diag_summary = None
    if sv_ids_dir is not None and sv_totals["sv_count"] > 0:
        K = float(sv_totals["sv_count"])
        W = float(sv_totals["sum_w"]) if float(sv_totals["sum_w"]) > 0 else float("nan")
        sv_diag_summary = {
            "sv_count": int(sv_totals["sv_count"]),
            "sv_vox_count": int(sv_totals["sv_vox_count"]),
            "sv_mean_purity": float(sv_totals["sum_purity"]) / K,
            "sv_mean_entropy_norm": float(sv_totals["sum_entropy_n"]) / K,
            "sv_vox_weighted_mean_purity": (float(sv_totals["sum_purity_w"]) / float(sv_totals["sum_w"])) if float(sv_totals["sum_w"]) > 0 else float("nan"),
            "sv_vox_weighted_mean_entropy_norm": (float(sv_totals["sum_entropy_n_w"]) / float(sv_totals["sum_w"])) if float(sv_totals["sum_w"]) > 0 else float("nan"),
            "sv_frac_svs_entropy_ge_0.3": float(sv_totals["sv_ge_0.3"]) / K,
            "sv_frac_svs_entropy_ge_0.5": float(sv_totals["sv_ge_0.5"]) / K,
            "sv_frac_vox_entropy_ge_0.3": (float(sv_totals["vox_ge_0.3"]) / float(sv_totals["sv_vox_count"])) if float(sv_totals["sv_vox_count"]) > 0 else float("nan"),
            "sv_frac_vox_entropy_ge_0.5": (float(sv_totals["vox_ge_0.5"]) / float(sv_totals["sv_vox_count"])) if float(sv_totals["sv_vox_count"]) > 0 else float("nan"),
            # Dataset-level corr not tracked here (requires SV-level streaming); omit to keep script light.
        }

    payload = {
        "per_class": per_class,
        "average": avg,
        "meta": {
            "ignore_label": int(args.ignore_class),
            "classes": classes,
            "both_empty_policy": "count_as_one",
            "heavy": bool(args.heavy),
            "hd_percentile": float(args.hd_percentile),
            "cases": case_ids,
        },
    }
    if sv_diag_summary is not None:
        payload["sv_diagnostics"] = sv_diag_summary

    summary_path = out_dir / "metrics" / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved dataset summary: {summary_path}")
    print("Per-class Dice/IoU (0..4):")
    for c in classes:
        e = per_class[str(c)]
        print(f"  class {c}: dice={e.get('dice'):.6f} iou={e.get('iou'):.6f}")
    print(f"Average: dice={avg.get('dice'):.6f} iou={avg.get('iou'):.6f}")
    # Also print dataset-level entropy if computed (normalized, 0..1)
    if sv_diag_summary is not None:
        print(
            "SV entropy (mean, normalized 0..1): "
            f"{sv_diag_summary.get('sv_mean_entropy_norm', float('nan')):.6f}"
        )
        print(
            "SV entropy (voxel-weighted mean, normalized 0..1): "
            f"{sv_diag_summary.get('sv_vox_weighted_mean_entropy_norm', float('nan')):.6f}"
        )


if __name__ == "__main__":
    main()
