#!/usr/bin/env python3
"""
WP5 evaluation script.

Loads a checkpoint, runs sliding-window inference on the test set, computes Dice and Jaccard (IoU)
over classes 0..4 while ignoring class 6, and optionally saves predictions for visualization.

Defaults align with AGENTS.md policies and train_finetune_wp5.py. Reuses model builders and
transforms from train_finetune_wp5 to keep behavior consistent.

Examples:
  python scripts/eval_wp5.py \
    --ckpt runs/grid_clip_zscore/scratch_subset_100/last.ckpt \
    --datalist datalist_test.json \
    --output_dir runs/grid_clip_zscore/scratch_subset_100/eval \
    --save_preds --heavy --hd_percentile 95 --empty_pair_policy count_as_one

Or build the test list via WP5 split config:
  python scripts/eval_wp5.py \
    --ckpt runs/grid_clip_zscore/scratch_subset_100/last.ckpt \
    --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
    --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json \
    --output_dir runs/grid_clip_zscore/scratch_subset_100_eval \
    --save_preds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from monai.data import DataLoader, Dataset

# Ensure project root is on sys.path so we can import train_finetune_wp5 when running as a script.
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import helpers from training script to ensure consistency
import train_finetune_wp5 as tfw


def _init_eval_logging(out_dir: Path, enable: bool = True, filename: str = "eval.log") -> None:
    if not enable:
        return
    (out_dir).mkdir(parents=True, exist_ok=True)
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


def load_test_list(args) -> List[Dict]:
    if args.datalist and Path(args.datalist).exists():
        return json.loads(Path(args.datalist).read_text())
    if args.data_root and args.split_cfg:
        _, test_list = tfw.build_datalists(Path(args.data_root) / "data", Path(args.split_cfg))
        return test_list
    raise FileNotFoundError(
        "Provide --datalist path or both --data_root and --split_cfg to locate the test set.")


def _discover_bundle_dir_from_log(ckpt_path: Path) -> str | None:
    """Try to discover bundle_dir by parsing a sibling train.log next to the checkpoint.
    Looks for a line like: 'Built network from bundle: <path>'.
    """
    try:
        logp = ckpt_path.parent / "train.log"
        if not logp.exists():
            return None
        import re
        for line in logp.read_text().splitlines():
            m = re.search(r"Built network from bundle:\s*(.+)$", line)
            if m:
                cand = m.group(1).strip()
                if cand and Path(cand).exists():
                    return cand
                return cand  # return even if missing; still helpful for error msg
    except Exception:
        return None
    return None


def build_model(args) -> torch.nn.Module:
    # Auto-discover bundle_dir if not provided but present in train.log near ckpt
    bundle_dir = args.bundle_dir
    if not bundle_dir and args.ckpt:
        auto = _discover_bundle_dir_from_log(Path(args.ckpt))
        if auto:
            print(f"Discovered bundle_dir from train.log: {auto}")
            bundle_dir = auto
    if bundle_dir:
        return tfw.build_model_from_bundle(Path(bundle_dir), out_channels=5)
    return tfw.build_model(args.net)


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("WP5 evaluation (Dice/IoU, HD/ASD, save predictions)")
    p.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (state_dict or container)")
    p.add_argument("--output_dir", type=str, required=True, help="Base dir to save metrics/predictions (timestamp appended unless --no_timestamp)")
    # Dataset
    p.add_argument("--datalist", type=str, default="datalist_test.json", help="JSON list for test set")
    p.add_argument("--data_root", type=str, default="", help="WP5 dataset root (if not using --datalist)")
    p.add_argument("--split_cfg", type=str, default="", help="Split config JSON (if not using --datalist)")
    # Model and transforms
    p.add_argument("--net", choices=["basicunet", "unet"], default="basicunet")
    p.add_argument("--bundle_dir", type=str, default="", help="MONAI bundle directory (optional)")
    p.add_argument("--norm", choices=["clip_zscore", "fixed_wp5", "none"], default="clip_zscore")
    p.add_argument("--roi_x", type=int, default=112)
    p.add_argument("--roi_y", type=int, default=112)
    p.add_argument("--roi_z", type=int, default=80)
    p.add_argument("--num_workers", type=int, default=4)
    # Eval options
    p.add_argument("--save_preds", action="store_true", help="Save predictions as NIfTI under <output_dir>/preds")
    p.add_argument("--max_cases", type=int, default=-1, help="Limit number of evaluated cases (for smoke tests)")
    p.add_argument("--heavy", action="store_true", help="Also compute HD/ASD (slower)")
    p.add_argument("--hd_percentile", type=float, default=95.0, help="Hausdorff percentile: 95.0 for HD95, 100.0 for full HD")
    p.add_argument("--no_timestamp", action="store_true", help="Do not append timestamp to --output_dir")
    # logging
    p.add_argument("--log_to_file", action="store_true", help="Tee stdout/stderr to <output_dir>/eval.log")
    p.add_argument("--log_file_name", type=str, default="eval.log", help="Eval log filename")
    return p


def main():
    p = get_parser()
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    if not args.no_timestamp:
        import time as _time
        ts = _time.strftime("%Y%m%d-%H%M%S")
        out_dir = out_dir.parent / f"{out_dir.name}_{ts}"
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    _init_eval_logging(out_dir=out_dir, enable=bool(getattr(args, "log_to_file", False)), filename=str(getattr(args, "log_file_name", "eval.log")))

    # Dataset
    test_list = load_test_list(args)
    _, t_val = tfw.get_transforms(roi=(args.roi_x, args.roi_y, args.roi_z), norm=args.norm)
    ds_test = Dataset(test_list, transform=t_val)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Model and checkpoint
    net = build_model(args).to(device)
    # Prefer strict load first (exact match from our training run), then fallback to robust non-strict
    sd = torch.load(args.ckpt, map_location=device)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    try:
        net.load_state_dict(sd, strict=True)
        print(f"Loaded checkpoint (strict): {args.ckpt}")
    except Exception as e_strict:
        print(f"Strict load failed ({e_strict}); trying robust non-strict loader...")
        try:
            tfw.load_pretrained_non_strict(net, Path(args.ckpt), device)
            print(f"Loaded checkpoint: {args.ckpt}")
        except Exception as e_non:
            print(f"WARNING: Non-strict loader also failed: {e_non}")
            net.load_state_dict(sd, strict=False)
            print(f"Loaded checkpoint (loose non-strict): {args.ckpt}")
            print("If metrics are unexpectedly low, verify you passed --bundle_dir if the model was trained from a MONAI bundle.")

    # Evaluate (Dice/IoU always; HD/ASD if --heavy)
    metrics = tfw.evaluate(
        net, dl_test, device, out_dir,
        save_preds=args.save_preds,
        max_cases=(None if args.max_cases < 0 else args.max_cases),
        heavy=bool(args.heavy),
        hd_percentile=float(args.hd_percentile),
    )
    (out_dir / "metrics" / "summary.json").write_text(json.dumps(metrics, indent=2))

    # Pretty print per-class metrics
    pc = metrics.get("per_class", {})
    avg = metrics.get("average", {})
    print("Per-class Dice/IoU (0..4):")
    for c in [0, 1, 2, 3, 4]:
        e = pc.get(str(c), {})
        print(f"  class {c}: dice={e.get('dice'):.6f} iou={e.get('iou'):.6f}")
    print(f"Average: dice={avg.get('dice'):.6f} iou={avg.get('iou'):.6f}")
    print(f"Saved metrics to: {(out_dir / 'metrics' / 'summary.json')}" )
    if args.save_preds:
        print(f"Saved predictions under: {(out_dir / 'preds')}" )


if __name__ == "__main__":
    main()
