#!/usr/bin/env python3
import argparse
from pathlib import Path

# Reuse the trainer's build_datalists so that split checks stay consistent with
# whatever dataset layout (legacy or new BumpDataset) the training code uses.
import sys as _sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))
import train_finetune_wp5 as tfw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="WP5 dataset root (contains data/)")
    ap.add_argument("--split_cfg", required=True, help="Split JSON with test_serial_numbers")
    args = ap.parse_args()

    data_dir = Path(args.data_root) / "data"
    train, test = tfw.build_datalists(data_dir, Path(args.split_cfg))
    train_ids = {d["id"] for d in train}
    test_ids = {d["id"] for d in test}
    inter = train_ids & test_ids
    print(f"Train: {len(train)} | Test: {len(test)} | Overlap: {len(inter)}")
    if inter:
        print("Overlap IDs (first 10):", sorted(list(inter))[:10])
        raise SystemExit(1)
    else:
        print("OK: Train/Test splits are disjoint by serial numbers.")


if __name__ == "__main__":
    main()
