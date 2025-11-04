#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Robust import from repo root
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import train_finetune_wp5 as tfw


def main():
    ap = argparse.ArgumentParser(description="Generate WP5 datalist_train.json and datalist_test.json using split config")
    ap.add_argument("--data_root", required=True, help="WP5 dataset root containing data/")
    ap.add_argument("--split_cfg", required=True, help="Split config JSON with test_serial_numbers")
    ap.add_argument("--out_train", default="datalist_train.json", help="Output path for train datalist JSON")
    ap.add_argument("--out_test", default="datalist_test.json", help="Output path for test datalist JSON")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    split_cfg = Path(args.split_cfg)
    train, test = tfw.build_datalists(data_root / "data", split_cfg)
    Path(args.out_train).write_text(json.dumps(train, indent=2))
    Path(args.out_test).write_text(json.dumps(test, indent=2))
    print(f"Wrote {len(train)} train and {len(test)} test entries.")


if __name__ == "__main__":
    main()

