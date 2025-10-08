#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path


def build_datalists(data_dir: Path, cfg_path: Path):
    cfg = json.loads(cfg_path.read_text())
    test_serials = set(cfg.get("test_serial_numbers", []))

    def serial_from_name(n: str):
        m = re.match(r"^SN(\d+)", n)
        return int(m.group(1)) if m else None

    pairs = {}
    for n in os.listdir(data_dir):
        if n.endswith("_image.nii"):
            base = n[:-10]
            img = str(data_dir / f"{base}_image.nii")
            lbl = str(data_dir / f"{base}_label.nii")
            if os.path.exists(lbl):
                pairs[base] = (img, lbl, serial_from_name(n))

    train, test = [], []
    for k, (img, lbl, serial) in pairs.items():
        rec = {"image": img, "label": lbl, "id": k}
        (test if serial in test_serials else train).append(rec)
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="WP5 dataset root (contains data/)")
    ap.add_argument("--split_cfg", required=True, help="Split JSON with test_serial_numbers")
    args = ap.parse_args()

    data_dir = Path(args.data_root) / "data"
    train, test = build_datalists(data_dir, Path(args.split_cfg))
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

