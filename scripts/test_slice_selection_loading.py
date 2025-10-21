#!/usr/bin/env python3
"""
Lightweight test for static few-slices supervision loading.

It verifies that:
- sup_masks produced by scripts/select_informative_slices.py can be loaded by train_finetune_wp5.get_transforms
- sup_mask aligns with label shape and matches the saved supmask file
- Reports basic coverage stats for the first few samples
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from monai.data import Dataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from train_finetune_wp5 import LoadSavedMasksD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_list", type=str, required=True)
    ap.add_argument("--sup_masks_dir", type=str, required=True)
    ap.add_argument("--roi", type=int, nargs=3, default=(112, 112, 80))
    ap.add_argument("--max_cases", type=int, default=3)
    args = ap.parse_args()

    train_list = json.loads(Path(args.train_list).read_text())
    t = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        LoadSavedMasksD(keys=["label"], id_key="id", dir_path=args.sup_masks_dir),
    ])
    ds = Dataset(train_list, transform=t)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    sup_dir = Path(args.sup_masks_dir)
    for i, batch in enumerate(dl):
        if i >= args.max_cases:
            break
        cid = batch.get("id", [f"case{i}"])[0]
        lbl = batch["label"].numpy()
        sup = batch["sup_mask"].numpy()
        assert lbl.shape == sup.shape, f"Shape mismatch for {cid}: lbl {lbl.shape} vs sup {sup.shape}"
        # Compare to saved supmask on disk
        sup_disk = np.load(sup_dir / f"{cid.replace('/', '_')}_supmask.npy")
        assert sup_disk.shape == sup.shape[1:], f"Disk sup shape mismatch for {cid}: {sup_disk.shape}"
        same = np.array_equal(sup_disk, sup[0])
        frac = float(sup[0].mean())
        print(f"[{i}] id={cid} sup_fraction={frac:.6f} equals_disk={same}")


if __name__ == "__main__":
    main()
