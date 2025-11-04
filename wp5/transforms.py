from __future__ import annotations

from typing import List, Tuple

import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandSpatialCropd,
    SpatialPadd,
    SaveImage,
    MapTransform,
    ScaleIntensityRanged,
    RandGaussianNoised,
    RandShiftIntensityd,
    RandScaleIntensityd,
)

from .masks import build_points_supervision_mask


class ClipZScoreNormalizeD(MapTransform):
    def __init__(self, keys: List[str]):
        super().__init__(keys)
        self.eps = 1e-8

    def __call__(self, data):
        d = dict(data)
        import numpy as np
        for key in self.keys:
            arr = d.get(key)
            if arr is None:
                continue
            if arr.ndim >= 3:
                if arr.ndim == 3:
                    flat = arr.reshape(-1)
                else:
                    flat = arr.reshape(arr.shape[0], -1)
                    flat = flat.reshape(-1)
            else:
                flat = arr.reshape(-1)
            p1 = np.percentile(flat, 1)
            p99 = np.percentile(flat, 99)
            clipped = np.clip(arr, p1, p99)
            mean = clipped.mean()
            std = clipped.std()
            d[key] = ((clipped - mean) / (std + self.eps)).astype(np.float32)
        return d


class FGBiasedCropD(MapTransform):
    def __init__(self, keys: List[str], roi_size: Tuple[int, int, int], prob: float = 0.0, margin: int = 8):
        super().__init__(keys)
        self.roi = roi_size
        self.prob = float(prob)
        self.margin = int(margin)

    def __call__(self, data):
        import numpy as np
        d = dict(data)
        img = d.get("image")
        lbl = d.get("label")
        if img is None or lbl is None or lbl.ndim != 4 or img.ndim != 4:
            return d
        _, X, Y, Z = lbl.shape
        rx, ry, rz = self.roi
        minx = 0; miny = 0; minz = 0
        maxx = max(0, X - rx); maxy = max(0, Y - ry); maxz = max(0, Z - rz)
        def apply_crop(dct, sx, sy, sz):
            for k in self.keys:
                arr = dct.get(k)
                if arr is None or arr.ndim != 4:
                    continue
                dct[k] = arr[:, sx : sx + rx, sy : sy + ry, sz : sz + rz]
        sx = int(np.random.randint(minx, maxx + 1)) if maxx > minx else 0
        sy = int(np.random.randint(miny, maxy + 1)) if maxy > miny else 0
        sz = int(np.random.randint(minz, maxz + 1)) if maxz > minz else 0
        use_fg = (self.prob > 0.0) and (np.random.rand() <= self.prob)
        if use_fg:
            fg_mask = (lbl[0] != 0) & (lbl[0] != 6)
            fg_indices = np.argwhere(fg_mask)
            if fg_indices.size > 0:
                cx, cy, cz = fg_indices[np.random.randint(0, fg_indices.shape[0])]
                sx = int(np.clip(cx - rx // 2, minx, maxx))
                sy = int(np.clip(cy - ry // 2, miny, maxy))
                sz = int(np.clip(cz - rz // 2, minz, maxz))
        apply_crop(d, sx, sy, sz)
        return d


class BuildStaticPointsMaskD(MapTransform):
    def __init__(
        self,
        keys: list[str],
        out_key: str = "sup_mask",
        id_key: str = "id",
        ratio: float = 0.01,
        dilate_radius: int = 1,
        balance: str = "proportional",
        max_seeds: int = -1,
        bg_frac: float = 0.25,
        seed_strategy: str = "random",
        no_overlap_after_dilation: bool = False,
        save_dir: str | None = None,
    ):
        super().__init__(keys)
        self.out_key = out_key
        self.id_key = id_key
        self.ratio = float(ratio)
        self.dilate_radius = int(dilate_radius)
        self.balance = str(balance)
        self.max_seeds = int(max_seeds)
        self.bg_frac = float(bg_frac)
        self.seed_strategy = str(seed_strategy)
        self._cache = {}
        self.save_dir = save_dir
        self.no_overlap_after_dilation = bool(no_overlap_after_dilation)

    def __call__(self, data):
        import numpy as np
        d = dict(data)
        case_id = d.get(self.id_key)
        if isinstance(case_id, list) and case_id:
            case_id = case_id[0]
        if case_id is None:
            case_id = f"case_{id(d)}"
        if case_id in self._cache:
            sup = self._cache[case_id]
            d[self.out_key] = sup.copy()
            return d
        lbl = d.get(self.keys[0])
        if lbl is None:
            return d
        if isinstance(lbl, np.ndarray):
            import torch
            t = torch.from_numpy(lbl).long().unsqueeze(0) if lbl.ndim == 3 else torch.from_numpy(lbl).long()
        else:
            import torch
            t = lbl if isinstance(lbl, torch.Tensor) else torch.as_tensor(lbl).long()
        if t.ndim == 4:
            t = t.unsqueeze(0)
        auto_max = self.max_seeds
        if auto_max <= 0:
            auto_max = 5000 if self.ratio >= 0.1 - 1e-8 else 500
        sup_mask, seed_mask = build_points_supervision_mask(
            labels=t,
            ratio=self.ratio,
            dilate_radius=self.dilate_radius,
            balance=self.balance,
            max_seeds=auto_max,
            bg_frac=self.bg_frac,
            seed_strategy=self.seed_strategy,
            ensure_min_coverage=True,
            max_iter=6,
            no_overlap_after_dilation=self.no_overlap_after_dilation,
            return_seeds=True,
        )
        sup_np = sup_mask[0].cpu().numpy().astype(bool)
        seed_np = seed_mask[0].cpu().numpy().astype(bool)
        d[self.out_key] = sup_np
        d["seed_mask"] = seed_np
        self._cache[case_id] = sup_np
        if self.save_dir:
            from pathlib import Path
            import json
            sdir = Path(self.save_dir)
            sdir.mkdir(parents=True, exist_ok=True)
            safe_id = str(case_id).replace('/', '_')
            np.save(sdir / f"{safe_id}_supmask.npy", sup_np)
            np.save(sdir / f"{safe_id}_seedmask.npy", seed_np)
            if isinstance(lbl, np.ndarray):
                lbl_np = lbl
            else:
                lbl_np = np.asarray(lbl)
            counts = {}
            seed_counts = {}
            for c in [0, 1, 2, 3, 4, 6]:
                counts[str(c)] = int((np.logical_and(sup_np[0], lbl_np[0] == c)).sum())
                seed_counts[str(c)] = int((np.logical_and(seed_np[0], lbl_np[0] == c)).sum())
            stats = {
                "id": case_id,
                "shape": list(sup_np.shape),
                "sup_fraction": float(sup_np.mean()),
                "seed_fraction": float(seed_np.mean()),
                "counts_per_class": counts,
                "seed_counts_per_class": seed_counts,
            }
            (sdir / f"{safe_id}_supmask_stats.json").write_text(json.dumps(stats, indent=2))
        return d


class LoadSavedMasksD(MapTransform):
    def __init__(self, keys: list[str], id_key: str, dir_path: str):
        super().__init__(keys)
        from pathlib import Path
        self.id_key = id_key
        self.dir = Path(dir_path)

    def __call__(self, data):
        import numpy as np
        d = dict(data)
        cid = d.get(self.id_key)
        if isinstance(cid, list) and cid:
            cid = cid[0]
        safe_id = str(cid).replace('/', '_')
        smp = self.dir / f"{safe_id}_supmask.npy"
        sdp = self.dir / f"{safe_id}_seedmask.npy"
        plp = self.dir / f"{safe_id}_pseudolabel.npy"
        if smp.exists():
            d["sup_mask"] = np.load(smp).astype(bool)
        if sdp.exists():
            d["seed_mask"] = np.load(sdp).astype(bool)
        if plp.exists():
            d["pseudo_label"] = np.load(plp).astype(np.int64)
        if "sup_mask" in d and isinstance(d["sup_mask"], np.ndarray):
            shp = d["sup_mask"].shape
            d["valid_mask"] = np.ones(shp, dtype=bool)
        else:
            label = d.get("label")
            if isinstance(label, np.ndarray):
                d["valid_mask"] = np.ones_like(label, dtype=bool)
        return d


def get_transforms(
    roi=(112, 112, 80),
    norm: str = "clip_zscore",
    aug_intensity: bool = False,
    aug_prob: float = 0.2,
    aug_noise_std: float = 0.01,
    aug_shift: float = 0.1,
    aug_scale: float = 0.1,
    fg_crop_prob: float = 0.0,
    fg_crop_margin: int = 8,
    fewshot_mode: str | None = None,
    fewshot_static: bool = False,
    fewshot_ratio: float = 0.0,
    fp_dilate_radius: int = 1,
    fp_balance: str = "proportional",
    fp_bg_frac: float = 0.25,
    fp_max_seeds: int = -1,
    fp_seed_strategy: str = "random",
    fp_no_overlap: bool = False,
    save_sup_masks_dir: str | None = None,
):
    norm_transform = None
    if norm == "clip_zscore":
        norm_transform = ClipZScoreNormalizeD(keys=["image"])
    elif norm == "fixed_wp5":
        norm_transform = ScaleIntensityRanged(keys=["image"], a_min=-3, a_max=8.5, b_min=0.0, b_max=1.0, clip=True)
    elif norm == "none":
        norm_transform = None
    else:
        raise ValueError(f"Unknown normalization option: {norm}")

    def build_seq(include_crop: bool, include_aug: bool, training: bool):
        seq = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
        ]
        static_sup = training and fewshot_static and (save_sup_masks_dir is not None)
        if static_sup:
            seq.append(LoadSavedMasksD(keys=["label"], id_key="id", dir_path=save_sup_masks_dir))
        if norm_transform is not None:
            seq.append(norm_transform)
        if training:
            pad_keys = ["image", "label", "sup_mask", "seed_mask", "pseudo_label", "valid_mask"] if static_sup else ["image", "label"]
            seq.append(SpatialPadd(keys=pad_keys, spatial_size=roi, allow_missing_keys=True))
        if training and aug_intensity:
            seq.extend(
                [
                    RandGaussianNoised(keys=["image"], prob=aug_prob, std=aug_noise_std),
                    RandShiftIntensityd(keys=["image"], offsets=aug_shift, prob=aug_prob),
                    RandScaleIntensityd(keys=["image"], factors=aug_scale, prob=aug_prob),
                ]
            )
        if include_aug:
            flip_keys = ["image", "label", "sup_mask", "seed_mask", "pseudo_label", "valid_mask"] if static_sup else ["image", "label"]
            seq.extend([
                RandFlipd(keys=flip_keys, spatial_axis=0, prob=0.5, allow_missing_keys=True),
                RandFlipd(keys=flip_keys, spatial_axis=1, prob=0.5, allow_missing_keys=True),
                RandFlipd(keys=flip_keys, spatial_axis=2, prob=0.5, allow_missing_keys=True),
            ])
        if include_crop:
            if fg_crop_prob > 0.0 and training:
                crop_keys = ["image", "label", "sup_mask", "seed_mask", "pseudo_label", "valid_mask"] if static_sup else ["image", "label"]
                seq.append(FGBiasedCropD(keys=crop_keys, roi_size=roi, prob=fg_crop_prob, margin=fg_crop_margin))
            else:
                crop_keys = ["image", "label", "sup_mask", "seed_mask", "pseudo_label", "valid_mask"] if static_sup else ["image", "label"]
                seq.append(RandSpatialCropd(keys=crop_keys, roi_size=roi, random_size=False, allow_missing_keys=True))
        return seq

    train = Compose(build_seq(include_crop=True, include_aug=True, training=True))
    val = Compose(build_seq(include_crop=False, include_aug=False, training=False))
    return train, val

