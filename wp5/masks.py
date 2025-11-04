from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _select_slices_mask_per_sample(lbl: torch.Tensor, axis: int, k: int) -> torch.Tensor:
    if lbl.ndim != 5:
        raise ValueError(f"Expected 5D lbl (B=1,C=1,X,Y,Z), got shape={tuple(lbl.shape)}")
    device = lbl.device
    fg = (lbl != 0) & (lbl != 6)
    if axis == 2:
        counts = fg.sum(dim=(0, 1, 2, 3))
    elif axis == 1:
        counts = fg.sum(dim=(0, 1, 2, 4))
    elif axis == 0:
        counts = fg.sum(dim=(0, 1, 3, 4))
    else:
        raise ValueError("axis must be 0,1,2 for X,Y,Z")
    k_eff = max(1, min(int(k), int(counts.numel())))
    topk = torch.topk(counts, k=k_eff, largest=True).indices.to(device)
    mask = torch.zeros_like(fg, dtype=torch.bool)
    if axis == 2:
        mask[:, :, :, :, topk] = True
    elif axis == 1:
        mask[:, :, :, topk, :] = True
    else:
        mask[:, :, topk, :, :] = True
    return mask


def build_slice_supervision_mask(labels: torch.Tensor, roi: Tuple[int, int, int], axis_mode: str, ratio: float, k_override: int | None) -> torch.Tensor:
    B, _, X, Y, Z = labels.shape
    def k_from_ratio(length: int) -> int:
        k = int(np.ceil(max(1e-6, ratio) * length))
        return max(1, k)
    if axis_mode == "z":
        kz = k_override if (k_override and k_override > 0) else k_from_ratio(Z)
        ks = {2: kz}
    elif axis_mode == "y":
        ky = k_override if (k_override and k_override > 0) else k_from_ratio(Y)
        ks = {1: ky}
    elif axis_mode == "x":
        kx = k_override if (k_override and k_override > 0) else k_from_ratio(X)
        ks = {0: kx}
    elif axis_mode == "multi":
        kx = k_override if (k_override and k_override > 0) else k_from_ratio(X)
        ky = k_override if (k_override and k_override > 0) else k_from_ratio(Y)
        kz = k_override if (k_override and k_override > 0) else k_from_ratio(Z)
        ks = {0: kx, 1: ky, 2: kz}
    else:
        raise ValueError("axis_mode must be one of {'x','y','z','multi'}")
    masks = []
    for b in range(B):
        m = torch.zeros_like(labels[b:b+1], dtype=torch.bool)
        for ax, kval in ks.items():
            if kval <= 0:
                continue
            m_ax = _select_slices_mask_per_sample(labels[b:b+1], axis=ax, k=kval)
            m |= m_ax
        masks.append(m)
    return torch.cat(masks, dim=0)


def build_points_supervision_mask(
    labels: torch.Tensor,
    ratio: float,
    dilate_radius: int,
    balance: str,
    max_seeds: int,
    bg_frac: float,
    seed_strategy: str,
    ensure_min_coverage: bool = False,
    max_iter: int = 5,
    no_overlap_after_dilation: bool = False,
    return_seeds: bool = False,
):
    B, _, X, Y, Z = labels.shape
    device = labels.device
    total_voxels = X * Y * Z
    target_voxels = int(np.ceil(max(1e-6, ratio) * total_voxels))
    kv = (2 * dilate_radius + 1) ** 3
    seeds_target = max(1, min(int(np.ceil(target_voxels / kv)), max_seeds if max_seeds > 0 else target_voxels))
    out_masks = []
    out_seed_masks = [] if return_seeds else None
    classes = [1, 2, 3, 4]
    for b in range(B):
        lbl = labels[b, 0]
        cls_counts = []
        cls_indices = []
        for c in classes:
            idx = (lbl == c).nonzero(as_tuple=False)
            cls_indices.append(idx)
            cls_counts.append(idx.shape[0])
        bg_idx = (lbl == 0).nonzero(as_tuple=False)
        n_bg = int(np.floor(seeds_target * np.clip(bg_frac, 0.0, 0.9)))
        n_fg_total = max(0, seeds_target - n_bg)
        per_cls = np.zeros(len(classes), dtype=int)
        if n_fg_total > 0:
            if balance == "uniform":
                base = n_fg_total // len(classes)
                rem = n_fg_total % len(classes)
                per_cls[:] = base
                per_cls[:rem] += 1
            else:
                counts = np.array(cls_counts, dtype=float)
                s = counts.sum()
                if s > 0:
                    per_cls = np.floor(n_fg_total * (counts / s)).astype(int)
                    rem = n_fg_total - per_cls.sum()
                    order = np.argsort(-counts)
                    for i in range(rem):
                        per_cls[order[i % len(order)]] += 1
                else:
                    n_bg = seeds_target
                    per_cls[:] = 0
        seed_mask = torch.zeros_like(lbl, dtype=torch.bool)
        blocked = torch.zeros_like(lbl, dtype=torch.bool) if no_overlap_after_dilation else None
        g = torch.Generator(device=device)
        g.manual_seed(torch.seed())
        for c_i, idx in enumerate(cls_indices):
            n = int(per_cls[c_i])
            if n <= 0 or idx.shape[0] == 0:
                continue
            if no_overlap_after_dilation:
                mask_ok = ~blocked[idx[:, 0], idx[:, 1], idx[:, 2]]
                idx = idx[mask_ok]
                if idx.shape[0] == 0:
                    continue
            perm = torch.randperm(idx.shape[0], generator=g, device=device)[: min(n, idx.shape[0])]
            sel = idx[perm]
            seed_mask[sel[:, 0], sel[:, 1], sel[:, 2]] = True
            if no_overlap_after_dilation:
                added = torch.zeros_like(lbl, dtype=torch.bool)
                added[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                added_t = added.unsqueeze(0).unsqueeze(0).float()
                k = 2 * dilate_radius + 1
                dil = F.max_pool3d(added_t, kernel_size=k, stride=1, padding=dilate_radius) > 0.5
                blocked |= dil[0, 0]
        if n_bg > 0 and bg_idx.shape[0] > 0:
            idx = bg_idx
            if no_overlap_after_dilation:
                mask_ok = ~blocked[idx[:, 0], idx[:, 1], idx[:, 2]]
                idx = idx[mask_ok]
            if idx.shape[0] > 0:
                perm = torch.randperm(idx.shape[0], generator=g, device=device)[: min(n_bg, idx.shape[0])]
                sel = idx[perm]
                seed_mask[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                if no_overlap_after_dilation:
                    added = torch.zeros_like(lbl, dtype=torch.bool)
                    added[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                    added_t = added.unsqueeze(0).unsqueeze(0).float()
                    k = 2 * dilate_radius + 1
                    dil = F.max_pool3d(added_t, kernel_size=k, stride=1, padding=dilate_radius) > 0.5
                    blocked |= dil[0, 0]
        seed_mask_t = seed_mask.unsqueeze(0).unsqueeze(0).float()
        k = 2 * dilate_radius + 1
        dilated = F.max_pool3d(seed_mask_t, kernel_size=k, stride=1, padding=dilate_radius)
        sup = dilated > 0.5
        if ensure_min_coverage:
            target_cov = float(max(1e-6, ratio))
            it = 0
            while it < max_iter:
                cur_cov = float(sup.float().mean().item())
                if cur_cov >= target_cov:
                    break
                need_voxels = int(np.ceil(target_cov * total_voxels - float(sup.sum().item())))
                if need_voxels <= 0:
                    break
                add_seeds = max(1, int(np.ceil(need_voxels / kv)))
                add_bg = int(np.floor(add_seeds * np.clip(bg_frac, 0.0, 0.9)))
                add_fg_total = max(0, add_seeds - add_bg)
                add_per_cls = np.zeros(len(classes), dtype=int)
                if add_fg_total > 0:
                    elig_counts = []
                    elig_indices = []
                    sup3 = sup[0, 0]
                    for idx in cls_indices:
                        if idx.shape[0] == 0:
                            elig_indices.append(idx)
                            elig_counts.append(0)
                            continue
                        mask = ~sup3[idx[:, 0], idx[:, 1], idx[:, 2]]
                        if no_overlap_after_dilation and blocked is not None:
                            mask &= ~blocked[idx[:, 0], idx[:, 1], idx[:, 2]]
                        elig = idx[mask]
                        elig_indices.append(elig)
                        elig_counts.append(elig.shape[0])
                    counts = np.array(elig_counts, dtype=float)
                    s = counts.sum()
                    if s > 0:
                        add_per_cls = np.floor(add_fg_total * (counts / s)).astype(int)
                        rem = add_fg_total - add_per_cls.sum()
                        order = np.argsort(-counts)
                        for i in range(rem):
                            add_per_cls[order[i % len(order)]] += 1
                    else:
                        add_bg = add_seeds
                        add_per_cls[:] = 0
                for c_i, elig in enumerate(elig_indices if add_fg_total > 0 else []):
                    n = int(add_per_cls[c_i])
                    if n <= 0 or elig.shape[0] == 0:
                        continue
                    perm = torch.randperm(elig.shape[0], generator=g, device=device)[: min(n, elig.shape[0])]
                    sel = elig[perm]
                    seed_mask[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                    if no_overlap_after_dilation:
                        added = torch.zeros_like(lbl, dtype=torch.bool)
                        added[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                        added_t = added.unsqueeze(0).unsqueeze(0).float()
                        k = 2 * dilate_radius + 1
                        dil = F.max_pool3d(added_t, kernel_size=k, stride=1, padding=dilate_radius) > 0.5
                        blocked |= dil[0, 0]
                if add_bg > 0 and bg_idx.shape[0] > 0:
                    sup3 = sup[0, 0]
                    mask = ~sup3[bg_idx[:, 0], bg_idx[:, 1], bg_idx[:, 2]]
                    if no_overlap_after_dilation and blocked is not None:
                        mask &= ~blocked[bg_idx[:, 0], bg_idx[:, 1], bg_idx[:, 2]]
                    elig_bg = bg_idx[mask]
                    if elig_bg.shape[0] > 0:
                        perm = torch.randperm(elig_bg.shape[0], generator=g, device=device)[: min(add_bg, elig_bg.shape[0])]
                        sel = elig_bg[perm]
                        seed_mask[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                        if no_overlap_after_dilation:
                            added = torch.zeros_like(lbl, dtype=torch.bool)
                            added[sel[:, 0], sel[:, 1], sel[:, 2]] = True
                            added_t = added.unsqueeze(0).unsqueeze(0).float()
                            k = 2 * dilate_radius + 1
                            dil = F.max_pool3d(added_t, kernel_size=k, stride=1, padding=dilate_radius) > 0.5
                            blocked |= dil[0, 0]
                seed_mask_t = seed_mask.unsqueeze(0).unsqueeze(0).float()
                dilated = F.max_pool3d(seed_mask_t, kernel_size=k, stride=1, padding=dilate_radius)
                sup = dilated > 0.5
                it += 1
        out_masks.append(sup.bool())
        if return_seeds:
            out_seed_masks.append(seed_mask.unsqueeze(0).unsqueeze(0))
    sup_out = torch.cat(out_masks, dim=0)
    if return_seeds:
        seed_out = torch.cat(out_seed_masks, dim=0)
        return sup_out, seed_out
    return sup_out

