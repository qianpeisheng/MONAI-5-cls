from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np

from .geom_utils import (
    boundary_mask,
    distance_inside,
    distance_outside,
    estimate_surface_weight,
    farthest_point_sampling,
    interior_mask,
    shell_inside,
    shell_outside,
)


@dataclass
class PointsForClass:
    # voxel coordinates (N,3) int, and signed distances (N,) float
    coords: np.ndarray
    sdf: np.ndarray
    # optional grouping id per point (e.g., connected component id for class 3)
    groups: np.ndarray | None = None


def _split_budget(n: int, boundary_ratio: float, interior_ratio: float, guard_ratio: float) -> Tuple[int, int, int, int]:
    n_b = int(round(n * boundary_ratio))
    n_i = int(round(n * interior_ratio))
    n_g = max(0, n - n_b - n_i)
    n_gi = n_g // 2
    n_go = n_g - n_gi
    return n_b, n_i, n_gi, n_go


def select_points_1pct(
    label: np.ndarray,
    classes: Tuple[int, ...] = (1, 2, 3, 4),
    budget_ratio: float = 0.01,
    # background (class 0) support
    include_background: bool = True,
    bg_ratio: float = 0.05,
    boundary_ratio: float = 0.7,
    interior_ratio: float = 0.2,
    guard_ratio: float = 0.1,
    margin_vox: float = 1.5,
    rng: np.random.RandomState | None = None,
) -> Tuple[Dict[int, PointsForClass], np.ndarray, Dict[int, Dict[str, int]]]:
    """Select sparse labeled points per class with distances for SDF fitting.

    Returns:
      - dict: class -> PointsForClass (coords int (N,3), sdf float (N,))
      - mask_selected_points: int32 volume with selected interior/boundary points marked by class id (no outside guards)
      - counts per class for reporting
    """
    if rng is None:
        rng = np.random.RandomState(42)

    shape = label.shape
    selected_mask = np.zeros(shape, dtype=np.int16)

    # Compute per-class weights (surface proxy)
    weights: Dict[int, float] = {}
    present: List[int] = []
    for c in classes:
        m = (label == c)
        if m.any():
            weights[c] = estimate_surface_weight(m)
            present.append(c)
        else:
            weights[c] = 0.0

    # Budget over entire volume voxels
    total_vox = int(np.prod(shape))
    total_pts = max(50, int(round(max(1e-6, budget_ratio) * total_vox)))
    total_pts = max(total_pts, len(present) * 10)  # minimum per-present-class quota later

    # Split budget: background vs foreground classes
    bg_pts_total = int(round(total_pts * (bg_ratio if include_background else 0.0)))
    fg_pts_total = max(0, total_pts - bg_pts_total)
    wsum = sum(weights[c] for c in present) or 1.0

    out: Dict[int, PointsForClass] = {}
    stats: Dict[int, Dict[str, int]] = {}

    # Precompute distance maps per class
    dist_inside: Dict[int, np.ndarray] = {}
    dist_outside: Dict[int, np.ndarray] = {}
    for c in classes:
        m = (label == c)
        if not m.any():
            continue
        dist_inside[c] = distance_inside(m)
        dist_outside[c] = distance_outside(m)

    # Optional: background far-from-FG negatives to help shape class SDFs
    bg_far_coords = np.zeros((0, 3), dtype=int)
    if include_background:
        bg_mask = (label == 0)
        fg_mask = (label != 0) & (label != 6)
        # distance from background to foreground
        dist_to_fg = distance_outside(fg_mask)
        thr = max(2, int(round(margin_vox * 2)))
        bg_far = bg_mask & (dist_to_fg >= thr)
        if bg_far.any():
            coords = np.argwhere(bg_far)
            # Sample up to bg_pts_total (if 0, still allow a small handful)
            n_bg_far = max(0, int(round(bg_pts_total * 0.5)))
            if n_bg_far > 0:
                idx = rng.choice(coords.shape[0], size=min(n_bg_far, coords.shape[0]), replace=False)
                bg_far_coords = coords[idx]

    # Order classes by ascending GT voxel count (present classes only)
    class_sizes = {c: int((label == c).sum()) for c in present}
    order = sorted(present, key=lambda x: class_sizes[x])

    # Allocate FG per class (min quota of 10 per present class)
    base_alloc = {c: 10 for c in order}
    rem = max(0, fg_pts_total - sum(base_alloc.values()))
    # Distribute remainder by weights
    if rem > 0:
        alloc_extra = {c: int(np.floor(rem * (weights[c] / wsum))) for c in order}
        extra_used = sum(alloc_extra.values())
        # distribute leftover by descending weights
        leftover = rem - extra_used
        for c in sorted(order, key=lambda x: -weights[x])[: leftover]:
            alloc_extra[c] += 1
    else:
        alloc_extra = {c: 0 for c in order}

    for c in order:
        m = (label == c)
        if not m.any():
            continue
        n_c = base_alloc[c] + alloc_extra[c]
        n_b, n_i, n_gi, n_go = _split_budget(n_c, boundary_ratio, interior_ratio, guard_ratio)
        # For class 3 (void), operate per-connected-component to avoid merging instances
        if c == 3:
            from scipy.ndimage import label as ndi_label, generate_binary_structure
            structure = generate_binary_structure(rank=3, connectivity=1)
            lab, ncomp = ndi_label(m.astype(np.uint8), structure=structure)
            # Allocate per-component by boundary weight
            comp_weights = []
            comp_masks = []
            for cid in range(1, ncomp + 1):
                cm = (lab == cid)
                comp_masks.append(cm)
                comp_weights.append(estimate_surface_weight(cm))
            comp_weights = np.array(comp_weights, dtype=float)
            wsum_c = comp_weights.sum() or 1.0
            # Split budgets
            def split_k(total):
                base = np.floor(total * (comp_weights / wsum_c)).astype(int)
                remk = total - base.sum()
                orderk = np.argsort(-comp_weights)
                for i in range(remk):
                    base[orderk[i % len(orderk)]] += 1
                return base

            nb_s = split_k(n_b)
            ni_s = split_k(n_i)
            ngi_s = split_k(n_gi)
            ngo_s = split_k(n_go)

            coords_all = []
            sdf_all = []
            groups_all = []
            bsel_total = isel_total = gisel_total = gosel_total = 0
            for cid, cm in enumerate(comp_masks, start=1):
                bmask = boundary_mask(cm)
                imask = interior_mask(cm, thickness=int(max(1, int(round(margin_vox)))))
                in_shell = shell_inside(cm, thickness=int(max(1, int(round(margin_vox)))))
                out_shell = shell_outside(cm, thickness=int(max(1, int(round(margin_vox)))))
                # Restrict guard-out for void to be within parent class-2 region to reduce bridging
                out_shell = out_shell & (label == 2)

                bcoords = np.argwhere(bmask)
                if bcoords.size > 0 and nb_s[cid - 1] > 0:
                    bsel = farthest_point_sampling(bcoords, int(nb_s[cid - 1]), rng)
                else:
                    bsel = bcoords[:0]

                icoords = np.argwhere(imask)
                if icoords.size > 0 and ni_s[cid - 1] > 0:
                    idx = rng.choice(icoords.shape[0], size=min(int(ni_s[cid - 1]), icoords.shape[0]), replace=False)
                    isel = icoords[idx]
                else:
                    isel = icoords[:0]

                gicoords = np.argwhere(in_shell)
                if gicoords.size > 0 and ngi_s[cid - 1] > 0:
                    gi = rng.choice(gicoords.shape[0], size=min(int(ngi_s[cid - 1]), gicoords.shape[0]), replace=False)
                    gisel = gicoords[gi]
                else:
                    gisel = gicoords[:0]
                gocoords = np.argwhere(out_shell)
                if gocoords.size > 0 and ngo_s[cid - 1] > 0:
                    go = rng.choice(gocoords.shape[0], size=min(int(ngo_s[cid - 1]), gocoords.shape[0]), replace=False)
                    gosel = gocoords[go]
                else:
                    gosel = gocoords[:0]

                dins = dist_inside[c]
                douts = dist_outside[c]
                sdf_b = np.zeros((bsel.shape[0],), dtype=np.float32)
                sdf_i = dins[isel[:, 0], isel[:, 1], isel[:, 2]].astype(np.float32)
                sdf_gi = dins[gisel[:, 0], gisel[:, 1], gisel[:, 2]].astype(np.float32)
                sdf_go = -douts[gosel[:, 0], gosel[:, 1], gosel[:, 2]].astype(np.float32)

                coords = (
                    np.concatenate([bsel, isel, gisel, gosel], axis=0)
                    if bsel.size or isel.size or gisel.size or gosel.size
                    else np.zeros((0, 3), dtype=int)
                )
                sdf = (
                    np.concatenate([sdf_b, sdf_i, sdf_gi, sdf_go], axis=0)
                    if coords.shape[0] > 0
                    else np.zeros((0,), dtype=np.float32)
                )

                if coords.shape[0] > 0:
                    coords_all.append(coords)
                    sdf_all.append(sdf)
                    groups_all.append(np.full((coords.shape[0],), cid, dtype=np.int32))

                # Mark selected points for overlay
                for arr in (bsel, isel, gisel):
                    if arr.size:
                        selected_mask[arr[:, 0], arr[:, 1], arr[:, 2]] = int(c)
                bsel_total += int(bsel.shape[0])
                isel_total += int(isel.shape[0])
                gisel_total += int(gisel.shape[0])
                gosel_total += int(gosel.shape[0])

            if coords_all:
                coords = np.concatenate(coords_all, axis=0)
                sdf = np.concatenate(sdf_all, axis=0)
                groups = np.concatenate(groups_all, axis=0)
            else:
                coords = np.zeros((0, 3), dtype=int)
                sdf = np.zeros((0,), dtype=np.float32)
                groups = np.zeros((0,), dtype=np.int32)

            # Append background far negatives
            if bg_far_coords.shape[0] > 0:
                n_bg_extra = max(0, int(round(bg_far_coords.shape[0] / max(1, len(order)))))
                if n_bg_extra > 0:
                    if bg_far_coords.shape[0] > n_bg_extra:
                        idx = rng.choice(bg_far_coords.shape[0], size=n_bg_extra, replace=False)
                        farc = bg_far_coords[idx]
                    else:
                        farc = bg_far_coords
                    far_sdf = -dist_outside[c][farc[:, 0], farc[:, 1], farc[:, 2]].astype(np.float32)
                    coords = np.concatenate([coords, farc], axis=0)
                    sdf = np.concatenate([sdf, far_sdf], axis=0)
                    groups = np.concatenate([groups, np.zeros((farc.shape[0],), dtype=np.int32)], axis=0)

            out[c] = PointsForClass(coords=coords.astype(np.int32), sdf=sdf, groups=groups)
            stats[c] = {
                "total": int(coords.shape[0]),
                "boundary": int(bsel_total),
                "interior": int(isel_total),
                "guard_in": int(gisel_total),
                "guard_out": int(gosel_total),
            }
            continue

        # General case for classes != 3
        bmask = boundary_mask(m)
        imask = interior_mask(m, thickness=int(max(1, int(round(margin_vox)))))
        in_shell = shell_inside(m, thickness=int(max(1, int(round(margin_vox)))))
        out_shell = shell_outside(m, thickness=int(max(1, int(round(margin_vox)))))

        # Sample boundary with FPS for better coverage
        bcoords = np.argwhere(bmask)
        if bcoords.size > 0 and n_b > 0:
            bsel = farthest_point_sampling(bcoords, n_b, rng)
        else:
            bsel = bcoords[:0]

        icoords = np.argwhere(imask)
        if icoords.size > 0 and n_i > 0:
            idx = rng.choice(icoords.shape[0], size=min(n_i, icoords.shape[0]), replace=False)
            isel = icoords[idx]
        else:
            isel = icoords[:0]

        # Guard points inside/outside near boundary shells
        gicoords = np.argwhere(in_shell)
        if gicoords.size > 0 and n_gi > 0:
            gi = rng.choice(gicoords.shape[0], size=min(n_gi, gicoords.shape[0]), replace=False)
            gisel = gicoords[gi]
        else:
            gisel = gicoords[:0]
        gocoords = np.argwhere(out_shell)
        if gocoords.size > 0 and n_go > 0:
            go = rng.choice(gocoords.shape[0], size=min(n_go, gocoords.shape[0]), replace=False)
            gosel = gocoords[go]
        else:
            gosel = gocoords[:0]

        # Distances
        dins = dist_inside[c]
        douts = dist_outside[c]
        # Boundary SDF ~ 0
        sdf_b = np.zeros((bsel.shape[0],), dtype=np.float32)
        # Interior positive distances
        sdf_i = dins[isel[:, 0], isel[:, 1], isel[:, 2]].astype(np.float32)
        sdf_gi = dins[gisel[:, 0], gisel[:, 1], gisel[:, 2]].astype(np.float32)
        # Outside negative distances
        sdf_go = -douts[gosel[:, 0], gosel[:, 1], gosel[:, 2]].astype(np.float32)

        coords = (
            np.concatenate([bsel, isel, gisel, gosel], axis=0) if bsel.size or isel.size or gisel.size or gosel.size else np.zeros((0, 3), dtype=int)
        )
        sdf = np.concatenate([sdf_b, sdf_i, sdf_gi, sdf_go], axis=0) if coords.shape[0] > 0 else np.zeros((0,), dtype=np.float32)

        # Append background far negatives (help expand negative region far from boundaries)
        if bg_far_coords.shape[0] > 0:
            # take a small share per class
            n_bg_extra = max(0, int(round(bg_far_coords.shape[0] / max(1, len(order)))))
            if n_bg_extra > 0:
                if bg_far_coords.shape[0] > n_bg_extra:
                    idx = rng.choice(bg_far_coords.shape[0], size=n_bg_extra, replace=False)
                    farc = bg_far_coords[idx]
                else:
                    farc = bg_far_coords
                far_sdf = -douts[farc[:, 0], farc[:, 1], farc[:, 2]].astype(np.float32)
                coords = np.concatenate([coords, farc], axis=0)
                sdf = np.concatenate([sdf, far_sdf], axis=0)

        out[c] = PointsForClass(coords=coords.astype(np.int32), sdf=sdf, groups=None)

        # Mark only interior+boundary (not outside guards) in selected mask
        for arr in (bsel, isel, gisel):
            if arr.size:
                selected_mask[arr[:, 0], arr[:, 1], arr[:, 2]] = int(c)

        stats[c] = {
            "total": int(coords.shape[0]),
            "boundary": int(bsel.shape[0]),
            "interior": int(isel.shape[0]),
            "guard_in": int(gisel.shape[0]),
            "guard_out": int(gosel.shape[0]),
        }

    # Optional background sampling (lightweight) for verification overlays only
    if include_background and bg_pts_total > 0:
        c = 0
        m0 = (label == 0)
        if m0.any():
            # distribute bg budget: mostly boundary + some interior
            n_b0, n_i0, n_gi0, n_go0 = _split_budget(bg_pts_total, boundary_ratio, interior_ratio, guard_ratio)
            b0 = boundary_mask(m0)
            i0 = interior_mask(m0, thickness=int(max(1, int(round(margin_vox)))))
            gi0 = shell_inside(m0, thickness=int(max(1, int(round(margin_vox)))))
            # guard_out for background corresponds to outside shell (i.e., near FG); keep small
            go0 = shell_outside(m0, thickness=int(max(1, int(round(margin_vox)))))

            bcoords = np.argwhere(b0)
            isel = np.argwhere(i0)
            gi_sel = np.argwhere(gi0)
            go_sel = np.argwhere(go0)
            def _pick(coords, k):
                if coords.size == 0 or k <= 0:
                    return coords[:0]
                idx = rng.choice(coords.shape[0], size=min(k, coords.shape[0]), replace=False)
                return coords[idx]
            bsel0 = _pick(bcoords, n_b0)
            isel0 = _pick(isel, n_i0)
            gis0 = _pick(gi_sel, n_gi0)
            gos0 = _pick(go_sel, n_go0)

            # Mark background points into selected_mask for overlay purposes
            for arr in (bsel0, isel0, gis0):
                if arr.size:
                    selected_mask[arr[:, 0], arr[:, 1], arr[:, 2]] = 0
            stats[c] = {
                "total": int(bsel0.shape[0] + isel0.shape[0] + gis0.shape[0] + gos0.shape[0]),
                "boundary": int(bsel0.shape[0]),
                "interior": int(isel0.shape[0]),
                "guard_in": int(gis0.shape[0]),
                "guard_out": int(gos0.shape[0]),
            }

    return out, selected_mask, stats
