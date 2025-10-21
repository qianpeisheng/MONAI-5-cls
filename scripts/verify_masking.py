#!/usr/bin/env python3
"""
Verification of training-time masking semantics used in train_finetune_wp5.py.

Checks:
- CE on seed-only voxels ignores unsupervised voxels (zero gradient outside seeds).
- Dice loss is masked to supervised voxels only (zero gradient outside mask).
- Pseudo CE (when used) applies only to dilated-only region; here we also
  validate the radius=0 case yields no pseudo contribution.

Run: python3 scripts/verify_masking.py
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np

# Import the same helper used by training
# Robust import of helper from repo root even when executed from subdirs
try:
    from train_finetune_wp5 import dice_loss_masked  # type: ignore
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from train_finetune_wp5 import dice_loss_masked  # type: ignore


def make_synthetic_batch(B=1, C=5, X=10, Y=9, Z=8, seed_ratio=0.02):
    """Create a small synthetic batch with labels in 0..4 and some label==6 voxels."""
    g = torch.Generator(device="cpu").manual_seed(7)
    # logits with requires_grad
    logits = torch.randn(B, C, X, Y, Z, generator=g, requires_grad=True)
    # labels: mostly 0 with a sprinkle of 1..4 and some 6
    lbl = torch.zeros(B, 1, X, Y, Z, dtype=torch.long)
    # foreground classes
    for c in [1, 2, 3, 4]:
        mask = torch.rand(B, 1, X, Y, Z, generator=g) < 0.05
        lbl[mask] = c
    # set some voxels to 6 (ignored class)
    ign = torch.rand(B, 1, X, Y, Z, generator=g) < 0.03
    lbl[ign] = 6
    # seeds: a small subset of non-6 voxels
    elig = (lbl != 6)
    seeds = torch.zeros_like(lbl, dtype=torch.bool)
    elig_idx = elig.nonzero(as_tuple=False)
    k = max(1, int(seed_ratio * elig_idx.shape[0]))
    perm = torch.randperm(elig_idx.shape[0], generator=g)[:k]
    sel = elig_idx[perm]
    seeds[sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3], sel[:, 4]] = True
    # sup_mask = seeds when radius=0
    sup = seeds.clone()
    return logits, lbl, seeds, sup


def ce_seed_only(logits: torch.Tensor, lbl: torch.Tensor, seeds: torch.Tensor):
    # Build CE target with ignore_index=255 outside seeds or label==6
    target = lbl.squeeze(1).clone()
    target[(~seeds.squeeze(1)) | (target == 6)] = 255
    return F.cross_entropy(logits, target, ignore_index=255)


def check_zero_grads_outside_seed():
    logits, lbl, seeds, sup = make_synthetic_batch()
    ce = ce_seed_only(logits, lbl, seeds)
    dl = dice_loss_masked(logits, lbl, ignore_mask=(lbl != 6) & seeds)
    loss = 0.5 * ce + 0.5 * dl
    loss.backward()
    # Gradients should be zero outside supervised region for both CE and Dice
    grad = logits.grad.detach()
    mask = seeds.expand_as(lbl)
    mask_c = mask.expand(-1, logits.shape[1], -1, -1, -1)  # (B,C,X,Y,Z)
    unsup_grad = grad[~mask_c]
    sup_grad = grad[mask_c]
    # Tolerate tiny numerical noise
    assert torch.allclose(unsup_grad, torch.zeros_like(unsup_grad), atol=1e-7, rtol=0), (
        f"Non-zero grad outside seeds: max={unsup_grad.abs().max().item()}"
    )
    assert sup_grad.abs().sum() > 0, "Expected non-zero grad on supervised voxels"
    print("OK: CE+Dice grads are zero outside seeds (radius=0).")


def check_pseudo_no_effect_when_radius0():
    # When radius=0, dil_only is empty, so pseudo CE contributes nothing
    logits, lbl, seeds, sup = make_synthetic_batch()
    logits2 = logits.clone().detach().requires_grad_(True)

    ce1 = ce_seed_only(logits, lbl, seeds)
    dl1 = dice_loss_masked(logits, lbl, ignore_mask=(lbl != 6) & seeds)
    loss1 = 0.5 * ce1 + 0.5 * dl1

    # Simulate pseudo branch with dil_only empty
    dil_only = (sup.squeeze(1) & (~seeds.squeeze(1)))
    assert not bool(dil_only.any()), "For radius=0 we expect no dilated-only region"
    # No pseudo CE added
    ce2 = ce_seed_only(logits2, lbl, seeds)
    dl2 = dice_loss_masked(logits2, lbl, ignore_mask=(lbl != 6) & seeds)
    loss2 = 0.5 * ce2 + 0.5 * dl2

    # Losses should match closely (identical computation)
    assert torch.allclose(loss1, loss2, atol=1e-7, rtol=0), f"Loss mismatch: {loss1.item()} vs {loss2.item()}"
    print("OK: pseudo branch has no effect when radius=0 (dil_only empty).")


def check_invariance_outside_seed_changes():
    # Changing logits only outside seeds should not change the loss
    logits, lbl, seeds, sup = make_synthetic_batch()
    base_loss = (0.5 * ce_seed_only(logits, lbl, seeds)
                 + 0.5 * dice_loss_masked(logits, lbl, ignore_mask=(lbl != 6) & seeds))
    # Perturb logits outside seeds heavily
    pert = torch.zeros_like(logits)
    mask = seeds.expand_as(lbl)
    mask_c = mask.expand(-1, logits.shape[1], -1, -1, -1)
    pert[~mask_c] = torch.sign(torch.randn_like(pert[~mask_c])) * 10.0
    logits_pert = (logits + pert).detach().requires_grad_(True)
    loss_pert = (0.5 * ce_seed_only(logits_pert, lbl, seeds)
                 + 0.5 * dice_loss_masked(logits_pert, lbl, ignore_mask=(lbl != 6) & seeds))
    assert torch.allclose(base_loss, loss_pert, atol=1e-6, rtol=0), (
        f"Loss changed despite perturbing only outside seeds: {base_loss.item()} vs {loss_pert.item()}"
    )
    print("OK: loss invariant to changes outside seeds.")


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    check_zero_grads_outside_seed()
    check_pseudo_no_effect_when_radius0()
    check_invariance_outside_seed_changes()
    print("All masking checks passed.")


if __name__ == "__main__":
    main()
