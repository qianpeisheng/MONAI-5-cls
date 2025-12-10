"""
Zhou-style label diffusion on 3D voxel grids using PyTorch 3D convolutions.

This module implements the Zhou iteration:
    F^{t+1} = alpha * S F^{t} + (1 - alpha) * Y

where S encodes local voxel neighbors, implemented via conv3d instead of
an explicit sparse matrix for efficiency.
"""

import torch
import torch.nn.functional as Fnn


def build_neighbor_kernel(connectivity: int = 6, device: str = "cuda") -> torch.Tensor:
    """
    Build a base 3x3x3 kernel encoding neighbors.

    Args:
        connectivity: 6 or 26 (neighborhood type).
        device: "cuda" or "cpu".

    Returns:
        base_kernel: (1, 1, 3, 3, 3) tensor.
            - If connectivity == 6: ones at +/-x, +/-y, +/-z positions; zero at center and all others.
            - If connectivity == 26: ones at all positions except center.
    """
    # Initialize zeros tensor
    kernel = torch.zeros((1, 1, 3, 3, 3), dtype=torch.float32, device=device)

    if connectivity == 6:
        # 6-connectivity: only direct face neighbors
        # Center is at index (1, 1, 1)
        # Set +/-x, +/-y, +/-z neighbors to 1
        kernel[0, 0, 1, 1, 0] = 1.0  # -z
        kernel[0, 0, 1, 1, 2] = 1.0  # +z
        kernel[0, 0, 1, 0, 1] = 1.0  # -y
        kernel[0, 0, 1, 2, 1] = 1.0  # +y
        kernel[0, 0, 0, 1, 1] = 1.0  # -x
        kernel[0, 0, 2, 1, 1] = 1.0  # +x
    elif connectivity == 26:
        # 26-connectivity: all neighbors except center
        kernel[:, :, :, :, :] = 1.0
        kernel[0, 0, 1, 1, 1] = 0.0  # center
    else:
        raise ValueError(f"connectivity must be 6 or 26, got {connectivity}")

    return kernel


def diffuse_labels_3d(
    Y: torch.Tensor,
    labeled_mask: torch.Tensor,
    alpha: float = 0.99,
    max_iter: int = 500,
    tol: float = 1e-4,
    connectivity: int = 6,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Run Zhou-style label diffusion on a 3D voxel grid using conv3d.

    Args:
        Y: (1, C, D, H, W) float tensor of initial labels (one-hot or soft).
           Labeled voxels: one-hot / soft distributions.
           Unlabeled voxels: all zeros in Y.
        labeled_mask: (1, 1, D, H, W) bool or {0,1} tensor indicating labeled voxels.
        alpha: propagation parameter in [0, 1).
        max_iter: max diffusion iterations.
        tol: stop when ||F_{t+1} - F_t||_F < tol.
        connectivity: 6 or 26 (neighborhood).
        device: "cuda" or "cpu".

    Returns:
        F: (1, C, D, H, W) tensor of final diffused label scores.
    """
    # Move to device and cast to float32
    Y = Y.to(device=device, dtype=torch.float32)
    labeled_mask = labeled_mask.to(device=device, dtype=torch.float32)

    # Extract dimensions
    batch, C, D, H, W = Y.shape
    assert batch == 1, f"Expected batch size 1, got {batch}"

    # Build neighbor kernel (base kernel for one channel)
    base_kernel = build_neighbor_kernel(connectivity=connectivity, device=device)

    # Expand kernel to apply same operation to each channel independently
    # Shape: (C, 1, 3, 3, 3) for use with groups=C
    kernel = base_kernel.repeat(C, 1, 1, 1, 1)

    # Compute degree map: number of neighbors for each voxel
    # This accounts for boundary effects automatically
    ones = torch.ones((1, 1, D, H, W), device=device, dtype=torch.float32)
    deg_map = Fnn.conv3d(ones, base_kernel, padding=1)
    deg_map = torch.clamp(deg_map, min=1.0)  # avoid division by zero

    # Initialize F = Y
    F = Y.clone()

    # Iterative diffusion
    for iteration in range(max_iter):
        # Compute neighbor sum for each channel independently
        # groups=C means each channel uses its own kernel (but they're all the same)
        neighbor_sum = Fnn.conv3d(F, kernel, padding=1, groups=C)

        # Normalize by degree to get S * F
        # deg_map shape (1, 1, D, H, W) broadcasts across channels
        S_F = neighbor_sum / deg_map

        # Zhou update: F_next = alpha * S * F + (1 - alpha) * Y
        F_next = alpha * S_F + (1 - alpha) * Y

        # Clamp labeled voxels to their original values
        # labeled_mask shape (1, 1, D, H, W) broadcasts across channels
        F_next = torch.where(labeled_mask.bool(), Y, F_next)

        # Check convergence
        diff = torch.norm(F_next - F, p='fro')

        # Update F for next iteration
        F = F_next

        if diff < tol:
            break

    return F
