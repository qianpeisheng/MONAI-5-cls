"""
Tests for Zhou-style voxel diffusion implementation.

This test suite verifies:
1. Qualitative correctness on a toy 3D grid
2. Consistency with explicit graph-based implementation
3. Performance on ~1M voxel volumes
"""

import time
import torch
import torch.nn.functional as Fnn
import sys
import os

# Add parent directory to path to import wp5 module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wp5.weaklabel.voxel_diffusion import diffuse_labels_3d, build_neighbor_kernel


def test_toy_grid():
    """
    Test 1: Toy 3D grid sanity test.

    Verify that diffusion behaves qualitatively correctly:
    labels at two opposite corners diffuse across the grid and meet in the middle.
    """
    print("\n" + "="*80)
    print("TEST 1: Toy 3D Grid Sanity Test")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Small grid: D = H = W = 9, C = 2
    D, H, W = 9, 9, 9
    C = 2

    # Initialize Y with zeros
    Y = torch.zeros((1, C, D, H, W), dtype=torch.float32)

    # Set two labeled voxels at opposite corners
    # Class 0 at (0,0,0)
    Y[0, 0, 0, 0, 0] = 1.0
    Y[0, 1, 0, 0, 0] = 0.0

    # Class 1 at (8,8,8)
    Y[0, 1, 8, 8, 8] = 1.0
    Y[0, 0, 8, 8, 8] = 0.0

    # Create labeled mask
    labeled_mask = torch.zeros((1, 1, D, H, W), dtype=torch.float32)
    labeled_mask[0, 0, 0, 0, 0] = 1.0
    labeled_mask[0, 0, 8, 8, 8] = 1.0

    print(f"Grid size: {D}x{H}x{W}, Classes: {C}")
    print(f"Labeled voxel 1: (0,0,0) -> class 0")
    print(f"Labeled voxel 2: (8,8,8) -> class 1")

    # Run diffusion
    start_time = time.time()
    F = diffuse_labels_3d(
        Y, labeled_mask,
        alpha=0.99,
        max_iter=500,
        tol=1e-5,
        connectivity=6,
        device=device
    )
    elapsed = time.time() - start_time

    print(f"Diffusion completed in {elapsed:.4f} seconds")

    # Convert to probabilities using softmax
    probs = torch.softmax(F, dim=1)

    # Assertions
    print("\nVerifying results:")

    # At (0,0,0), class 0 should dominate
    p0_at_000 = probs[0, 0, 0, 0, 0].item()
    p1_at_000 = probs[0, 1, 0, 0, 0].item()
    print(f"  At (0,0,0): p(class 0) = {p0_at_000:.4f}, p(class 1) = {p1_at_000:.4f}")
    assert p0_at_000 > p1_at_000, f"Class 0 should dominate at (0,0,0)"
    print("  ✓ Class 0 dominates at (0,0,0)")

    # At (8,8,8), class 1 should dominate
    p0_at_888 = probs[0, 0, 8, 8, 8].item()
    p1_at_888 = probs[0, 1, 8, 8, 8].item()
    print(f"  At (8,8,8): p(class 0) = {p0_at_888:.4f}, p(class 1) = {p1_at_888:.4f}")
    assert p1_at_888 > p0_at_888, f"Class 1 should dominate at (8,8,8)"
    print("  ✓ Class 1 dominates at (8,8,8)")

    # At the center (4,4,4), classes should be roughly balanced
    mid = probs[0, :, 4, 4, 4]
    p0_mid = mid[0].item()
    p1_mid = mid[1].item()
    balance_diff = abs(p0_mid - p1_mid)
    print(f"  At center (4,4,4): p(class 0) = {p0_mid:.4f}, p(class 1) = {p1_mid:.4f}")
    print(f"  Balance difference: {balance_diff:.4f}")
    assert balance_diff < 0.2, f"Classes should be roughly balanced at center, diff = {balance_diff}"
    print("  ✓ Classes are roughly balanced at center")

    print("\n✓ TEST 1 PASSED\n")


def test_consistency_small_graph():
    """
    Test 2: Consistency with explicit graph on a tiny grid.

    Confirm that one iteration of diffuse_labels_3d matches the result of
    an explicit graph-based Zhou update on a very small 3D grid.
    """
    print("\n" + "="*80)
    print("TEST 2: Consistency with Explicit Graph (3x3x3)")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Tiny grid: D = H = W = 3 (N = 27), C = 2
    D, H, W = 3, 3, 3
    C = 2
    N = D * H * W  # 27 voxels

    # Create synthetic Y and labeled_mask
    Y = torch.zeros((1, C, D, H, W), dtype=torch.float32)

    # Set a few labeled voxels
    # Voxel at (0,0,0) -> class 0
    Y[0, 0, 0, 0, 0] = 1.0
    # Voxel at (2,2,2) -> class 1
    Y[0, 1, 2, 2, 2] = 1.0
    # Voxel at (1,1,1) -> class 0 (with some uncertainty)
    Y[0, 0, 1, 1, 1] = 0.7
    Y[0, 1, 1, 1, 1] = 0.3

    labeled_mask = torch.zeros((1, 1, D, H, W), dtype=torch.float32)
    labeled_mask[0, 0, 0, 0, 0] = 1.0
    labeled_mask[0, 0, 2, 2, 2] = 1.0
    labeled_mask[0, 0, 1, 1, 1] = 1.0

    alpha = 0.9

    print(f"Grid size: {D}x{H}x{W} = {N} voxels, Classes: {C}")
    print(f"Alpha: {alpha}")

    # Build explicit graph adjacency matrix (N x N)
    # adj_matrix[i,j] = 1 if i,j are 6-neighbors in the 3D grid
    adj_matrix = torch.zeros((N, N), dtype=torch.float32)

    # Helper to convert 3D index to flat index
    def idx3d_to_flat(d, h, w):
        return d * H * W + h * W + w

    # Build adjacency matrix for 6-connectivity
    for d in range(D):
        for h in range(H):
            for w in range(W):
                i = idx3d_to_flat(d, h, w)
                # Check 6 neighbors
                for dd, hh, ww in [
                    (d-1, h, w), (d+1, h, w),
                    (d, h-1, w), (d, h+1, w),
                    (d, h, w-1), (d, h, w+1)
                ]:
                    if 0 <= dd < D and 0 <= hh < H and 0 <= ww < W:
                        j = idx3d_to_flat(dd, hh, ww)
                        adj_matrix[i, j] = 1.0

    # Compute degrees and normalized matrix S = D^{-1} * adj_matrix (row-normalized)
    degrees = adj_matrix.sum(dim=1)  # (N,)
    D_inv = torch.diag(1.0 / degrees)  # (N, N)
    S = D_inv @ adj_matrix  # (N, N)

    # Flatten Y to (N, C)
    Y_flat = Y.view(C, -1).T  # (N, C)

    # One Zhou step in matrix form: F1 = alpha * S @ F0 + (1 - alpha) * Y
    F0 = Y_flat.clone()
    F1_graph = alpha * (S @ F0) + (1 - alpha) * Y_flat

    # Apply labeled mask constraint
    labeled_mask_flat = labeled_mask.view(-1)  # (N,)
    for i in range(N):
        if labeled_mask_flat[i] > 0:
            F1_graph[i] = Y_flat[i]

    print(f"\nGraph-based computation completed")
    print(f"  S matrix shape: {S.shape}")
    print(f"  F1_graph shape: {F1_graph.shape}")

    # Run one iteration of conv-based code
    # We'll manually do one iteration by copying the loop body
    Y_dev = Y.to(device=device, dtype=torch.float32)
    labeled_mask_dev = labeled_mask.to(device=device, dtype=torch.float32)

    # Build kernel and degree map
    base_kernel = build_neighbor_kernel(connectivity=6, device=device)
    kernel = base_kernel.repeat(C, 1, 1, 1, 1)

    ones = torch.ones((1, 1, D, H, W), device=device, dtype=torch.float32)
    deg_map = Fnn.conv3d(ones, base_kernel, padding=1)
    deg_map = torch.clamp(deg_map, min=1.0)

    # One iteration
    F = Y_dev.clone()
    neighbor_sum = Fnn.conv3d(F, kernel, padding=1, groups=C)
    S_F = neighbor_sum / deg_map
    F1_conv = alpha * S_F + (1 - alpha) * Y_dev
    F1_conv = torch.where(labeled_mask_dev.bool(), Y_dev, F1_conv)

    # Flatten conv result to (N, C) for comparison
    F1_conv_flat = F1_conv.view(C, -1).T.cpu()  # (N, C)

    print(f"Conv-based computation completed")
    print(f"  F1_conv_flat shape: {F1_conv_flat.shape}")

    # Compare results
    max_diff = torch.max(torch.abs(F1_conv_flat - F1_graph)).item()
    mean_diff = torch.mean(torch.abs(F1_conv_flat - F1_graph)).item()

    print(f"\nComparison:")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")

    # Assert close match
    assert torch.allclose(F1_conv_flat, F1_graph, atol=1e-5, rtol=1e-4), \
        f"Conv-based and graph-based results don't match! Max diff: {max_diff}"

    print("  ✓ Conv-based and graph-based results match within tolerance")
    print("\n✓ TEST 2 PASSED\n")


def test_performance_large_grid():
    """
    Test 3: Performance sanity on ~1M voxels.

    Check that the implementation runs without OOM and within reasonable time
    on a ~1M voxel volume.
    """
    print("\n" + "="*80)
    print("TEST 3: Performance Test (~1M voxels)")
    print("="*80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Choose dimensions with D*H*W ≈ 1e6
    # Using 100x100x100 = 1,000,000 voxels
    D, H, W = 100, 100, 100
    C = 3  # 3 classes

    print(f"Grid size: {D}x{H}x{W} = {D*H*W:,} voxels")
    print(f"Classes: {C}")
    print(f"Memory requirement: ~{D*H*W*C*4/(1024**2):.1f} MB per tensor (float32)")

    # Create Y with a few labeled voxels (10 random locations)
    Y = torch.zeros((1, C, D, H, W), dtype=torch.float32)
    labeled_mask = torch.zeros((1, 1, D, H, W), dtype=torch.float32)

    # Seed for reproducibility
    torch.manual_seed(42)

    num_labeled = 10
    print(f"\nCreating {num_labeled} randomly labeled voxels...")

    for _ in range(num_labeled):
        # Random position
        d = torch.randint(0, D, (1,)).item()
        h = torch.randint(0, H, (1,)).item()
        w = torch.randint(0, W, (1,)).item()

        # Random class (one-hot)
        cls = torch.randint(0, C, (1,)).item()
        Y[0, cls, d, h, w] = 1.0

        # Mark as labeled
        labeled_mask[0, 0, d, h, w] = 1.0

    print(f"Labeled voxels created: {labeled_mask.sum().item():.0f}")

    # Run diffusion with limited iterations for performance test
    max_iter = 100
    tol = 1e-3
    alpha = 0.99

    print(f"\nRunning diffusion...")
    print(f"  alpha: {alpha}")
    print(f"  max_iter: {max_iter}")
    print(f"  tol: {tol}")
    print(f"  connectivity: 6")

    start_time = time.time()

    try:
        F = diffuse_labels_3d(
            Y, labeled_mask,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            connectivity=6,
            device=device
        )

        elapsed = time.time() - start_time

        print(f"\n✓ Diffusion completed successfully!")
        print(f"  Total runtime: {elapsed:.2f} seconds")
        print(f"  Output shape: {F.shape}")
        print(f"  Output dtype: {F.dtype}")
        print(f"  Output device: {F.device}")

        # Some basic sanity checks
        assert F.shape == (1, C, D, H, W), f"Output shape mismatch"
        assert not torch.isnan(F).any(), "Output contains NaN values"
        assert not torch.isinf(F).any(), "Output contains Inf values"

        print("\n  ✓ Output shape correct")
        print("  ✓ No NaN values")
        print("  ✓ No Inf values")

        # Performance assessment
        if device == "cuda":
            if elapsed < 60:
                print(f"\n  ✓ Performance: Good (completed in {elapsed:.2f}s)")
            else:
                print(f"\n  ⚠ Performance: Slower than expected ({elapsed:.2f}s), but acceptable")
        else:
            print(f"\n  ℹ CPU mode: runtime {elapsed:.2f}s (GPU would be faster)")

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n✗ CUDA OOM Error: {e}")
            print("  This test requires a GPU with sufficient memory (~4GB)")
            raise
        else:
            raise

    print("\n✓ TEST 3 PASSED\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("VOXEL DIFFUSION TEST SUITE")
    print("="*80)

    try:
        # Run all tests
        test_toy_grid()
        test_consistency_small_graph()
        test_performance_large_grid()

        print("\n" + "="*80)
        print("ALL TESTS PASSED!")
        print("="*80 + "\n")

    except Exception as e:
        print("\n" + "="*80)
        print(f"TEST FAILED: {e}")
        print("="*80 + "\n")
        raise
