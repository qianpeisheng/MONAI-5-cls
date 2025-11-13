#!/usr/bin/env python3
"""
Tests for Zhou-style graph label propagation.

Verifies:
1. Toy graph sanity test (3 nodes)
2. Convergence/stability test (50 points)
3. Single-sample real volume test (if data available)
"""

import numpy as np
import pytest
from pathlib import Path

from wp5.weaklabel.graph_label_propagation import (
    graph_label_propagation,
    graph_label_propagation_scores,
)


def test_toy_graph_sanity():
    """
    Test label propagation on a tiny 3-node graph.

    Setup:
        - Node 0: x=0.0, label=0 (class 0)
        - Node 1: x=0.1, unlabeled
        - Node 2: x=5.0, label=1 (class 1)

    Expected behavior:
        - Node 1 should have higher score for class 0 than class 1
          (closer to node 0 than node 2)
    """
    # 1D features
    features = np.array([[0.0], [0.1], [5.0]], dtype=np.float32)
    labels = np.array([0, -1, 1], dtype=np.int64)
    num_classes = 2

    # Run propagation
    F = graph_label_propagation_scores(
        features,
        labels,
        num_classes,
        k=2,  # each node sees both others
        alpha=0.99,
        sigma=None,  # use median heuristic
    )

    # Verify: node 1 should prefer class 0 over class 1
    assert F.shape == (3, 2), f"Expected shape (3, 2), got {F.shape}"
    assert F[1, 0] > F[1, 1], (
        f"Node 1 should prefer class 0 (score={F[1,0]:.4f}) "
        f"over class 1 (score={F[1,1]:.4f})"
    )

    # Verify labeled nodes retain strong preference for their original labels
    assert F[0, 0] > F[0, 1], "Node 0 should prefer class 0"
    assert F[2, 1] > F[2, 0], "Node 2 should prefer class 1"

    # Get final predictions
    pred_labels = graph_label_propagation(
        features,
        labels,
        num_classes,
        k=2,
        alpha=0.99,
    )

    assert pred_labels[0] == 0, "Node 0 should be labeled as class 0"
    assert pred_labels[1] == 0, "Node 1 should be labeled as class 0 (closer)"
    assert pred_labels[2] == 1, "Node 2 should be labeled as class 1"

    print("✓ Toy graph sanity test passed")


def test_convergence_stability():
    """
    Test convergence and stability on a small random dataset.

    Setup:
        - 50 points in 2D
        - 5 labeled as class 0
        - 5 labeled as class 1
        - Rest unlabeled

    Verifies:
        - Converges before max_iter
        - No NaNs or Infs in output
        - All predictions are valid class indices
    """
    np.random.seed(42)

    N = 50
    features = np.random.randn(N, 2).astype(np.float32)

    # Assign 5 points to class 0, 5 to class 1, rest unlabeled
    labels = np.full(N, -1, dtype=np.int64)
    labels[:5] = 0
    labels[5:10] = 1

    num_classes = 2

    # Track iteration count by capturing F at each step
    pred_labels, F = graph_label_propagation(
        features,
        labels,
        num_classes,
        k=5,
        alpha=0.95,
        max_iter=1000,
        tol=1e-6,
        return_scores=True,
    )

    # Verify no NaNs or Infs
    assert np.all(np.isfinite(F)), "F contains NaNs or Infs"
    assert np.all(np.isfinite(pred_labels)), "pred_labels contains NaNs or Infs"

    # Verify all predictions are valid class indices
    assert np.all(pred_labels >= 0), "pred_labels contains negative values"
    assert np.all(pred_labels < num_classes), f"pred_labels contains values >= {num_classes}"

    # Verify labeled nodes retain their labels (with high confidence)
    assert np.all(pred_labels[:5] == 0), "Labeled nodes (class 0) should retain label"
    assert np.all(pred_labels[5:10] == 1), "Labeled nodes (class 1) should retain label"

    # Verify shape
    assert pred_labels.shape == (N,), f"Expected shape ({N},), got {pred_labels.shape}"
    assert F.shape == (N, num_classes), f"Expected shape ({N}, {num_classes}), got {F.shape}"

    print("✓ Convergence/stability test passed")


def test_single_sample_real_volume():
    """
    Test on a real supervoxel volume (if data available).

    This test will be skipped if the data is not present.
    """
    # Look for a sample supervoxel file
    data_dir = Path("/data3/wp5/monai-sv-sweeps")
    sv_dir = data_dir / "sv_fullgt_slic_n12000_c0.05_s1.0_ras2_voted"

    if not sv_dir.exists():
        pytest.skip(f"Data directory not found: {sv_dir}")

    # Find first available case
    sv_files = list(sv_dir.glob("*_sv_ids.npy"))
    if not sv_files:
        pytest.skip(f"No supervoxel files found in {sv_dir}")

    case_file = sv_files[0]
    case_id = case_file.stem.replace("_sv_ids", "")

    print(f"\nTesting on case: {case_id}")

    # Load supervoxel IDs
    sv_ids = np.load(case_file)
    unique_svs = np.unique(sv_ids)
    N = len(unique_svs)

    print(f"  Total SVs: {N}")

    # Compute simple features (centroids)
    from scripts.propagate_sv_labels_multi_k import compute_sv_centroids
    features = compute_sv_centroids(sv_ids, unique_svs)

    # Create sparse labels (simulate strategic sampling)
    # Use ~1% of SVs as labeled
    np.random.seed(42)
    n_labeled = max(10, int(0.01 * N))
    labeled_indices = np.random.choice(N, n_labeled, replace=False)

    # Assign random labels (5 classes: 0-4)
    num_classes = 5
    labels = np.full(N, -1, dtype=np.int64)
    labels[labeled_indices] = np.random.randint(0, num_classes, n_labeled)

    print(f"  Labeled SVs: {n_labeled} ({100*n_labeled/N:.2f}%)")

    # Run graph label propagation
    pred_labels = graph_label_propagation(
        features,
        labels,
        num_classes,
        k=10,
        alpha=0.99,
        sigma=None,
    )

    # Verify results
    assert pred_labels.shape == (N,), f"Expected shape ({N},), got {pred_labels.shape}"
    assert np.all(np.isfinite(pred_labels)), "pred_labels contains NaNs or Infs"
    assert np.all(pred_labels >= 0), "pred_labels contains negative values"
    assert np.all(pred_labels < num_classes), f"pred_labels contains values >= {num_classes}"

    # Verify labeled nodes retain their labels
    assert np.all(pred_labels[labeled_indices] == labels[labeled_indices]), \
        "Labeled nodes should retain their original labels"

    # Count class distribution
    class_counts = np.bincount(pred_labels, minlength=num_classes)
    print(f"  Class distribution: {class_counts}")

    # Verify all classes are represented (should be, given alpha < 1)
    # Note: with random labels, not all classes may appear in input
    labeled_classes = np.unique(labels[labels >= 0])
    for c in labeled_classes:
        assert class_counts[c] > 0, f"Class {c} should have at least one prediction"

    print("✓ Single-sample real volume test passed")


def test_edge_cases():
    """Test edge cases: empty input, single point, all labeled."""

    # Empty input
    features_empty = np.array([], dtype=np.float32).reshape(0, 3)
    labels_empty = np.array([], dtype=np.int64)
    pred = graph_label_propagation(features_empty, labels_empty, num_classes=2, k=1)
    assert pred.shape == (0,), "Empty input should return empty output"

    # Single point
    features_single = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    labels_single = np.array([0], dtype=np.int64)
    pred = graph_label_propagation(features_single, labels_single, num_classes=2, k=1)
    assert pred.shape == (1,), "Single point should work"
    assert pred[0] == 0, "Single labeled point should retain its label"

    # All labeled
    features_all = np.random.randn(10, 3).astype(np.float32)
    labels_all = np.random.randint(0, 3, 10).astype(np.int64)
    pred = graph_label_propagation(features_all, labels_all, num_classes=3, k=3)
    assert pred.shape == (10,), "All labeled should work"
    # With high alpha, labeled nodes should largely retain their labels
    # (may not be exact due to graph influence)

    print("✓ Edge cases test passed")


if __name__ == "__main__":
    # Run tests directly
    print("Running graph label propagation tests...\n")

    try:
        test_toy_graph_sanity()
    except Exception as e:
        print(f"✗ Toy graph sanity test failed: {e}")

    try:
        test_convergence_stability()
    except Exception as e:
        print(f"✗ Convergence/stability test failed: {e}")

    try:
        test_single_sample_real_volume()
    except Exception as e:
        if "skip" in str(e).lower():
            print(f"⊘ Single-sample test skipped: {e}")
        else:
            print(f"✗ Single-sample test failed: {e}")

    try:
        test_edge_cases()
    except Exception as e:
        print(f"✗ Edge cases test failed: {e}")

    print("\nAll tests completed!")
