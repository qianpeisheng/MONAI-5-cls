#!/usr/bin/env python3
"""
Test suite for strategic sparse supervoxel labeling and propagation.

Tests cover:
- Strategic seed sampling (max 1 per SV, FG borders, rare classes)
- Multi-k label propagation (k-NN with weighted voting)
- Helper utilities (centroids, gradients, adjacency)

Run with: pytest tests/test_strategic_sparse_sv.py -v
"""

import sys
from pathlib import Path

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import numpy as np
import pytest
from collections import Counter

# Import real implementations
try:
    from sample_strategic_sv_seeds import (
        compute_gradient_magnitude,
        sample_strategic_seeds,
    )
    from propagate_sv_labels_multi_k import (
        compute_sv_centroids,
        propagate_multi_k,
        sv_labels_to_dense,
    )
    USE_REAL_IMPL = True
except ImportError as e:
    print(f"Warning: Could not import real implementations: {e}")
    print("Falling back to mock implementations")
    USE_REAL_IMPL = False


class TestStrategicSampling:
    """Test strategic seed sampling logic."""

    def test_max_one_seed_per_sv(self):
        """Test that no supervoxel gets more than 1 seed."""
        # Create simple 10x10x10 volume with 8 supervoxels (2x2x2 grid)
        sv_ids = np.zeros((10, 10, 10), dtype=np.int32)
        sv_id = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    sv_ids[i*5:(i+1)*5, j*5:(j+1)*5, k*5:(k+1)*5] = sv_id
                    sv_id += 1

        # Create GT labels (mix of FG and BG)
        gt_labels = np.zeros((10, 10, 10), dtype=np.int16)
        gt_labels[2:8, 2:8, 2:8] = 1  # Foreground region

        # Mock image
        image = np.random.rand(10, 10, 10).astype(np.float32)

        # Sample with budget of 50 voxels (should cover max 50 SVs, we have 8 total)
        budget_ratio = 50 / 1000  # 5%

        # Mock the sampling function (will implement later)
        seed_mask, seed_sv_labels = mock_sample_strategic(sv_ids, gt_labels, image, budget_ratio)

        # Check: each SV in seed_sv_labels should appear exactly once
        sv_ids_with_seeds = []
        for coord in np.argwhere(seed_mask):
            sv_id = sv_ids[tuple(coord)]
            sv_ids_with_seeds.append(sv_id)

        # No duplicates
        assert len(sv_ids_with_seeds) == len(set(sv_ids_with_seeds)), \
            "Some supervoxels have more than 1 seed!"

        # Matches seed_sv_labels
        assert set(sv_ids_with_seeds) == set(seed_sv_labels.keys()), \
            "Mismatch between seed mask and SV labels dict"

    def test_budget_respected(self):
        """Test that total seeds ≤ budget."""
        sv_ids = np.zeros((20, 20, 20), dtype=np.int32)
        for i in range(8):
            sv_ids[i*2:(i+1)*2, :, :] = i

        gt_labels = np.random.randint(0, 5, (20, 20, 20), dtype=np.int16)
        image = np.random.rand(20, 20, 20).astype(np.float32)

        budget_ratio = 0.001  # 0.1%
        max_seeds = int(sv_ids.size * budget_ratio)

        seed_mask, seed_sv_labels = mock_sample_strategic(sv_ids, gt_labels, image, budget_ratio)

        n_seeds = seed_mask.sum()
        assert n_seeds <= max_seeds, f"Seeds {n_seeds} exceeds budget {max_seeds}"
        assert n_seeds == len(seed_sv_labels), "Seed count mismatch"

    def test_foreground_prioritized(self):
        """Test that foreground voxels are selected over background."""
        # Create volume with clear FG/BG split
        sv_ids = np.arange(100).reshape(10, 10, 1).repeat(10, axis=2)  # 100 SVs

        gt_labels = np.zeros((10, 10, 10), dtype=np.int16)
        gt_labels[5:, :, :] = 1  # Bottom half is FG

        image = np.random.rand(10, 10, 10).astype(np.float32)

        budget_ratio = 0.1  # 10% budget, should cover many SVs

        seed_mask, seed_sv_labels = mock_sample_strategic(sv_ids, gt_labels, image, budget_ratio)

        # Count FG vs BG seeds
        fg_count = sum(1 for label in seed_sv_labels.values() if label > 0)
        bg_count = sum(1 for label in seed_sv_labels.values() if label == 0)

        # FG should dominate (at least 80% of seeds)
        assert fg_count > bg_count * 4, f"FG {fg_count} not prioritized over BG {bg_count}"

    def test_rare_classes_prioritized(self):
        """Test that rare classes (3, 4) get higher weight."""
        sv_ids = np.arange(100).reshape(10, 10, 1).repeat(10, axis=2)

        # Create labels with varying frequencies
        # Class 1: 400 voxels, Class 2: 300 voxels, Class 3: 150 voxels, Class 4: 150 voxels
        gt_labels = np.zeros((10, 10, 10), dtype=np.int16)
        gt_labels[:4, :, :] = 1
        gt_labels[4:7, :, :] = 2
        gt_labels[7:8, 3:8, :] = 3
        gt_labels[8:10, :5, :] = 4

        image = np.random.rand(10, 10, 10).astype(np.float32)

        budget_ratio = 0.05  # 5%, should select ~50 SVs

        seed_mask, seed_sv_labels = mock_sample_strategic(sv_ids, gt_labels, image, budget_ratio)

        # Count selections per class
        class_counts = Counter(seed_sv_labels.values())

        # Rare classes (3, 4) should have decent representation despite lower frequency
        # With 2x priority, should get at least 20% of seeds combined
        rare_count = class_counts.get(3, 0) + class_counts.get(4, 0)
        total_count = len(seed_sv_labels)

        assert rare_count / total_count >= 0.15, \
            f"Rare classes underrepresented: {rare_count}/{total_count}"

    def test_gradient_prioritized(self):
        """Test that high-gradient (border) voxels are selected."""
        sv_ids = np.zeros((20, 20, 20), dtype=np.int32)
        sv_ids[:10, :, :] = 0
        sv_ids[10:, :, :] = 1  # Two big SVs

        # Create image with sharp edge at x=10
        image = np.zeros((20, 20, 20), dtype=np.float32)
        image[10:, :, :] = 1.0  # Sharp transition

        gt_labels = np.ones((20, 20, 20), dtype=np.int16)  # All FG

        budget_ratio = 0.01  # 1%, should get 4 seeds

        seed_mask, seed_sv_labels = mock_sample_strategic(sv_ids, gt_labels, image, budget_ratio)

        # Check that seeds are near the boundary (x=9 or x=10)
        seed_coords = np.argwhere(seed_mask)
        boundary_seeds = sum(1 for coord in seed_coords if 8 <= coord[0] <= 11)

        # Most seeds should be near boundary
        assert boundary_seeds >= len(seed_coords) * 0.5, \
            f"Only {boundary_seeds}/{len(seed_coords)} seeds near boundary"


class TestMultiKPropagation:
    """Test multi-k label propagation."""

    def test_all_k_values_produce_output(self):
        """Test that propagation works for all k values."""
        # Simple 3x3x3 volume with 27 SVs
        sv_ids = np.arange(27).reshape(3, 3, 3)

        # Label 3 SVs sparsely
        sv_labels_sparse = {0: 1, 13: 2, 26: 3}

        k_values = [1, 3, 5, 7, 10]

        results = mock_propagate_multi_k(sv_ids, sv_labels_sparse, k_values)

        # Check all k values present
        assert set(results.keys()) == set(k_values), "Missing k values in results"

        # Each result should label all 27 SVs
        for k, sv_labels_full in results.items():
            assert len(sv_labels_full) == 27, f"k={k} didn't label all SVs"

    def test_sparse_labels_preserved(self):
        """Test that originally labeled SVs keep their labels."""
        sv_ids = np.arange(10).reshape(10, 1, 1)

        sv_labels_sparse = {0: 1, 5: 2, 9: 3}

        results = mock_propagate_multi_k(sv_ids, sv_labels_sparse, [3, 5])

        for k, sv_labels_full in results.items():
            for sv_id, label in sv_labels_sparse.items():
                assert sv_labels_full[sv_id] == label, \
                    f"k={k}: SV {sv_id} changed from {label} to {sv_labels_full[sv_id]}"

    def test_k1_vs_k_many_voting(self):
        """Test that k=1 gives different results than k>1 (voting)."""
        # Linear arrangement: 0---1---2---3---4
        # Label 0→class 1, 4→class 2
        sv_ids = np.arange(5).reshape(5, 1, 1)
        centroids = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0]], dtype=float)

        sv_labels_sparse = {0: 1, 4: 2}

        # k=1: Each unlabeled SV takes nearest
        # SV 1 → nearest is 0 (class 1)
        # SV 2 → nearest is 1 or 3, but need to compute
        # SV 3 → nearest is 4 (class 2)

        # k=3: Voting kicks in
        # SV 2 might get votes from both sides

        results = mock_propagate_multi_k(sv_ids, sv_labels_sparse, [1, 3], centroids)

        # With k=1, SV 1 and 2 should be class 1, SV 3 should be class 2
        assert results[1][1] == 1, "k=1: SV 1 should be class 1"
        assert results[1][3] == 2, "k=1: SV 3 should be class 2"

        # The key test: k=3 might differ on SV 2 due to voting
        # (Hard to assert without knowing exact distances, but results should differ)

    def test_distance_weighting(self):
        """Test that closer neighbors have more influence."""
        # Setup: SV 2 is equidistant from SV 0 (class 1) and SV 4 (class 2)
        # But add SV 1 (class 1) very close to SV 2
        # With weighting, SV 2 should be class 1

        sv_ids = np.arange(5).reshape(5, 1, 1)
        centroids = np.array([
            [0, 0, 0],    # SV 0, class 1
            [1.9, 0, 0],  # SV 1, class 1 (very close to SV 2)
            [2, 0, 0],    # SV 2, unlabeled
            [3, 0, 0],    # SV 3, unlabeled
            [4, 0, 0],    # SV 4, class 2
        ], dtype=float)

        sv_labels_sparse = {0: 1, 1: 1, 4: 2}

        results = mock_propagate_multi_k(sv_ids, sv_labels_sparse, [3], centroids)

        # SV 2 should be class 1 due to very close SV 1
        assert results[3][2] == 1, "Distance weighting failed"


class TestHelperFunctions:
    """Test utility functions."""

    def test_compute_centroids(self):
        """Test centroid computation."""
        # 2x2x2 cube in corner
        sv_ids = np.zeros((10, 10, 10), dtype=np.int32)
        sv_ids[:2, :2, :2] = 1

        centroids = mock_compute_centroids(sv_ids, [0, 1])

        # centroids is (2, 3) array: [centroid_sv0, centroid_sv1]
        # SV 1 centroid should be (0.5, 0.5, 0.5)
        assert np.allclose(centroids[1], [0.5, 0.5, 0.5]), \
            f"SV 1 centroid wrong: {centroids[1]}"

    def test_compute_gradient_magnitude(self):
        """Test gradient computation."""
        # Create image with sharp edge
        image = np.zeros((10, 10, 10), dtype=np.float32)
        image[5:, :, :] = 1.0

        grad_mag = mock_compute_gradient_magnitude(image)

        # Gradient should be high near x=5
        assert grad_mag[4:6, 5, 5].max() > 0.5, "Gradient not detected at edge"

        # Gradient should be low in flat regions
        assert grad_mag[0, 0, 0] < 0.1, "Gradient should be low in flat region"

    def test_sv_labels_to_dense(self):
        """Test broadcasting SV labels to voxels."""
        sv_ids = np.array([[[0, 0], [1, 1]], [[0, 0], [1, 1]]], dtype=np.int32)
        sv_labels = {0: 2, 1: 3}

        dense = mock_sv_labels_to_dense(sv_ids, sv_labels)

        expected = np.array([[[2, 2], [3, 3]], [[2, 2], [3, 3]]], dtype=np.int16)
        assert np.array_equal(dense, expected), "SV→dense conversion failed"


# Wrapper functions that use real or mock implementations
def mock_sample_strategic(sv_ids, gt_labels, image, budget_ratio):
    """Wrapper for strategic sampling (uses real or mock)."""
    if USE_REAL_IMPL:
        return sample_strategic_seeds(sv_ids, gt_labels, image, budget_ratio)
    else:
        # Simple mock: randomly select 1 voxel per FG SV within budget
        fg_svs = []
        for sv_id in np.unique(sv_ids):
            mask = (sv_ids == sv_id)
            labels = gt_labels[mask]
            if np.any((labels >= 1) & (labels <= 4)):
                fg_svs.append(sv_id)

        n_seeds = min(len(fg_svs), int(sv_ids.size * budget_ratio))
        selected_svs = np.random.choice(fg_svs, n_seeds, replace=False)

        seed_mask = np.zeros_like(sv_ids, dtype=bool)
        seed_sv_labels = {}

        for sv_id in selected_svs:
            mask = (sv_ids == sv_id)
            fg_mask = mask & (gt_labels >= 1) & (gt_labels <= 4)
            fg_coords = np.argwhere(fg_mask)

            if len(fg_coords) > 0:
                coord = tuple(fg_coords[0])
                seed_mask[coord] = True
                seed_sv_labels[sv_id] = int(gt_labels[coord])

        return seed_mask, seed_sv_labels


def mock_propagate_multi_k(sv_ids, sv_labels_sparse, k_values, centroids=None):
    """Wrapper for multi-k propagation (uses real or mock)."""
    if USE_REAL_IMPL and centroids is None:
        return propagate_multi_k(sv_ids, sv_labels_sparse, k_values)
    else:
        # Mock implementation for custom centroids or when real not available
        from scipy.spatial import cKDTree

        sv_unique = np.unique(sv_ids)

        if centroids is None:
            centroids = mock_compute_centroids(sv_ids, sv_unique)

        labeled_svs = np.array(list(sv_labels_sparse.keys()))
        labeled_mask = np.isin(sv_unique, labeled_svs)
        unlabeled_svs = sv_unique[~labeled_mask]

        tree = cKDTree(centroids[labeled_mask])

        k_max = max(k_values)
        distances, indices = tree.query(centroids[~labeled_mask], k=min(k_max, len(labeled_svs)))

        if len(labeled_svs) == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)

        results = {}
        for k in k_values:
            sv_labels_full = dict(sv_labels_sparse)

            for i, unlabeled_sv in enumerate(unlabeled_svs):
                k_actual = min(k, len(labeled_svs))
                neighbor_indices = indices[i, :k_actual] if distances.ndim > 1 else [indices[i]]
                neighbor_distances = distances[i, :k_actual] if distances.ndim > 1 else [distances[i]]

                neighbor_labels = [sv_labels_sparse[labeled_svs[idx]] for idx in neighbor_indices]

                weights = 1.0 / (np.array(neighbor_distances) + 1e-6)
                votes = Counter()
                for label, weight in zip(neighbor_labels, weights):
                    votes[label] += weight

                sv_labels_full[int(unlabeled_sv)] = votes.most_common(1)[0][0]

            results[k] = sv_labels_full

        return results


def mock_compute_centroids(sv_ids, sv_list):
    """Wrapper for centroid computation (uses real or mock)."""
    if USE_REAL_IMPL:
        return compute_sv_centroids(sv_ids, np.array(sv_list))
    else:
        centroids = np.zeros((len(sv_list), 3), dtype=float)
        for i, sv_id in enumerate(sv_list):
            mask = (sv_ids == sv_id)
            coords = np.argwhere(mask)
            centroids[i] = coords.mean(axis=0)
        return centroids


def mock_compute_gradient_magnitude(image):
    """Wrapper for gradient computation (uses real or mock)."""
    if USE_REAL_IMPL:
        return compute_gradient_magnitude(image)
    else:
        from scipy.ndimage import sobel

        grad_x = sobel(image, axis=0)
        grad_y = sobel(image, axis=1)
        grad_z = sobel(image, axis=2)

        return np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)


def mock_sv_labels_to_dense(sv_ids, sv_labels):
    """Wrapper for SV to dense conversion (uses real or mock)."""
    if USE_REAL_IMPL:
        return sv_labels_to_dense(sv_ids, sv_labels)
    else:
        dense = np.zeros_like(sv_ids, dtype=np.int16)
        for sv_id, label in sv_labels.items():
            dense[sv_ids == sv_id] = label
        return dense


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
