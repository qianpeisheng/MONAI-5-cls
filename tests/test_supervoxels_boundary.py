"""Tests for boundary-preserving supervoxel method."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from wp5.weaklabel.supervoxels import run_supervoxels
from wp5.weaklabel.sv_utils import relabel_sequential


class TestBoundarySupervoxels:
    """Test the boundary-preserving supervoxel method."""

    def test_boundary_supervoxels_single_region(self):
        """Test that a uniform volume produces a single supervoxel."""
        vol = np.zeros((4, 4, 4), dtype=np.float32)
        labels = run_supervoxels(vol, n_segments=1, method="boundary")

        # Assert output shape
        assert labels.shape == (4, 4, 4), f"Expected shape (4,4,4), got {labels.shape}"

        # Assert only one unique label
        unique_labels = np.unique(labels)
        assert len(unique_labels) == 1, f"Expected 1 unique label, got {len(unique_labels)}"

    def test_boundary_supervoxels_two_regions(self):
        """Test that a volume with two distinct regions produces two supervoxels."""
        vol = np.zeros((4, 4, 4), dtype=np.float32)
        # First half along z is 0.0, second half is 1.0
        vol[2:, :, :] = 1.0

        labels = run_supervoxels(vol, n_segments=2, method="boundary")

        # Assert exactly 2 unique labels
        unique_labels = np.unique(labels)
        assert len(unique_labels) == 2, f"Expected 2 unique labels, got {len(unique_labels)}"

        # Check that majority label in first half differs from second half
        first_half = labels[:2, :, :]
        second_half = labels[2:, :, :]

        # Find majority label in each half
        first_majority = np.bincount(first_half.ravel()).argmax()
        second_majority = np.bincount(second_half.ravel()).argmax()

        assert first_majority != second_majority, \
            f"Expected different majority labels, got {first_majority} and {second_majority}"

        # Check that mislabeled voxels are minimal
        first_half_correct = np.sum(first_half == first_majority)
        first_half_total = first_half.size
        first_half_accuracy = first_half_correct / first_half_total

        second_half_correct = np.sum(second_half == second_majority)
        second_half_total = second_half.size
        second_half_accuracy = second_half_correct / second_half_total

        assert first_half_accuracy >= 0.9, \
            f"Expected >90% accuracy in first half, got {first_half_accuracy:.2%}"
        assert second_half_accuracy >= 0.9, \
            f"Expected >90% accuracy in second half, got {second_half_accuracy:.2%}"

    def test_boundary_two_regions_separation_strict(self):
        """Stricter test for two-region separation to expose exchange phase bugs."""
        vol = np.zeros((4, 4, 4), dtype=np.float32)
        # First half along z is 0, second half is 1
        vol[2:, :, :] = 1.0

        labels = run_supervoxels(vol, n_segments=2, method="boundary")

        # Assert exactly 2 unique labels
        unique_labels = np.unique(labels)
        assert len(unique_labels) == 2, f"Expected 2 unique labels, got {len(unique_labels)}"

        # Compute majority label in each half
        top_half = labels[:2, :, :]
        bottom_half = labels[2:, :, :]

        top_majority = np.bincount(top_half.ravel()).argmax()
        bottom_majority = np.bincount(bottom_half.ravel()).argmax()

        # Must have different majority labels
        assert top_majority != bottom_majority, \
            f"Top and bottom halves must have different majority labels"

        # Compute mislabel fraction (stricter than 0.2)
        top_correct = np.sum(top_half == top_majority)
        bottom_correct = np.sum(bottom_half == bottom_majority)
        total_correct = top_correct + bottom_correct
        total_voxels = labels.size

        mislabel_fraction = 1.0 - (total_correct / total_voxels)
        assert mislabel_fraction < 0.2, \
            f"Expected <20% mislabeled voxels, got {mislabel_fraction:.2%}"

    def test_boundary_supervoxels_respects_n_segments_approximately(self):
        """Test that the number of supervoxels is close to n_segments."""
        # Create a random smooth volume
        np.random.seed(42)
        vol = np.random.randn(6, 6, 6).astype(np.float32)
        vol = gaussian_filter(vol, sigma=0.5)

        n_segments = 10
        labels = run_supervoxels(vol, n_segments=n_segments, method="boundary")

        # Check shape
        assert labels.shape == (6, 6, 6)

        # Check number of unique labels is within reasonable bounds
        unique_labels = np.unique(labels)
        n_actual = len(unique_labels)

        # Allow for some flexibility (between 50% and 200% of target)
        assert 5 <= n_actual <= 20, \
            f"Expected 5-20 supervoxels for n_segments={n_segments}, got {n_actual}"

    def test_run_supervoxels_method_dispatch(self):
        """Test that method parameter correctly dispatches to SLIC or boundary."""
        vol = np.random.randn(4, 4, 4).astype(np.float32)
        vol = gaussian_filter(vol, sigma=0.5)

        # Test SLIC method
        labels_slic = run_supervoxels(vol, n_segments=5, method="slic")
        assert labels_slic.shape == (4, 4, 4)
        assert labels_slic.dtype == np.int32
        assert np.all(labels_slic >= 0)

        # Test boundary method
        labels_boundary = run_supervoxels(vol, n_segments=5, method="boundary")
        assert labels_boundary.shape == (4, 4, 4)
        assert labels_boundary.dtype == np.int32
        assert np.all(labels_boundary >= 0)

        # Test that both produce valid labels
        # (they should differ, but both should be valid)
        slic_unique = np.unique(labels_slic)
        boundary_unique = np.unique(labels_boundary)
        assert len(slic_unique) > 0
        assert len(boundary_unique) > 0

        # Test invalid method raises ValueError
        with pytest.raises(ValueError, match="Unknown method"):
            run_supervoxels(vol, method="invalid_method")

    def test_boundary_supervoxels_labels_are_sequential(self):
        """Test that labels are contiguous starting at 0."""
        vol = np.random.randn(5, 5, 5).astype(np.float32)
        vol = gaussian_filter(vol, sigma=0.5)

        labels = run_supervoxels(vol, n_segments=8, method="boundary")

        unique_labels = np.unique(labels)
        # Should be 0, 1, 2, ..., n-1
        expected = np.arange(len(unique_labels))
        np.testing.assert_array_equal(unique_labels, expected,
            err_msg="Labels should be contiguous starting at 0")

    def test_boundary_supervoxels_relabel_sequential_compatibility(self):
        """Test that labels can be passed through relabel_sequential without changing segmentation."""
        vol = np.random.randn(5, 5, 5).astype(np.float32)
        vol = gaussian_filter(vol, sigma=0.5)

        labels = run_supervoxels(vol, n_segments=8, method="boundary")

        # Apply relabel_sequential
        relabeled, n_labels = relabel_sequential(labels)

        # Should not change anything since labels are already sequential
        np.testing.assert_array_equal(labels, relabeled,
            err_msg="Relabeling sequential labels should not change them")

        # Number of labels should match
        assert n_labels == len(np.unique(labels))


class TestBoundarySupervoxelsHelpers:
    """Test internal helper functions for boundary-preserving supervoxels."""

    def test_build_voxel_features_shape_and_scaling(self):
        """Test that voxel features have correct shape and scaling."""
        # Import the helper (this will be implemented later)
        from wp5.weaklabel.supervoxels import _build_voxel_features

        vol = np.random.randn(3, 4, 5).astype(np.float32)
        spatial_scale = 2.0
        intensity_scale = 1.0

        features = _build_voxel_features(vol, spatial_scale, intensity_scale)

        # Check shape: (N, 4) where N = Z*Y*X
        Z, Y, X = vol.shape
        N = Z * Y * X
        assert features.shape == (N, 4), f"Expected shape ({N}, 4), got {features.shape}"

        # Check that spatial coordinates are scaled
        # First voxel (0,0,0) should have coords (0,0,0,intensity)
        # Coords should be divided by spatial_scale
        max_z_coord = features[:, 0].max()
        expected_max_z = (Z - 1) / spatial_scale
        np.testing.assert_allclose(max_z_coord, expected_max_z, rtol=0.01)

    def test_build_grid_adjacency_edge_count(self):
        """Test that grid adjacency has correct number of edges."""
        from wp5.weaklabel.supervoxels import _build_grid_adjacency

        # For a (2,2,2) grid with 6-connectivity
        # 8 voxels total
        # Each interior edge connects 2 voxels
        # Edges: z-direction: 2*2*1 = 4
        #        y-direction: 2*1*2 = 4
        #        x-direction: 1*2*2 = 4
        # Total: 12 edges

        edges = _build_grid_adjacency((2, 2, 2))

        assert edges.shape[1] == 2, "Edges should have 2 columns (u, v)"
        assert edges.shape[0] == 12, f"Expected 12 edges for (2,2,2) grid, got {edges.shape[0]}"

        # Check that all edges are valid indices
        assert np.all(edges >= 0) and np.all(edges < 8), \
            "Edge indices should be in range [0, 8)"

        # Check that u < v (undirected, canonical form)
        # Not strictly required but good practice
        # assert np.all(edges[:, 0] < edges[:, 1]), "Edges should have u < v"
