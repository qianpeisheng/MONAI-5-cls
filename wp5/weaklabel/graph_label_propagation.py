#!/usr/bin/env python3
"""
Zhou-style Graph Label Propagation for supervoxels.

Based on: Zhou et al., "Learning with Local and Global Consistency", NIPS 2003.

This module implements graph-based label propagation on supervoxels using the
iterative update formula:
    F^(t+1) = alpha * S * F^(t) + (1 - alpha) * Y
where:
    - S is the normalized similarity matrix
    - F is the label score matrix
    - Y is the initial label matrix
    - alpha is the propagation parameter
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix, diags
from typing import Optional, Tuple


def _build_knn_affinity_matrix(
    features: np.ndarray,
    k: int,
    sigma: Optional[float] = None,
) -> csr_matrix:
    """
    Build kNN affinity matrix W using Gaussian RBF kernel.

    Args:
        features: (N, D) feature array
        k: number of neighbors
        sigma: RBF kernel width; if None, use median distance heuristic

    Returns:
        W: (N, N) sparse symmetric affinity matrix
    """
    N = features.shape[0]

    # Build KD-tree for kNN search
    tree = cKDTree(features)

    # Query k+1 neighbors (includes self), but not more than N
    k_query = min(k + 1, N)
    distances, indices = tree.query(features, k=k_query)

    # Handle edge cases
    if k_query == 1:
        # Only one point - no neighbors
        # Return empty sparse matrix
        return csr_matrix((N, N), dtype=np.float32)

    # Ensure distances and indices are 2D
    if distances.ndim == 1:
        distances = distances.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    # Remove self (first column)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # Estimate sigma from median distance if not provided
    if sigma is None:
        # Use median of non-zero distances
        nonzero_distances = distances[distances > 0]
        if len(nonzero_distances) > 0:
            sigma = float(np.median(nonzero_distances))
        else:
            sigma = 1.0  # fallback for degenerate cases
        if sigma == 0:
            sigma = 1.0  # additional fallback

    # Compute RBF similarities
    # W[i,j] = exp(-dist(i,j)^2 / (2*sigma^2))
    # For zero distances (shouldn't happen after removing self), use similarity 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        similarities = np.exp(-distances**2 / (2 * sigma**2))
        similarities = np.nan_to_num(similarities, nan=1.0, posinf=1.0, neginf=0.0)

    # Build sparse matrix from kNN edges
    # Handle case where k_actual < k (e.g., when N is small)
    k_actual = indices.shape[1] if indices.ndim == 2 else 1
    if indices.ndim == 1:
        # Single neighbor case
        indices = indices.reshape(-1, 1)
        similarities = similarities.reshape(-1, 1)

    row_indices = np.repeat(np.arange(N), k_actual)
    col_indices = indices.ravel()
    data = similarities.ravel()

    W = csr_matrix((data, (row_indices, col_indices)), shape=(N, N))

    # Symmetrize: W = (W + W^T) / 2
    W = (W + W.T) / 2

    # Zero out diagonal (no self-loops)
    W.setdiag(0)
    W.eliminate_zeros()

    return W


def _compute_normalized_similarity(W: csr_matrix) -> csr_matrix:
    """
    Compute normalized similarity matrix S = D^{-1/2} W D^{-1/2}.

    Args:
        W: (N, N) sparse affinity matrix

    Returns:
        S: (N, N) normalized similarity matrix
    """
    # Compute degree vector: d_i = sum_j W[i,j]
    degrees = np.array(W.sum(axis=1)).ravel()

    # Avoid division by zero
    degrees[degrees == 0] = 1.0

    # Compute D^{-1/2}
    d_inv_sqrt = 1.0 / np.sqrt(degrees)

    # S = D^{-1/2} W D^{-1/2}
    # Create diagonal matrix D^{-1/2} as sparse
    D_inv_sqrt = diags(d_inv_sqrt, format='csr')

    # Matrix multiplication: S = D^{-1/2} @ W @ D^{-1/2}
    S = D_inv_sqrt @ W @ D_inv_sqrt

    return S


def _construct_label_matrix(
    sv_labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Construct initial label matrix Y from sparse labels.

    Args:
        sv_labels: (N,) array with -1 for unlabeled, 0..C-1 for classes
        num_classes: number of classes C

    Returns:
        Y: (N, C) label matrix with one-hot encoding for labeled nodes
    """
    N = len(sv_labels)
    Y = np.zeros((N, num_classes), dtype=np.float32)

    # One-hot encode labeled nodes
    labeled_mask = sv_labels >= 0
    if np.any(labeled_mask):
        Y[labeled_mask, sv_labels[labeled_mask]] = 1.0

    return Y


def _iterative_propagation(
    S: csr_matrix,
    Y: np.ndarray,
    alpha: float,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, int, float]:
    """
    Perform iterative label propagation using Zhou's update formula.

    F^(t+1) = alpha * S * F^(t) + (1 - alpha) * Y

    Labeled nodes are clamped to their original labels after each iteration.

    Args:
        S: (N, N) normalized similarity matrix
        Y: (N, C) initial label matrix
        alpha: propagation parameter
        max_iter: maximum iterations
        tol: convergence tolerance (Frobenius norm)

    Returns:
        F: (N, C) final label score matrix
        n_iter: number of iterations performed
        final_diff: final Frobenius norm difference
    """
    F = Y.copy()

    # Identify labeled nodes (rows with at least one non-zero entry)
    labeled_mask = np.any(Y > 0, axis=1)

    for iteration in range(max_iter):
        # F_next = alpha * S @ F + (1 - alpha) * Y
        F_next = alpha * (S @ F) + (1 - alpha) * Y

        # Clamp labeled nodes to their original labels
        # This ensures labeled nodes retain their labels
        F_next[labeled_mask] = Y[labeled_mask]

        # Compute Frobenius norm difference
        diff = np.linalg.norm(F_next - F, ord='fro')

        # Update F
        F = F_next

        # Check convergence
        if diff < tol:
            return F, iteration + 1, float(diff)

    # Max iterations reached
    final_diff = np.linalg.norm(F - (alpha * (S @ F) + (1 - alpha) * Y), ord='fro')
    return F, max_iter, float(final_diff)


def graph_label_propagation(
    sv_features: np.ndarray,
    sv_labels: np.ndarray,
    num_classes: int,
    k: int = 10,
    alpha: float = 0.99,
    max_iter: int = 1000,
    tol: float = 1e-6,
    sigma: Optional[float] = None,
    return_scores: bool = False,
) -> np.ndarray:
    """
    Graph-based label propagation for supervoxels using Zhou's algorithm.

    Args:
        sv_features: (N, D) feature vectors per supervoxel
        sv_labels: (N,) labels with -1 for unlabeled, 0..C-1 for classes
        num_classes: number of classes C (includes background class 0)
        k: number of neighbors for kNN graph
        alpha: propagation parameter (0 < alpha < 1)
        max_iter: maximum iterations
        tol: convergence tolerance (Frobenius norm)
        sigma: RBF kernel width; if None, use median distance heuristic
        return_scores: if True, return (pred_labels, F) instead of just pred_labels

    Returns:
        pred_labels: (N,) predicted labels 0..C-1
        (or tuple of (pred_labels, F) if return_scores=True)
    """
    N = sv_features.shape[0]

    if N == 0:
        result = np.array([], dtype=np.int64)
        return (result, np.array([], dtype=np.float32)) if return_scores else result

    # Validate inputs
    assert sv_labels.shape[0] == N, "Features and labels must have same length"
    assert 0 < alpha < 1, "Alpha must be in (0, 1)"
    assert k > 0, "k must be positive"
    assert num_classes > 0, "num_classes must be positive"

    # Step 1: Build kNN affinity matrix W
    W = _build_knn_affinity_matrix(sv_features, k, sigma)

    # Step 2: Compute normalized similarity matrix S
    S = _compute_normalized_similarity(W)

    # Step 3: Construct initial label matrix Y
    Y = _construct_label_matrix(sv_labels, num_classes)

    # Step 4: Iterative propagation
    F, n_iter, final_diff = _iterative_propagation(S, Y, alpha, max_iter, tol)

    # Step 5: Extract final labels
    pred_labels = np.argmax(F, axis=1).astype(np.int64)

    if return_scores:
        return pred_labels, F
    return pred_labels


def graph_label_propagation_scores(
    sv_features: np.ndarray,
    sv_labels: np.ndarray,
    num_classes: int,
    k: int = 10,
    alpha: float = 0.99,
    max_iter: int = 1000,
    tol: float = 1e-6,
    sigma: Optional[float] = None,
) -> np.ndarray:
    """
    Helper function to get label scores (F matrix) for testing.

    Returns:
        F: (N, C) label score matrix
    """
    _, F = graph_label_propagation(
        sv_features, sv_labels, num_classes, k, alpha, max_iter, tol, sigma,
        return_scores=True
    )
    return F


__all__ = [
    'graph_label_propagation',
    'graph_label_propagation_scores',
]
