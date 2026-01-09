from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix


Metric = Literal["l2", "cosine", "chi2"]


def _median_sigma(distances: np.ndarray) -> float:
    """Median heuristic sigma from a distance array."""
    d = distances[np.isfinite(distances)]
    d = d[d > 0]
    if d.size == 0:
        return 1.0
    sigma = float(np.median(d))
    return 1.0 if sigma == 0.0 else sigma


def _cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    dot = np.sum(a * b, axis=1)
    na = np.sqrt(np.sum(a * a, axis=1))
    nb = np.sqrt(np.sum(b * b, axis=1))
    denom = (na * nb) + eps
    cos = dot / denom
    # Numeric safety: keep in [-1,1]
    cos = np.clip(cos, -1.0, 1.0)
    return 1.0 - cos


def _chi2_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # Chi-square distance: sqrt(0.5 * sum((a-b)^2 / (a+b+eps)))
    num = (a - b) ** 2
    den = a + b + eps
    chi2 = 0.5 * np.sum(num / den, axis=1)
    chi2 = np.maximum(chi2, 0.0)
    return np.sqrt(chi2)


def build_spatial_intensity_affinity(
    *,
    centroids: np.ndarray,
    phi: np.ndarray,
    k: int,
    sigma_phi: Union[float, str] = "median",
    use_cosine: bool = False,
    metric: Metric = "l2",
    sigma_c: Optional[float] = None,
    sample_edges_for_sigma: int = 50_000,
    seed: int = 42,
    chi2_eps: float = 1e-8,
) -> csr_matrix:
    """Build kNN affinity matrix with a spatial term and an intensity descriptor term.

    w_ij = exp( -||c_i - c_j||^2 / (2*sigma_c^2) ) * exp( -d(phi_i,phi_j)^2 / (2*sigma_phi^2) )
    """
    if centroids.ndim != 2 or centroids.shape[1] != 3:
        raise ValueError(f"Expected centroids shape (N,3), got {centroids.shape}")
    if phi.ndim != 2 or phi.shape[0] != centroids.shape[0]:
        raise ValueError(f"Expected phi shape (N,D) with N={centroids.shape[0]}, got {phi.shape}")
    if k <= 0:
        raise ValueError("k must be > 0")

    N = int(centroids.shape[0])
    if N == 0:
        return csr_matrix((0, 0), dtype=np.float32)

    tree = cKDTree(centroids.astype(np.float32, copy=False))
    k_query = min(int(k) + 1, N)
    distances_c, indices = tree.query(centroids, k=k_query)

    if k_query == 1:
        return csr_matrix((N, N), dtype=np.float32)

    if distances_c.ndim == 1:
        distances_c = distances_c.reshape(-1, 1)
        indices = indices.reshape(-1, 1)

    # Remove self-neighbor
    distances_c = distances_c[:, 1:]
    indices = indices[:, 1:]
    k_actual = int(indices.shape[1])

    row = np.repeat(np.arange(N, dtype=np.int64), k_actual)
    col = indices.reshape(-1).astype(np.int64, copy=False)
    d_c = distances_c.reshape(-1).astype(np.float32, copy=False)

    if sigma_c is None:
        sigma_c = _median_sigma(d_c)
    else:
        sigma_c = float(sigma_c)
        if sigma_c <= 0:
            raise ValueError("sigma_c must be > 0")

    w_c = np.exp(-(d_c ** 2) / (2.0 * float(sigma_c) ** 2)).astype(np.float32)

    # Descriptor distances
    a = phi[row]
    b = phi[col]
    if metric == "chi2":
        d_phi = _chi2_distance(a, b, eps=float(chi2_eps)).astype(np.float32)
    else:
        if use_cosine or metric == "cosine":
            d_phi = _cosine_distance(a, b).astype(np.float32)
        else:
            d_phi = np.linalg.norm(a - b, axis=1).astype(np.float32)

    # sigma_phi: number or median heuristic over sampled edges
    if isinstance(sigma_phi, str):
        if sigma_phi != "median":
            raise ValueError("sigma_phi must be a float or 'median'")
        if d_phi.size == 0:
            sigma_phi_val = 1.0
        else:
            rng = np.random.RandomState(int(seed))
            n_sample = min(int(sample_edges_for_sigma), int(d_phi.size))
            if n_sample < d_phi.size:
                sel = rng.choice(d_phi.size, size=n_sample, replace=False)
                d_sample = d_phi[sel]
            else:
                d_sample = d_phi
            sigma_phi_val = _median_sigma(d_sample)
        sigma_phi = sigma_phi_val
    else:
        sigma_phi = float(sigma_phi)
        if sigma_phi <= 0:
            raise ValueError("sigma_phi must be > 0")

    w_phi = np.exp(-(d_phi ** 2) / (2.0 * float(sigma_phi) ** 2)).astype(np.float32)
    w = (w_c * w_phi).astype(np.float32)

    W = csr_matrix((w, (row, col)), shape=(N, N), dtype=np.float32)
    W = (W + W.T) / 2
    W.setdiag(0)
    W.eliminate_zeros()
    return W


__all__ = ["build_spatial_intensity_affinity"]

