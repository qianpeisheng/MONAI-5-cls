from typing import Dict, Optional, Set, Tuple

import numpy as np

try:
    from skimage.segmentation import slic  # type: ignore
    from skimage.transform import resize  # type: ignore
except Exception as e:  # pragma: no cover - required dependency
    slic = None  # type: ignore
    resize = None  # type: ignore
    _import_err = e

from .sv_utils import relabel_sequential


def _resize_vol(vol: np.ndarray, out_shape: Tuple[int, int, int], order: int) -> np.ndarray:
    # skimage.transform.resize expects (Z,Y,X) with preserve_range=True
    return resize(vol, out_shape, order=order, mode="edge", anti_aliasing=False, preserve_range=True).astype(vol.dtype)


def _run_supervoxels_slic(
    image_vol: np.ndarray,
    *,
    n_segments: int = 10000,
    compactness: float = 0.05,
    sigma: float = 0.5,
    downsample: int = 1,
    enforce_connectivity: bool = True,
) -> np.ndarray:
    """Compute 3D supervoxels using SLIC.

    Args:
      image_vol: float32 array (Z,Y,X), normalized upstream.
      n_segments: approximate number of supervoxels.
      compactness: balance spatial vs intensity proximity.
      sigma: Gaussian smoothing prior to segmentation.
      downsample: integer factor >=1. If >1, compute SLIC at lower resolution
                  and upsample labels back via nearest (order=0).
      enforce_connectivity: ensure each segment is spatially contiguous.

    Returns:
      sv_ids: int32 array (Z,Y,X) with contiguous ids starting at 0.
    """
    if slic is None or resize is None:
        raise RuntimeError(
            f"scikit-image is required for SLIC supervoxels but could not be imported: {_import_err}"
        )

    if image_vol.ndim != 3:
        raise ValueError(f"Expected 3D volume (Z,Y,X), got shape {image_vol.shape}")

    img = image_vol
    orig_shape = img.shape
    if downsample and downsample > 1:
        ds_shape = tuple(int(np.ceil(s / downsample)) for s in orig_shape)
        img_ds = _resize_vol(img, ds_shape, order=1).astype(np.float32)
    else:
        img_ds = img

    labels_ds = slic(
        img_ds,
        n_segments=int(n_segments),
        compactness=float(compactness),
        sigma=float(sigma),
        start_label=0,
        channel_axis=None,
        enforce_connectivity=bool(enforce_connectivity),
    ).astype(np.int64)

    if img_ds.shape != orig_shape:
        # Upsample via nearest
        labels = _resize_vol(labels_ds, orig_shape, order=0).astype(np.int64)
    else:
        labels = labels_ds

    labels, _ = relabel_sequential(labels)
    return labels.astype(np.int32, copy=False)


# ===== Boundary-preserving supervoxel helpers =====


def _sqeuclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Squared Euclidean distance between two feature vectors."""
    diff = a - b
    return float(diff @ diff)


def _build_voxel_features(
    image_vol: np.ndarray,
    spatial_scale: float,
    intensity_scale: float = 1.0,
) -> np.ndarray:
    """Build voxel features combining spatial coordinates and intensity.

    Args:
        image_vol: float32 array (Z, Y, X)
        spatial_scale: R in voxels; roughly desired supervoxel radius.
        intensity_scale: factor to balance intensity vs spatial terms.

    Returns:
        features: (N, D) where N = Z*Y*X, D = 4 (z, y, x, intensity)
    """
    Z, Y, X = image_vol.shape
    # Grid indices
    zz, yy, xx = np.meshgrid(
        np.arange(Z), np.arange(Y), np.arange(X), indexing="ij"
    )
    # Normalize spatial coords by spatial_scale
    coords = np.stack(
        [zz / spatial_scale, yy / spatial_scale, xx / spatial_scale],
        axis=-1,
    ).astype(np.float32)

    # Normalize intensity
    I = (image_vol.astype(np.float32) * intensity_scale)[..., None]

    feats = np.concatenate([coords, I], axis=-1)
    return feats.reshape(-1, feats.shape[-1])  # (N, 4)


def _build_grid_adjacency(shape: Tuple[int, int, int]) -> np.ndarray:
    """Build voxel adjacency as a list of undirected edges (u, v) using 6-neighborhood.

    Vectorized implementation for better performance on large volumes.

    Args:
        shape: (Z, Y, X) grid dimensions

    Returns:
        edges: (E, 2) int64 array of voxel indices in [0, N).
    """
    Z, Y, X = shape
    # Create index array mapping (z, y, x) -> linear index
    idx = np.arange(Z * Y * X, dtype=np.int64).reshape(Z, Y, X)

    edges = []

    # +x neighbors
    if X > 1:
        edges_x = np.stack(
            [idx[:, :, :-1].ravel(), idx[:, :, 1:].ravel()],
            axis=1
        )
        edges.append(edges_x)

    # +y neighbors
    if Y > 1:
        edges_y = np.stack(
            [idx[:, :-1, :].ravel(), idx[:, 1:, :].ravel()],
            axis=1
        )
        edges.append(edges_y)

    # +z neighbors
    if Z > 1:
        edges_z = np.stack(
            [idx[:-1, :, :].ravel(), idx[1:, :, :].ravel()],
            axis=1
        )
        edges.append(edges_z)

    if edges:
        edges_all = np.vstack(edges)
    else:
        # Single voxel case
        edges_all = np.empty((0, 2), dtype=np.int64)

    return edges_all


def _estimate_initial_lambda(features: np.ndarray, edges: np.ndarray) -> float:
    """Estimate initial lambda using median of minimal neighbor distances.

    Args:
        features: (N, D) feature array
        edges: (E, 2) edge array

    Returns:
        lambda_0: initial regularization parameter
    """
    N = features.shape[0]
    # For each voxel, track min distance to a neighbor
    min_d = np.full(N, np.inf, dtype=np.float32)
    for u, v in edges:
        d = _sqeuclidean(features[u], features[v])
        if d < min_d[u]:
            min_d[u] = d
        if d < min_d[v]:
            min_d[v] = d
    # Replace inf (isolated) with 0
    min_d[np.isinf(min_d)] = 0.0
    # Use median, but start conservatively small to encourage initial merging
    median_val = float(np.median(min_d))
    # Start with 10% of median to allow aggressive initial merging
    return max(median_val * 0.1, 1e-6)


def _init_clusters(
    features: np.ndarray, edges: np.ndarray
) -> Tuple[np.ndarray, Dict[int, dict], Dict[int, Set[int]]]:
    """Initialize clusters: each voxel is its own cluster with aggregates.

    Args:
        features: (N, D) feature array
        edges: (E, 2) edge array

    Returns:
        cluster_id_per_voxel: (N,) int32 array
        clusters: dict mapping cluster_id -> {
            "size": int,
            "sum_x": ndarray (D,),
            "sum_x2": float,
            "rep_feat": ndarray (D,),
            "voxels": array (for now, will be removed later)
        }
        cluster_neighbors: dict mapping cluster_id -> set of neighbor cluster_ids
    """
    N, D = features.shape
    cluster_id_per_voxel = np.arange(N, dtype=np.int32)

    clusters = {}
    for vid in range(N):
        fv = features[vid]
        clusters[vid] = {
            "size": 1,
            "sum_x": fv.copy(),
            "sum_x2": float(fv @ fv),
            "rep_feat": fv.copy(),
            "voxels": np.array([vid], dtype=np.int64),  # Will remove later
            "rep": vid,  # Keep for exchange phase compatibility
        }

    cluster_neighbors: Dict[int, Set[int]] = {cid: set() for cid in clusters.keys()}
    for u, v in edges:
        cu = int(cluster_id_per_voxel[u])
        cv = int(cluster_id_per_voxel[v])
        if cu != cv:
            cluster_neighbors[cu].add(cv)
            cluster_neighbors[cv].add(cu)

    return cluster_id_per_voxel, clusters, cluster_neighbors


def _merge_delta_agg(
    clusters: Dict[int, dict],
    i: int,
    j: int,
    lambda_val: float,
) -> float:
    """Compute energy difference if we merge cluster j into i using aggregates.

    This is O(D) instead of O(|cluster_j|), where D is the feature dimension.

    Args:
        clusters: cluster dict with aggregates (size, sum_x, sum_x2, rep_feat)
        i: keep cluster id
        j: kill cluster id
        lambda_val: regularization parameter

    Returns:
        delta_E: E_after - E_before (negative means good: reduces energy)
    """
    cl_i = clusters[i]
    cl_j = clusters[j]

    rep_i = cl_i["rep_feat"]  # shape (D,)
    rep_j = cl_j["rep_feat"]  # shape (D,)

    n_j = cl_j["size"]
    sum_x_j = cl_j["sum_x"]   # shape (D,)
    sum_x2_j = cl_j["sum_x2"] # scalar

    def sse_to(rep: np.ndarray) -> float:
        """Compute Σ ||f_v - rep||² over v ∈ S_j using aggregates."""
        # Σ ||f_v - rep||² = Σ ||f_v||² - 2*rep·Σf_v + n*||rep||²
        rep_dot = float(rep @ rep)
        return float(
            sum_x2_j
            - 2.0 * float(rep @ sum_x_j)
            + n_j * rep_dot
        )

    cost_before = sse_to(rep_j) + lambda_val  # cluster j contributes 1 to |C|
    cost_after = sse_to(rep_i)  # cluster count decreases by 1

    return cost_after - cost_before


def _fusion_phase(
    features: np.ndarray,
    edges: np.ndarray,
    target_K: int,
    max_passes: int = 30,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[int, dict], Dict[int, Set[int]]]:
    """Fusion phase: greedy bottom-up merging to reach target number of clusters.

    Args:
        features: (N, D) feature array
        edges: (E, 2) edge array
        target_K: target number of clusters
        max_passes: maximum number of passes
        verbose: if True, print progress

    Returns:
        cluster_id_per_voxel: (N,) int32 array
        clusters: cluster dict
        cluster_neighbors: cluster neighbor dict
    """
    N = features.shape[0]
    cid_per_voxel, clusters, neighbors = _init_clusters(features, edges)
    lambda_val = _estimate_initial_lambda(features, edges)

    cur_K = len(clusters)
    pass_idx = 0

    if verbose:
        print(f"    Pass 0: {cur_K:,} clusters (lambda={lambda_val:.2e})")

    while cur_K > target_K and pass_idx < max_passes:
        pass_idx += 1
        merged_any = False

        # Collect merges to apply in this pass
        to_merge = []  # list of (keep_cid, kill_cid, delta)

        # Adaptive threshold based on how far we are from target
        ratio = cur_K / target_K
        if ratio > 3.0:
            # Very far from target: accept merges with delta < 2*lambda
            threshold = 2.0 * lambda_val
        elif ratio > 1.5:
            # Moderately far: accept merges with delta < lambda
            threshold = lambda_val
        elif ratio > 1.2:
            # Approaching target: accept merges with delta < 0.5*lambda
            threshold = 0.5 * lambda_val
        elif ratio > 1.05:
            # Close to target: accept merges with delta < 0.2*lambda
            threshold = 0.2 * lambda_val
        else:
            # Very close or at target: only accept energy-reducing merges
            threshold = 0.0

        for i in list(clusters.keys()):
            if i not in clusters:
                continue

            for j in list(neighbors[i]):
                if j not in clusters or j == i:
                    continue
                delta = _merge_delta_agg(clusters, i, j, lambda_val)
                if delta < threshold:
                    # Good merge
                    to_merge.append((i, j, delta))

        # Sort by delta (best merges first)
        to_merge.sort(key=lambda x: x[2])

        # If no beneficial merges found but we're still above target, force merges
        if not to_merge and cur_K > target_K:
            # Find all possible merges and pick the best ones
            all_merges = []
            for i in list(clusters.keys()):
                if i not in clusters:
                    continue
                for j in list(neighbors[i]):
                    if j not in clusters or j == i:
                        continue
                    if i < j:  # Avoid duplicates
                        delta = _merge_delta_agg(clusters, i, j, lambda_val)
                        all_merges.append((i, j, delta))
            # Sort and take best merges
            all_merges.sort(key=lambda x: x[2])
            # Merge enough to reach target - be precise near target
            num_excess = cur_K - target_K
            if num_excess <= 5:
                # Very close to target: merge exactly to reach target
                num_to_merge = min(len(all_merges), num_excess)
            elif num_excess <= 20:
                # Close: merge 70% of excess to avoid overshooting
                num_to_merge = min(len(all_merges), max(1, int(num_excess * 0.7)))
            elif num_excess <= 100:
                # Moderately close: merge 60% of excess
                num_to_merge = min(len(all_merges), max(1, int(num_excess * 0.6)))
            else:
                # Still far: merge half the excess
                num_to_merge = min(len(all_merges), max(1, num_excess // 2))
            to_merge = all_merges[:num_to_merge]

        # Apply merges
        for i, j, delta in to_merge:
            if j not in clusters or i not in clusters:
                continue

            # Merge j into i
            cl_i = clusters[i]
            cl_j = clusters[j]

            # Update aggregates
            cl_i["size"] += cl_j["size"]
            cl_i["sum_x"] += cl_j["sum_x"]
            cl_i["sum_x2"] += cl_j["sum_x2"]
            # Update representative to centroid
            cl_i["rep_feat"] = cl_i["sum_x"] / cl_i["size"]

            # Update voxel lists (will be removed later for optimization)
            vox_j = cl_j["voxels"]
            cl_i["voxels"] = np.concatenate([cl_i["voxels"], vox_j])

            # Update cid_per_voxel
            cid_per_voxel[vox_j] = i

            # Update neighbors: Ni ∪ Nj \ {i,j}
            neighbors[i].update(neighbors[j])
            neighbors[i].discard(i)
            neighbors[i].discard(j)

            # Remove j from neighbors of others
            for k in list(neighbors[j]):
                if k in neighbors:
                    neighbors[k].discard(j)
                    neighbors[k].add(i)

            # Remove cluster j
            del clusters[j]
            del neighbors[j]
            merged_any = True

        cur_K = len(clusters)
        lambda_val *= 1.5  # make merges gradually harder

        if verbose:
            print(f"    Pass {pass_idx}: {cur_K:,} clusters (merged {len(to_merge):,}, lambda={lambda_val:.2e})")

        if not merged_any:
            break

    return cid_per_voxel, clusters, neighbors


def _exchange_phase(
    features: np.ndarray,
    cid_per_voxel: np.ndarray,
    clusters: Dict[int, dict],
    neighbors: Dict[int, Set[int]],
    max_iters: int = 3,
    verbose: bool = False,
) -> np.ndarray:
    """Exchange phase: refine boundaries via local voxel reassignments.

    Runs up to max_iters full sweeps over all voxels, reassigning each voxel
    to a neighboring cluster if that reduces distance to the cluster representative.

    Args:
        features: (N, D) feature array
        cid_per_voxel: (N,) int32 array
        clusters: cluster dict
        neighbors: cluster neighbor dict
        max_iters: maximum number of full sweeps (default 3)
        verbose: if True, print progress

    Returns:
        cid_per_voxel: updated (N,) int32 array
    """
    # Use precomputed representative features (centroids) from clusters
    rep_feat = {cid: info["rep_feat"] for cid, info in clusters.items()}

    N = features.shape[0]

    # Run max_iters full sweeps over all voxels
    for sweep in range(max_iters):
        moved = 0

        for v in range(N):
            cid = int(cid_per_voxel[v])
            f_v = features[v]
            best_cid = cid
            best_dist = _sqeuclidean(f_v, rep_feat[cid])

            # Candidate clusters: neighbors of current cluster
            if cid in neighbors:
                for nb_cid in neighbors[cid]:
                    if nb_cid in rep_feat:
                        d = _sqeuclidean(f_v, rep_feat[nb_cid])
                        if d < best_dist:
                            best_dist = d
                            best_cid = nb_cid

            if best_cid != cid:
                # Reassign voxel
                cid_per_voxel[v] = best_cid
                moved += 1

        if verbose:
            print(f"    Sweep {sweep+1}/{max_iters}: {moved:,} voxels reassigned")

        # Early stopping if no voxels moved
        if moved == 0:
            if verbose:
                print(f"    Converged at sweep {sweep+1}")
            break

    return cid_per_voxel


def _run_supervoxels_boundary(
    image_vol: np.ndarray,
    *,
    n_segments: int = 10000,
    **kwargs,
) -> np.ndarray:
    """Compute boundary-preserving supervoxels for 3D volume.

    Args:
        image_vol: float32 array (Z,Y,X).
        n_segments: approximate number of supervoxels desired.

    Returns:
        labels: (Z,Y,X) int32, contiguous ids starting at 0.
    """
    if image_vol.ndim != 3:
        raise ValueError(f"Expected 3D volume (Z,Y,X), got shape {image_vol.shape}")

    Z, Y, X = image_vol.shape
    N = Z * Y * X

    # Size guard: boundary method works but may be slow on very large volumes
    max_boundary_N = 1_500_000  # ~114x114x114, supports volumes up to ~1.5M voxels
    if N > max_boundary_N:
        raise ValueError(
            f"Boundary method not recommended for N={N} voxels (shape {image_vol.shape}). "
            f"Maximum supported: {max_boundary_N} voxels. "
            f"Please use method='slic', downsample the volume, or reduce ROI."
        )

    # Infer spatial_scale from volume size and desired n_segments
    avg_voxels_per_sv = max(N / float(n_segments), 1.0)
    spatial_scale = float(avg_voxels_per_sv ** (1.0 / 3.0))

    # 1) Build features
    print(f"  Building features for {N:,} voxels...")
    feats = _build_voxel_features(
        image_vol=image_vol,
        spatial_scale=spatial_scale,
        intensity_scale=1.0,
    )

    # 2) Build voxel adjacency
    print(f"  Building grid adjacency (6-connected)...")
    edges = _build_grid_adjacency((Z, Y, X))
    print(f"  Created {edges.shape[0]:,} edges")

    # 3) Fusion phase
    print(f"  Running fusion phase (target: {n_segments:,} segments)...")
    target_K = int(n_segments)
    cid_per_voxel, clusters, neighbors = _fusion_phase(feats, edges, target_K, verbose=True)
    print(f"  Fusion complete: {len(clusters):,} clusters")

    # 4) Exchange phase
    print(f"  Running exchange phase (boundary refinement)...")
    cid_per_voxel = _exchange_phase(feats, cid_per_voxel, clusters, neighbors, verbose=True)

    # 5) Reshape and relabel sequentially
    print(f"  Finalizing labels...")
    labels = cid_per_voxel.reshape(Z, Y, X)
    labels, _ = relabel_sequential(labels)
    return labels.astype(np.int32, copy=False)


def run_supervoxels(
    image_vol: np.ndarray,
    *,
    n_segments: int = 10000,
    compactness: float = 0.05,
    sigma: float = 0.5,
    downsample: int = 1,
    enforce_connectivity: bool = True,
    method: str = "slic",
) -> np.ndarray:
    """Compute 3D supervoxels using SLIC or boundary-preserving method.

    Args:
      image_vol: float32 array (Z,Y,X), normalized upstream.
      n_segments: approximate number of supervoxels.
      compactness: balance spatial vs intensity proximity (SLIC only).
      sigma: Gaussian smoothing prior to segmentation (SLIC only).
      downsample: integer factor >=1 (SLIC only).
      enforce_connectivity: ensure spatial contiguity (SLIC only).
      method: "slic" or "boundary" for boundary-preserving.

    Returns:
      sv_ids: int32 array (Z,Y,X) with contiguous ids starting at 0.
    """
    if method == "slic":
        return _run_supervoxels_slic(
            image_vol,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            downsample=downsample,
            enforce_connectivity=enforce_connectivity,
        )
    elif method == "boundary":
        return _run_supervoxels_boundary(image_vol, n_segments=n_segments)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'slic' or 'boundary'.")


__all__ = ["run_supervoxels"]
