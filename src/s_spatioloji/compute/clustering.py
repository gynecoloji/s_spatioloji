"""Clustering functions.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.

Optimized for large datasets (10M+ cells): KNN graph building uses
``scipy.spatial.cKDTree`` (O(N log N)) with vectorized edge construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from s_spatioloji.compute import _atomic_write_parquet, _load_dense

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def _build_knn_edges(
    matrix: np.ndarray,
    k: int,
    n_jobs: int = -1,
) -> np.ndarray:
    """Build KNN edge list using the fastest available method.

    Tries ``pynndescent`` (approximate, fastest for high-dim), falls back
    to ``scipy.spatial.cKDTree`` (exact, fast for low-dim PCA).

    Args:
        matrix: (n_cells, n_features) array.
        k: Number of nearest neighbors.
        n_jobs: Parallel workers (for pynndescent). -1 = all cores.

    Returns:
        Array of shape (n_edges, 2) with (source, target) pairs, deduplicated.
    """
    n_cells = matrix.shape[0]
    k = min(k, n_cells - 1)

    try:
        from pynndescent import NNDescent

        index = NNDescent(matrix, n_neighbors=k + 1, n_jobs=n_jobs, random_state=42)
        indices, _ = index.neighbor_graph
        indices = indices[:, 1:]  # drop self
    except ImportError:
        from scipy.spatial import cKDTree

        tree = cKDTree(matrix)
        _, indices = tree.query(matrix, k=k + 1, workers=n_jobs)
        indices = indices[:, 1:]  # drop self

    # Vectorized edge construction (no Python for-loop)
    sources = np.repeat(np.arange(n_cells), k)
    targets = indices.ravel()
    # Deduplicate: keep (min, max) pairs
    lo = np.minimum(sources, targets)
    hi = np.maximum(sources, targets)
    edge_pairs = np.column_stack([lo, hi])
    # Remove self-loops
    mask = edge_pairs[:, 0] != edge_pairs[:, 1]
    edge_pairs = edge_pairs[mask]
    # Unique edges
    edge_pairs = np.unique(edge_pairs, axis=0)
    return edge_pairs


def leiden(
    sj: s_spatioloji,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    n_jobs: int = -1,
    gpu: bool = False,
    input_key: str = "X_pca",
    output_key: str = "leiden",
    force: bool = True,
) -> str:
    """Leiden community detection on a KNN graph.

    Supports both CPU and GPU backends.  CPU mode uses ``pynndescent``
    (approximate KNN, fast) if available, otherwise ``scipy.spatial.cKDTree``.
    GPU mode uses ``cuml`` for KNN and ``cugraph`` for Leiden.

    Args:
        sj: Dataset instance.
        resolution: Resolution parameter (higher = more clusters).
        n_neighbors: Number of neighbours for KNN graph.
        n_jobs: Parallel workers for KNN building (CPU only). ``-1`` = all cores.
        gpu: If ``True``, use ``cuml`` + ``cugraph`` for GPU acceleration.
        input_key: Key to read input from (typically PCA embedding).
        output_key: Key to write cluster labels under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Raises:
        ImportError: If required packages are not installed.

    Example:
        >>> leiden(sj, resolution=1.0)
        'leiden'
        >>> leiden(sj, gpu=True)                     # GPU (~10x faster)
        'leiden'
        >>> leiden(sj, n_neighbors=30, n_jobs=8)     # fast CPU
        'leiden'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, _ = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float32)
    n_cells = matrix.shape[0]

    if gpu:
        try:
            from cuml.neighbors import NearestNeighbors as cuNearestNeighbors
        except ImportError:
            raise ImportError(
                "GPU mode requires cuml + cugraph (NVIDIA RAPIDS). "
                "See https://rapids.ai/start.html"
            ) from None
        try:
            import cugraph
            import cudf
        except ImportError:
            raise ImportError(
                "GPU Leiden requires cugraph. "
                "Install with: pip install cugraph-cu12  (or cu11)"
            ) from None

        k = min(n_neighbors, n_cells - 1)
        nn = cuNearestNeighbors(n_neighbors=k + 1)
        nn.fit(matrix)
        distances, indices = nn.kneighbors(matrix)

        # Convert to numpy if cupy
        if hasattr(indices, "get"):
            indices = indices.get()
        indices = np.asarray(indices)[:, 1:]  # drop self

        # Build edge list
        sources = np.repeat(np.arange(n_cells), k)
        targets = indices.ravel()
        lo = np.minimum(sources, targets)
        hi = np.maximum(sources, targets)
        edge_pairs = np.unique(np.column_stack([lo, hi]), axis=0)
        mask = edge_pairs[:, 0] != edge_pairs[:, 1]
        edge_pairs = edge_pairs[mask]

        # cugraph Leiden
        edge_df = cudf.DataFrame({"src": edge_pairs[:, 0], "dst": edge_pairs[:, 1]})
        G = cugraph.Graph()
        G.from_cudf_edgelist(edge_df, source="src", destination="dst")
        parts, _ = cugraph.leiden(G, resolution=resolution)
        parts = parts.sort_values("vertex").reset_index(drop=True)
        labels = parts["partition"].values
        if hasattr(labels, "get"):
            labels = labels.get()
        labels = np.asarray(labels)
    else:
        try:
            import igraph as ig
            import leidenalg
        except ImportError:
            raise ImportError("Install with: pip install s_spatioloji[clustering]") from None

        edges = _build_knn_edges(matrix, n_neighbors, n_jobs=n_jobs)

        g = ig.Graph(n=n_cells, edges=edges.tolist(), directed=False)
        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution
        )
        labels = partition.membership

    df = pd.DataFrame({"cell_id": cell_ids, output_key: labels})
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def louvain(
    sj: s_spatioloji,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    n_jobs: int = -1,
    input_key: str = "X_pca",
    output_key: str = "louvain",
    force: bool = True,
) -> str:
    """Louvain community detection on a KNN graph.

    Uses the same fast KNN building as :func:`leiden`.

    Args:
        sj: Dataset instance.
        resolution: Resolution parameter (higher = more clusters).
        n_neighbors: Number of neighbours for KNN graph.
        n_jobs: Parallel workers for KNN building. ``-1`` = all cores.
        input_key: Key to read input from (typically PCA embedding).
        output_key: Key to write cluster labels under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Raises:
        ImportError: If ``python-louvain`` is not installed.

    Example:
        >>> louvain(sj, resolution=1.0)
        'louvain'
    """
    try:
        import community as community_louvain
    except ImportError:
        raise ImportError("Install with: pip install s_spatioloji[clustering]") from None

    import networkx as nx

    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, _ = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float32)
    n_cells = matrix.shape[0]

    edges = _build_knn_edges(matrix, n_neighbors, n_jobs=n_jobs)

    G = nx.Graph()
    G.add_nodes_from(range(n_cells))
    G.add_edges_from(edges.tolist())

    partition = community_louvain.best_partition(G, resolution=resolution, random_state=42)
    labels = [partition[i] for i in range(n_cells)]

    df = pd.DataFrame({"cell_id": cell_ids, output_key: labels})
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def kmeans(
    sj: s_spatioloji,
    n_clusters: int = 10,
    input_key: str = "X_pca",
    output_key: str = "kmeans",
    force: bool = True,
) -> str:
    """K-Means clustering.

    Args:
        sj: Dataset instance.
        n_clusters: Number of clusters.
        input_key: Key to read input from (typically PCA embedding).
        output_key: Key to write cluster labels under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> kmeans(sj, n_clusters=10)
        'kmeans'
    """
    from sklearn.cluster import KMeans as _KMeans
    from sklearn.cluster import MiniBatchKMeans as _MiniBatchKMeans

    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, _ = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float32)

    # Use MiniBatchKMeans for large datasets (>500K cells)
    if matrix.shape[0] > 500_000:
        model = _MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10_000)
    else:
        model = _KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(matrix)

    df = pd.DataFrame({"cell_id": cell_ids, output_key: labels})
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def hierarchical(
    sj: s_spatioloji,
    n_clusters: int = 10,
    input_key: str = "X_pca",
    output_key: str = "hierarchical",
    force: bool = True,
) -> str:
    """Agglomerative hierarchical clustering.

    Args:
        sj: Dataset instance.
        n_clusters: Number of clusters.
        input_key: Key to read input from (typically PCA embedding).
        output_key: Key to write cluster labels under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> hierarchical(sj, n_clusters=10)
        'hierarchical'
    """
    from sklearn.cluster import AgglomerativeClustering

    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, _ = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float32)

    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(matrix)

    df = pd.DataFrame({"cell_id": cell_ids, output_key: labels})
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
