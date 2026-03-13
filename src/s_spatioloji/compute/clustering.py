"""Clustering functions.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from s_spatioloji.compute import _atomic_write_parquet, _load_dense

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def leiden(
    sj: s_spatioloji,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    input_key: str = "X_pca",
    output_key: str = "leiden",
    force: bool = True,
) -> str:
    """Leiden community detection on a KNN graph.

    Builds a KNN graph from the input embedding and partitions it using
    the Leiden algorithm.  ``n_neighbors`` is clamped to
    ``min(n_neighbors, n_cells - 1)``.

    Args:
        sj: Dataset instance.
        resolution: Resolution parameter (higher → more clusters).
        n_neighbors: Number of neighbours for KNN graph.
        input_key: Key to read input from (typically PCA embedding).
        output_key: Key to write cluster labels under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Raises:
        ImportError: If ``leidenalg`` or ``igraph`` is not installed.

    Example:
        >>> leiden(sj, resolution=1.0)
        'leiden'
    """
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        raise ImportError("Install with: pip install s_spatioloji[clustering]") from None

    from sklearn.neighbors import NearestNeighbors

    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, _ = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float32)
    n_cells = matrix.shape[0]
    k = min(n_neighbors, n_cells - 1)

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(matrix)
    _, indices = nn.kneighbors()

    edges = set()
    for i in range(n_cells):
        for j in indices[i]:
            if i != j:
                edges.add((min(i, j), max(i, j)))

    g = ig.Graph(n=n_cells, edges=list(edges), directed=False)
    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution
    )

    df = pd.DataFrame({"cell_id": cell_ids, output_key: partition.membership})
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def louvain(
    sj: s_spatioloji,
    resolution: float = 1.0,
    n_neighbors: int = 15,
    input_key: str = "X_pca",
    output_key: str = "louvain",
    force: bool = True,
) -> str:
    """Louvain community detection on a KNN graph.

    ``n_neighbors`` is clamped to ``min(n_neighbors, n_cells - 1)``.

    Args:
        sj: Dataset instance.
        resolution: Resolution parameter (higher → more clusters).
        n_neighbors: Number of neighbours for KNN graph.
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
    from sklearn.neighbors import NearestNeighbors

    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, _ = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float32)
    n_cells = matrix.shape[0]
    k = min(n_neighbors, n_cells - 1)

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(matrix)
    _, indices = nn.kneighbors()

    G = nx.Graph()
    G.add_nodes_from(range(n_cells))
    for i in range(n_cells):
        for j in indices[i]:
            if i != j:
                G.add_edge(i, j)

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

    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, _ = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float32)

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
