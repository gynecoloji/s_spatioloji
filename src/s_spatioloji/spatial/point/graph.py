"""KNN and radius graph construction from cell centroids.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.spatial import cKDTree

from s_spatioloji.compute import _atomic_write_parquet

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def build_knn_graph(
    sj: s_spatioloji,
    k: int = 10,
    output_key: str = "knn_graph",
    force: bool = True,
) -> str:
    """Build a KNN graph from cell centroids.

    Uses ``scipy.spatial.cKDTree`` for O(N log N) nearest-neighbor queries.

    Args:
        sj: Dataset instance.
        k: Number of nearest neighbors per cell.
        output_key: Key to write edge list under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> build_knn_graph(sj, k=10)
        'knn_graph'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    cells_df = sj.cells.df.compute()
    cell_ids = cells_df["cell_id"].values
    coords = cells_df[["x", "y"]].values.astype(np.float64)

    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=k + 1)  # +1 because self is included

    edges_set: set[tuple[str, str, float]] = set()
    for i in range(len(cell_ids)):
        for j_idx in range(1, k + 1):  # skip index 0 (self)
            j = indices[i, j_idx]
            dist = float(distances[i, j_idx])
            a, b = str(cell_ids[i]), str(cell_ids[j])
            if a > b:
                a, b = b, a
            # Clamp zero distance to epsilon
            dist = max(dist, 1e-10)
            edges_set.add((a, b, dist))

    edges = sorted(edges_set)
    df = pd.DataFrame(edges, columns=["cell_id_a", "cell_id_b", "distance"])
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def build_radius_graph(
    sj: s_spatioloji,
    radius: float,
    output_key: str = "radius_graph",
    force: bool = True,
) -> str:
    """Build a radius graph from cell centroids.

    Uses ``cKDTree.query_ball_point`` for efficient radius queries.

    Args:
        sj: Dataset instance.
        radius: Distance threshold in coordinate units (required, no default).
        output_key: Key to write edge list under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> build_radius_graph(sj, radius=50.0)
        'radius_graph'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    cells_df = sj.cells.df.compute()
    cell_ids = cells_df["cell_id"].values
    coords = cells_df[["x", "y"]].values.astype(np.float64)

    tree = cKDTree(coords)
    neighbors_list = tree.query_ball_point(coords, r=radius)

    edges_set: set[tuple[str, str, float]] = set()
    for i, neighbors in enumerate(neighbors_list):
        for j in neighbors:
            if i == j:
                continue
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            a, b = str(cell_ids[i]), str(cell_ids[j])
            if a > b:
                a, b = b, a
            dist = max(dist, 1e-10)
            edges_set.add((a, b, dist))

    edges = sorted(edges_set)
    df = pd.DataFrame(edges, columns=["cell_id_a", "cell_id_b", "distance"])
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def _load_point_graph(sj: s_spatioloji, graph_key: str = "knn_graph") -> nx.Graph:
    """Load a point graph from maps/ as a networkx Graph.

    Args:
        sj: Dataset instance.
        graph_key: Key for the graph edge list.

    Returns:
        ``networkx.Graph`` with cell_id nodes and distance edge weights.

    Raises:
        FileNotFoundError: If the graph Parquet does not exist.
    """
    if not sj.maps.has(graph_key):
        raise FileNotFoundError("Point graph not found. Run build_knn_graph() or build_radius_graph() first.")

    df = sj.maps[graph_key].compute()
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row["cell_id_a"], row["cell_id_b"], distance=row["distance"])
    return G


def _load_point_graph_sparse(
    sj: s_spatioloji,
    graph_key: str = "knn_graph",
    weighted: bool = False,
) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
    """Load a point graph as a sparse adjacency matrix.

    Args:
        sj: Dataset instance.
        graph_key: Key for the graph edge list.
        weighted: If ``True``, use inverse-distance weights (``1/max(d, 1e-10)``).
            If ``False``, binary adjacency (0/1).

    Returns:
        Tuple of ``(adjacency_matrix, cell_ids)`` where ``cell_ids`` maps
        matrix indices to cell_id strings.

    Raises:
        FileNotFoundError: If the graph Parquet does not exist.
    """
    if not sj.maps.has(graph_key):
        raise FileNotFoundError("Point graph not found. Run build_knn_graph() or build_radius_graph() first.")

    df = sj.maps[graph_key].compute()
    all_ids = sorted(set(df["cell_id_a"]).union(set(df["cell_id_b"])))
    cell_ids = np.array(all_ids)
    id_to_idx = {cid: i for i, cid in enumerate(all_ids)}
    n = len(all_ids)

    rows = df["cell_id_a"].map(id_to_idx).values
    cols = df["cell_id_b"].map(id_to_idx).values

    if weighted:
        distances = df["distance"].values.astype(np.float64)
        distances = np.maximum(distances, 1e-10)
        vals = 1.0 / distances
    else:
        vals = np.ones(len(df), dtype=np.float64)

    # Build symmetric matrix
    row_idx = np.concatenate([rows, cols])
    col_idx = np.concatenate([cols, rows])
    data = np.concatenate([vals, vals])

    adj = scipy.sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
    return adj, cell_ids
