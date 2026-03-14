"""Neighborhood analysis functions for point-based spatial data.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.

Uses sparse matrix operations for O(N*k) scalability to 50-100M cells.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse

from s_spatioloji.compute import _atomic_write_parquet
from s_spatioloji.spatial.point.graph import _load_point_graph_sparse

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def neighborhood_composition(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "knn_graph",
    weighted: bool = False,
    output_key: str = "pt_nhood_composition",
    force: bool = True,
) -> str:
    """Count neighbor cluster types for each cell using sparse matrix multiplication.

    Args:
        sj: Dataset instance.
        cluster_key: Key for cluster labels in maps/.
        graph_key: Key for the graph edge list.
        weighted: If ``True``, weight by inverse distance.
        output_key: Key to write composition table under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Raises:
        FileNotFoundError: If graph not built.

    Example:
        >>> neighborhood_composition(sj, cluster_key="leiden")
        'pt_nhood_composition'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    adj, graph_cell_ids = _load_point_graph_sparse(sj, graph_key, weighted=weighted)

    cluster_df = sj.maps[cluster_key].compute()
    cell_to_cluster = dict(zip(cluster_df["cell_id"], cluster_df[cluster_key], strict=True))
    all_clusters = sorted(set(cell_to_cluster.values()), key=str)

    n = len(graph_cell_ids)
    n_clusters = len(all_clusters)
    cluster_to_idx = {str(c): i for i, c in enumerate(all_clusters)}

    indicator_rows = []
    indicator_cols = []
    indicator_data = []
    for i, cid in enumerate(graph_cell_ids):
        if cid in cell_to_cluster:
            label = str(cell_to_cluster[cid])
            if label in cluster_to_idx:
                indicator_rows.append(i)
                indicator_cols.append(cluster_to_idx[label])
                indicator_data.append(1.0)

    indicator = scipy.sparse.csr_matrix((indicator_data, (indicator_rows, indicator_cols)), shape=(n, n_clusters))

    # Sparse matrix multiplication: adjacency @ indicator
    composition = (adj @ indicator).toarray()

    records = {"cell_id": graph_cell_ids}
    for j, label in enumerate(all_clusters):
        records[str(label)] = composition[:, j]

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def nth_order_neighbors(
    sj: s_spatioloji,
    order: int = 2,
    graph_key: str = "knn_graph",
    output_key: str = "pt_nhood_nth_order",
    force: bool = True,
) -> str:
    """Count neighbors at each hop distance using sparse matrix powers.

    Counts at each order are exclusive (order 2 excludes order 1 neighbors).
    Excludes the cell itself.

    Args:
        sj: Dataset instance.
        order: Maximum hop distance.
        graph_key: Key for the graph edge list.
        output_key: Key to write neighbor counts under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> nth_order_neighbors(sj, order=2)
        'pt_nhood_nth_order'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    adj, graph_cell_ids = _load_point_graph_sparse(sj, graph_key, weighted=False)
    n = adj.shape[0]

    order_counts = np.zeros((n, order), dtype=np.int64)

    identity = scipy.sparse.eye(n, format="csr")
    prev_reachable = identity  # self only
    A_power = adj.copy()

    for k in range(1, order + 1):
        reachable_k = (A_power > 0).astype(np.float64)
        already = (prev_reachable > 0).astype(np.float64)
        new_at_k = reachable_k - already
        new_at_k.data[new_at_k.data < 0] = 0
        new_at_k.eliminate_zeros()

        order_counts[:, k - 1] = np.asarray(new_at_k.sum(axis=1)).ravel().astype(np.int64)

        prev_reachable = prev_reachable + reachable_k
        if k < order:
            A_power = A_power @ adj

    records = {"cell_id": graph_cell_ids}
    for k in range(1, order + 1):
        records[f"n_order_{k}"] = order_counts[:, k - 1]

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def neighborhood_diversity(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "knn_graph",
    weighted: bool = False,
    output_key: str = "pt_nhood_diversity",
    force: bool = True,
) -> str:
    """Shannon entropy and Gini-Simpson index of neighbor cluster composition.

    Args:
        sj: Dataset instance.
        cluster_key: Key for cluster labels in maps/.
        graph_key: Key for the graph edge list.
        weighted: If ``True``, use inverse-distance weights for proportions.
        output_key: Key to write diversity metrics under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> neighborhood_diversity(sj, cluster_key="leiden")
        'pt_nhood_diversity'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    adj, graph_cell_ids = _load_point_graph_sparse(sj, graph_key, weighted=weighted)

    cluster_df = sj.maps[cluster_key].compute()
    cell_to_cluster = dict(zip(cluster_df["cell_id"], cluster_df[cluster_key], strict=True))
    all_clusters = sorted(set(cell_to_cluster.values()), key=str)

    n = len(graph_cell_ids)
    n_clusters = len(all_clusters)
    cluster_to_idx = {str(c): i for i, c in enumerate(all_clusters)}

    indicator_rows = []
    indicator_cols = []
    for i, cid in enumerate(graph_cell_ids):
        if cid in cell_to_cluster:
            label = str(cell_to_cluster[cid])
            if label in cluster_to_idx:
                indicator_rows.append(i)
                indicator_cols.append(cluster_to_idx[label])

    indicator = scipy.sparse.csr_matrix(
        (np.ones(len(indicator_rows)), (indicator_rows, indicator_cols)),
        shape=(n, n_clusters),
    )

    composition_dense = (adj @ indicator).toarray()

    row_sums = composition_dense.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = composition_dense / row_sums

    with np.errstate(divide="ignore", invalid="ignore"):
        log_probs = np.log(probs + 1e-15)
    shannon = -np.sum(probs * log_probs, axis=1)
    shannon = np.maximum(shannon, 0.0)

    simpson = 1.0 - np.sum(probs**2, axis=1)

    no_neighbors = composition_dense.sum(axis=1) == 0
    shannon[no_neighbors] = 0.0
    simpson[no_neighbors] = 0.0

    df = pd.DataFrame(
        {
            "cell_id": graph_cell_ids,
            "shannon": shannon,
            "simpson": simpson,
        }
    )
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
