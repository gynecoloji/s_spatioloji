"""Neighborhood analysis functions for polygon-based spatial data.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from s_spatioloji.compute import _atomic_write_parquet
from s_spatioloji.spatial.polygon.graph import _load_contact_graph

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def neighborhood_composition(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "contact_graph",
    output_key: str = "nhood_composition",
    force: bool = True,
) -> str:
    """Count neighbor cluster types for each cell.

    Args:
        sj: Dataset instance.
        cluster_key: Key for cluster labels in maps/.
        graph_key: Key for the contact graph edge list.
        output_key: Key to write composition table under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Raises:
        FileNotFoundError: If contact graph not built.

    Example:
        >>> neighborhood_composition(sj, cluster_key="leiden")
        'nhood_composition'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    G = _load_contact_graph(sj, graph_key)

    cluster_df = sj.maps[cluster_key].compute()
    cell_to_cluster = dict(zip(cluster_df["cell_id"], cluster_df[cluster_key], strict=True))
    all_clusters = sorted(set(cell_to_cluster.values()), key=str)

    cells_df = sj.cells.df.compute()
    all_cell_ids = list(cells_df["cell_id"])

    records = []
    for cell_id in all_cell_ids:
        counts = {str(c): 0 for c in all_clusters}
        if G.has_node(cell_id):
            for neighbor in G.neighbors(cell_id):
                if neighbor in cell_to_cluster:
                    label = str(cell_to_cluster[neighbor])
                    counts[label] += 1
        record = {"cell_id": cell_id}
        record.update(counts)
        records.append(record)

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def nth_order_neighbors(
    sj: s_spatioloji,
    order: int = 2,
    graph_key: str = "contact_graph",
    output_key: str = "nhood_nth_order",
    force: bool = True,
) -> str:
    """Count neighbors at each hop distance up to ``order``.

    Uses BFS. Counts at each order are exclusive (order 2 excludes order 1
    neighbors). Excludes the cell itself.

    Args:
        sj: Dataset instance.
        order: Maximum hop distance.
        graph_key: Key for the contact graph edge list.
        output_key: Key to write neighbor counts under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> nth_order_neighbors(sj, order=2)
        'nhood_nth_order'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    G = _load_contact_graph(sj, graph_key)

    cells_df = sj.cells.df.compute()
    all_cell_ids = list(cells_df["cell_id"])
    order_cols = [f"n_order_{k}" for k in range(1, order + 1)]

    records = []
    for cell_id in all_cell_ids:
        counts = {col: 0 for col in order_cols}
        if G.has_node(cell_id):
            visited = {cell_id}
            current_level = {cell_id}
            for k in range(1, order + 1):
                next_level = set()
                for node in current_level:
                    for neighbor in G.neighbors(node):
                        if neighbor not in visited:
                            next_level.add(neighbor)
                visited.update(next_level)
                counts[f"n_order_{k}"] = len(next_level)
                current_level = next_level
        record = {"cell_id": cell_id}
        record.update(counts)
        records.append(record)

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def neighborhood_diversity(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "contact_graph",
    output_key: str = "nhood_diversity",
    force: bool = True,
) -> str:
    """Shannon entropy and Gini-Simpson index of neighbor cluster composition.

    Args:
        sj: Dataset instance.
        cluster_key: Key for cluster labels in maps/.
        graph_key: Key for the contact graph edge list.
        output_key: Key to write diversity metrics under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> neighborhood_diversity(sj, cluster_key="leiden")
        'nhood_diversity'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    G = _load_contact_graph(sj, graph_key)

    cluster_df = sj.maps[cluster_key].compute()
    cell_to_cluster = dict(zip(cluster_df["cell_id"], cluster_df[cluster_key], strict=True))

    cells_df = sj.cells.df.compute()
    all_cell_ids = list(cells_df["cell_id"])

    records = []
    for cell_id in all_cell_ids:
        neighbor_labels = []
        if G.has_node(cell_id):
            for neighbor in G.neighbors(cell_id):
                if neighbor in cell_to_cluster:
                    neighbor_labels.append(cell_to_cluster[neighbor])

        if len(neighbor_labels) == 0:
            records.append({"cell_id": cell_id, "shannon": 0.0, "simpson": 0.0})
            continue

        counts = defaultdict(int)
        for lab in neighbor_labels:
            counts[lab] += 1
        total = len(neighbor_labels)
        probs = np.array([c / total for c in counts.values()])

        shannon = float(max(0.0, -np.sum(probs * np.log(probs + 1e-15))))
        simpson = 1.0 - np.sum(probs**2)

        records.append({"cell_id": cell_id, "shannon": shannon, "simpson": float(simpson)})

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
