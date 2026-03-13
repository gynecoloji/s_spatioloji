"""Contact graph construction from polygon boundaries.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import pandas as pd
from shapely.strtree import STRtree

from s_spatioloji.compute import _atomic_write_parquet

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def build_contact_graph(
    sj: s_spatioloji,
    buffer_distance: float = 0.0,
    output_key: str = "contact_graph",
    force: bool = True,
) -> str:
    """Build a contact graph from polygon boundaries.

    Uses STRtree spatial index for efficient pairwise intersection testing.
    Buffers each geometry by ``buffer_distance`` before testing.

    Args:
        sj: Dataset instance.
        buffer_distance: Gap tolerance in coordinate units (0 = strict touching).
        output_key: Key to write edge list under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> build_contact_graph(sj, buffer_distance=0.0)
        'contact_graph'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    gdf = sj.boundaries.load()
    cell_ids = list(gdf["cell_id"])
    geometries = list(gdf.geometry)

    if buffer_distance > 0:
        buffered = [g.buffer(buffer_distance) for g in geometries]
    else:
        buffered = geometries

    tree = STRtree(buffered)

    edges = []
    seen = set()
    for i, geom in enumerate(buffered):
        candidates = tree.query(geom, predicate="intersects")
        for j in candidates:
            if i >= j:
                continue
            pair = (i, j)
            if pair in seen:
                continue
            seen.add(pair)

            shared = geometries[i].intersection(geometries[j])
            shared_len = shared.length if not shared.is_empty else 0.0

            a, b = cell_ids[i], cell_ids[j]
            if a > b:
                a, b = b, a
            edges.append((a, b, shared_len))

    df = pd.DataFrame(edges, columns=["cell_id_a", "cell_id_b", "shared_length"])
    df = df.sort_values(["cell_id_a", "cell_id_b"]).reset_index(drop=True)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def _load_contact_graph(sj: s_spatioloji, graph_key: str = "contact_graph") -> nx.Graph:
    """Load a contact graph from maps/ as a networkx Graph.

    Args:
        sj: Dataset instance.
        graph_key: Key for the contact graph edge list.

    Returns:
        ``networkx.Graph`` with cell_id nodes and shared_length edge weights.

    Raises:
        FileNotFoundError: If the contact graph Parquet does not exist.
    """
    if not sj.maps.has(graph_key):
        raise FileNotFoundError("Contact graph not found. Run build_contact_graph() first.")

    df = sj.maps[graph_key].compute()
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row["cell_id_a"], row["cell_id_b"], shared_length=row["shared_length"])
    return G
