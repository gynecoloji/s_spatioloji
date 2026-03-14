# Spatial Point Module Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement point-based (centroid) spatial analysis module with KNN/radius graphs, neighborhoods, patterns, Ripley's functions, and statistical tests — scalable to 50–100M cells.

**Architecture:** All functions follow the compute layer contract (accept `s_spatioloji`, write Parquet to `maps/`, return output key). Graph built with `cKDTree`, loaded as sparse matrix for vectorized ops. `pt_` prefix on all output keys. Networkx only for clustering coefficient (with 5M cell guard).

**Tech Stack:** scipy (cKDTree, sparse, stats), numpy, pandas, networkx (limited use)

**Spec:** `docs/superpowers/specs/2026-03-13-spatial-point-design.md`

**Reference implementation:** `src/s_spatioloji/spatial/polygon/` (same contract, different algorithms)

---

## File Structure

```
src/s_spatioloji/spatial/point/
├── __init__.py          # Lazy imports of all public functions
├── graph.py             # build_knn_graph, build_radius_graph, _load_point_graph, _load_point_graph_sparse
├── neighborhoods.py     # neighborhood_composition, nth_order_neighbors, neighborhood_diversity
├── patterns.py          # colocalization, morans_i, gearys_c, clustering_coefficient, getis_ord_gi
├── ripley.py            # ripley_k, ripley_l, ripley_g, ripley_f
└── statistics.py        # permutation_test, quadrat_density, clark_evans, dclf_envelope

tests/unit/
├── conftest.py          # Add sj_with_knn_graph, sj_with_pt_clusters fixtures
├── test_spatial_point_graph.py
├── test_spatial_point_neighborhoods.py
├── test_spatial_point_patterns.py
├── test_spatial_point_ripley.py
└── test_spatial_point_statistics.py
```

---

## Chunk 1: Graph Layer (Tasks 1–2)

### Task 1: Test fixtures + graph.py tests

**Files:**
- Modify: `tests/unit/conftest.py` (add `sj_with_knn_graph`, `sj_with_pt_clusters`)
- Create: `tests/unit/test_spatial_point_graph.py`

- [ ] **Step 1: Add point fixtures to conftest.py**

Add after the existing `sj_with_clusters` fixture at end of file:

```python
@pytest.fixture()
def sj_with_knn_graph(sj):
    """sj + pre-built KNN graph (k=6)."""
    from s_spatioloji.spatial.point.graph import build_knn_graph

    build_knn_graph(sj, k=6)
    return sj


@pytest.fixture()
def sj_with_pt_clusters(sj_with_knn_graph):
    """sj_with_knn_graph + synthetic cluster labels in maps/leiden.parquet."""
    cells_df = sj_with_knn_graph.cells.df.compute()
    x = cells_df["x"].values
    y = cells_df["y"].values
    median_x = np.median(x)
    median_y = np.median(y)
    labels = np.where(
        x < median_x,
        np.where(y < median_y, 0, 1),
        np.where(y < median_y, 2, 3),
    )
    df = pd.DataFrame({"cell_id": cells_df["cell_id"].values, "leiden": labels})
    from s_spatioloji.compute import _atomic_write_parquet

    maps_dir = sj_with_knn_graph.config.root / "maps"
    maps_dir.mkdir(exist_ok=True)
    _atomic_write_parquet(df, maps_dir, "leiden")
    return sj_with_knn_graph
```

- [ ] **Step 2: Write graph tests**

Create `tests/unit/test_spatial_point_graph.py`:

```python
"""Unit tests for s_spatioloji.spatial.point.graph."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import scipy.sparse

from s_spatioloji.spatial.point.graph import (
    _load_point_graph,
    _load_point_graph_sparse,
    build_knn_graph,
    build_radius_graph,
)


class TestBuildKnnGraph:
    def test_returns_key(self, sj):
        assert build_knn_graph(sj, k=6) == "knn_graph"

    def test_output_written(self, sj):
        build_knn_graph(sj, k=6)
        assert sj.maps.has("knn_graph")

    def test_columns(self, sj):
        build_knn_graph(sj, k=6)
        df = sj.maps["knn_graph"].compute()
        assert list(df.columns) == ["cell_id_a", "cell_id_b", "distance"]

    def test_edges_exist(self, sj):
        build_knn_graph(sj, k=6)
        df = sj.maps["knn_graph"].compute()
        assert len(df) > 100

    def test_edge_ordering(self, sj):
        """cell_id_a < cell_id_b lexicographically."""
        build_knn_graph(sj, k=6)
        df = sj.maps["knn_graph"].compute()
        assert (df["cell_id_a"] < df["cell_id_b"]).all()

    def test_distance_positive(self, sj):
        build_knn_graph(sj, k=6)
        df = sj.maps["knn_graph"].compute()
        assert (df["distance"] > 0).all()

    def test_force_false_skips(self, sj):
        build_knn_graph(sj, k=6)
        build_knn_graph(sj, k=6, force=False)

    def test_custom_output_key(self, sj):
        assert build_knn_graph(sj, k=6, output_key="my_knn") == "my_knn"
        assert sj.maps.has("my_knn")


class TestBuildRadiusGraph:
    def test_returns_key(self, sj):
        assert build_radius_graph(sj, radius=60.0) == "radius_graph"

    def test_output_written(self, sj):
        build_radius_graph(sj, radius=60.0)
        assert sj.maps.has("radius_graph")

    def test_columns(self, sj):
        build_radius_graph(sj, radius=60.0)
        df = sj.maps["radius_graph"].compute()
        assert list(df.columns) == ["cell_id_a", "cell_id_b", "distance"]

    def test_edges_within_radius(self, sj):
        build_radius_graph(sj, radius=60.0)
        df = sj.maps["radius_graph"].compute()
        assert (df["distance"] <= 60.0).all()
        assert (df["distance"] > 0).all()

    def test_force_false_skips(self, sj):
        build_radius_graph(sj, radius=60.0)
        build_radius_graph(sj, radius=60.0, force=False)


class TestLoadPointGraph:
    def test_returns_networkx_graph(self, sj_with_knn_graph):
        G = _load_point_graph(sj_with_knn_graph)
        assert isinstance(G, nx.Graph)

    def test_node_count(self, sj_with_knn_graph):
        G = _load_point_graph(sj_with_knn_graph)
        assert len(G.nodes) > 100

    def test_edge_weights(self, sj_with_knn_graph):
        G = _load_point_graph(sj_with_knn_graph)
        for _, _, data in G.edges(data=True):
            assert "distance" in data
            assert data["distance"] > 0

    def test_missing_graph_raises(self, sj):
        with pytest.raises(FileNotFoundError, match="Point graph not found"):
            _load_point_graph(sj)


class TestLoadPointGraphSparse:
    def test_returns_tuple(self, sj_with_knn_graph):
        adj, cell_ids = _load_point_graph_sparse(sj_with_knn_graph)
        assert isinstance(adj, scipy.sparse.csr_matrix)
        assert isinstance(cell_ids, np.ndarray)

    def test_symmetric(self, sj_with_knn_graph):
        adj, _ = _load_point_graph_sparse(sj_with_knn_graph)
        diff = adj - adj.T
        assert diff.nnz == 0

    def test_binary_unweighted(self, sj_with_knn_graph):
        adj, _ = _load_point_graph_sparse(sj_with_knn_graph)
        assert set(adj.data).issubset({1.0, 1})

    def test_weighted_inverse_distance(self, sj_with_knn_graph):
        adj, _ = _load_point_graph_sparse(sj_with_knn_graph, weighted=True)
        assert (adj.data > 0).all()

    def test_missing_graph_raises(self, sj):
        with pytest.raises(FileNotFoundError):
            _load_point_graph_sparse(sj)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/unit/test_spatial_point_graph.py -v`
Expected: FAIL (module not found)

- [ ] **Step 4: Commit test files**

```bash
git add tests/unit/conftest.py tests/unit/test_spatial_point_graph.py
git commit -m "test(spatial/point): add graph test suite and point fixtures"
```

### Task 2: Implement graph.py + __init__.py

**Files:**
- Create: `src/s_spatioloji/spatial/point/__init__.py`
- Create: `src/s_spatioloji/spatial/point/graph.py`

- [ ] **Step 1: Create `__init__.py`**

```python
"""Point-based spatial analysis.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "build_knn_graph": ("s_spatioloji.spatial.point.graph", "build_knn_graph"),
    "build_radius_graph": ("s_spatioloji.spatial.point.graph", "build_radius_graph"),
    "neighborhood_composition": ("s_spatioloji.spatial.point.neighborhoods", "neighborhood_composition"),
    "neighborhood_diversity": ("s_spatioloji.spatial.point.neighborhoods", "neighborhood_diversity"),
    "nth_order_neighbors": ("s_spatioloji.spatial.point.neighborhoods", "nth_order_neighbors"),
    "colocalization": ("s_spatioloji.spatial.point.patterns", "colocalization"),
    "morans_i": ("s_spatioloji.spatial.point.patterns", "morans_i"),
    "gearys_c": ("s_spatioloji.spatial.point.patterns", "gearys_c"),
    "clustering_coefficient": ("s_spatioloji.spatial.point.patterns", "clustering_coefficient"),
    "getis_ord_gi": ("s_spatioloji.spatial.point.patterns", "getis_ord_gi"),
    "ripley_k": ("s_spatioloji.spatial.point.ripley", "ripley_k"),
    "ripley_l": ("s_spatioloji.spatial.point.ripley", "ripley_l"),
    "ripley_g": ("s_spatioloji.spatial.point.ripley", "ripley_g"),
    "ripley_f": ("s_spatioloji.spatial.point.ripley", "ripley_f"),
    "permutation_test": ("s_spatioloji.spatial.point.statistics", "permutation_test"),
    "quadrat_density": ("s_spatioloji.spatial.point.statistics", "quadrat_density"),
    "clark_evans": ("s_spatioloji.spatial.point.statistics", "clark_evans"),
    "dclf_envelope": ("s_spatioloji.spatial.point.statistics", "dclf_envelope"),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

- [ ] **Step 2: Create `graph.py`**

```python
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
        raise FileNotFoundError(
            "Point graph not found. Run build_knn_graph() or build_radius_graph() first."
        )

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
        raise FileNotFoundError(
            "Point graph not found. Run build_knn_graph() or build_radius_graph() first."
        )

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
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/unit/test_spatial_point_graph.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add src/s_spatioloji/spatial/point/__init__.py src/s_spatioloji/spatial/point/graph.py
git commit -m "feat(spatial/point): add KNN/radius graph building with sparse loader"
```

---

## Chunk 2: Neighborhoods (Task 3)

### Task 3: neighborhoods.py

**Files:**
- Create: `tests/unit/test_spatial_point_neighborhoods.py`
- Create: `src/s_spatioloji/spatial/point/neighborhoods.py`

- [ ] **Step 1: Write neighborhood tests**

Create `tests/unit/test_spatial_point_neighborhoods.py`:

```python
"""Unit tests for s_spatioloji.spatial.point.neighborhoods."""

from __future__ import annotations

from s_spatioloji.spatial.point.neighborhoods import (
    neighborhood_composition,
    neighborhood_diversity,
    nth_order_neighbors,
)


class TestNeighborhoodComposition:
    def test_returns_key(self, sj_with_pt_clusters):
        assert neighborhood_composition(sj_with_pt_clusters) == "pt_nhood_composition"

    def test_output_written(self, sj_with_pt_clusters):
        neighborhood_composition(sj_with_pt_clusters)
        assert sj_with_pt_clusters.maps.has("pt_nhood_composition")

    def test_columns_start_with_cell_id(self, sj_with_pt_clusters):
        neighborhood_composition(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_nhood_composition"].compute()
        assert df.columns[0] == "cell_id"
        assert df.shape[0] == 200

    def test_weighted_differs(self, sj_with_pt_clusters):
        neighborhood_composition(sj_with_pt_clusters, weighted=False, output_key="pt_nc_uw")
        neighborhood_composition(sj_with_pt_clusters, weighted=True, output_key="pt_nc_w")
        df_uw = sj_with_pt_clusters.maps["pt_nc_uw"].compute()
        df_w = sj_with_pt_clusters.maps["pt_nc_w"].compute()
        # Weighted values should differ from unweighted counts
        cols = [c for c in df_uw.columns if c != "cell_id"]
        assert not df_uw[cols].equals(df_w[cols])

    def test_force_false_skips(self, sj_with_pt_clusters):
        neighborhood_composition(sj_with_pt_clusters)
        neighborhood_composition(sj_with_pt_clusters, force=False)


class TestNthOrderNeighbors:
    def test_returns_key(self, sj_with_knn_graph):
        assert nth_order_neighbors(sj_with_knn_graph) == "pt_nhood_nth_order"

    def test_columns(self, sj_with_knn_graph):
        nth_order_neighbors(sj_with_knn_graph, order=2)
        df = sj_with_knn_graph.maps["pt_nhood_nth_order"].compute()
        assert list(df.columns) == ["cell_id", "n_order_1", "n_order_2"]

    def test_shape(self, sj_with_knn_graph):
        nth_order_neighbors(sj_with_knn_graph, order=2)
        df = sj_with_knn_graph.maps["pt_nhood_nth_order"].compute()
        assert df.shape[0] == 200

    def test_order_1_positive(self, sj_with_knn_graph):
        nth_order_neighbors(sj_with_knn_graph, order=1)
        df = sj_with_knn_graph.maps["pt_nhood_nth_order"].compute()
        assert (df["n_order_1"] > 0).all()

    def test_force_false_skips(self, sj_with_knn_graph):
        nth_order_neighbors(sj_with_knn_graph)
        nth_order_neighbors(sj_with_knn_graph, force=False)


class TestNeighborhoodDiversity:
    def test_returns_key(self, sj_with_pt_clusters):
        assert neighborhood_diversity(sj_with_pt_clusters) == "pt_nhood_diversity"

    def test_columns(self, sj_with_pt_clusters):
        neighborhood_diversity(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_nhood_diversity"].compute()
        assert list(df.columns) == ["cell_id", "shannon", "simpson"]

    def test_shape(self, sj_with_pt_clusters):
        neighborhood_diversity(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_nhood_diversity"].compute()
        assert df.shape[0] == 200

    def test_shannon_nonneg(self, sj_with_pt_clusters):
        neighborhood_diversity(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_nhood_diversity"].compute()
        assert (df["shannon"] >= 0).all()

    def test_simpson_range(self, sj_with_pt_clusters):
        neighborhood_diversity(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_nhood_diversity"].compute()
        assert (df["simpson"] >= 0).all()
        assert (df["simpson"] <= 1.0).all()

    def test_force_false_skips(self, sj_with_pt_clusters):
        neighborhood_diversity(sj_with_pt_clusters)
        neighborhood_diversity(sj_with_pt_clusters, force=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_spatial_point_neighborhoods.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `neighborhoods.py`**

```python
"""Neighborhood analysis functions for point-based spatial data.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.

Uses sparse matrix operations for O(N*k) scalability to 50–100M cells.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

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

    # Build cluster indicator matrix (N x C)
    import scipy.sparse

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

    indicator = scipy.sparse.csr_matrix(
        (indicator_data, (indicator_rows, indicator_cols)), shape=(n, n_clusters)
    )

    # Sparse matrix multiplication: adjacency @ indicator
    composition = adj @ indicator  # (N x C) — counts (or weighted sums) per cluster

    records = []
    for i, cid in enumerate(graph_cell_ids):
        record = {"cell_id": cid}
        row = composition.getrow(i).toarray().ravel()
        for j, label in enumerate(all_clusters):
            record[str(label)] = float(row[j])
        records.append(record)

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

    # Compute reachability at each order via sparse matrix powers
    # A^k gives reachability at order k
    order_counts = np.zeros((n, order), dtype=np.int64)

    # prev_reachable tracks cumulative reachable set (as binary matrix > 0)
    import scipy.sparse

    identity = scipy.sparse.eye(n, format="csr")
    prev_reachable = identity  # self only
    A_power = adj.copy()

    for k in range(1, order + 1):
        # Cells reachable at exactly order k (but not before)
        reachable_k = (A_power > 0).astype(np.float64)
        already = (prev_reachable > 0).astype(np.float64)
        new_at_k = reachable_k - already
        new_at_k.data[new_at_k.data < 0] = 0  # clip negative
        new_at_k.eliminate_zeros()

        order_counts[:, k - 1] = np.asarray(new_at_k.sum(axis=1)).ravel().astype(np.int64)

        prev_reachable = prev_reachable + reachable_k
        if k < order:
            A_power = A_power @ adj

    records = []
    for i, cid in enumerate(graph_cell_ids):
        record = {"cell_id": cid}
        for k in range(1, order + 1):
            record[f"n_order_{k}"] = int(order_counts[i, k - 1])
        records.append(record)

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

    import scipy.sparse

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

    composition = adj @ indicator  # (N x C)
    composition_dense = composition.toarray()

    row_sums = composition_dense.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid division by zero
    probs = composition_dense / row_sums

    # Shannon entropy: -sum(p * log(p))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probs = np.log(probs + 1e-15)
    shannon = -np.sum(probs * log_probs, axis=1)
    shannon = np.maximum(shannon, 0.0)

    # Gini-Simpson: 1 - sum(p^2)
    simpson = 1.0 - np.sum(probs**2, axis=1)

    # Cells with no neighbors get 0
    no_neighbors = composition_dense.sum(axis=1) == 0
    shannon[no_neighbors] = 0.0
    simpson[no_neighbors] = 0.0

    df = pd.DataFrame({
        "cell_id": graph_cell_ids,
        "shannon": shannon,
        "simpson": simpson,
    })
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_spatial_point_neighborhoods.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/s_spatioloji/spatial/point/neighborhoods.py tests/unit/test_spatial_point_neighborhoods.py
git commit -m "feat(spatial/point): add neighborhood composition, nth-order, and diversity"
```

---

## Chunk 3: Patterns (Task 4)

### Task 4: patterns.py

**Files:**
- Create: `tests/unit/test_spatial_point_patterns.py`
- Create: `src/s_spatioloji/spatial/point/patterns.py`

- [ ] **Step 1: Write pattern tests**

Create `tests/unit/test_spatial_point_patterns.py`:

```python
"""Unit tests for s_spatioloji.spatial.point.patterns."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from s_spatioloji.spatial.point.patterns import (
    clustering_coefficient,
    colocalization,
    gearys_c,
    getis_ord_gi,
    morans_i,
)


class TestColocalization:
    def test_returns_key(self, sj_with_pt_clusters):
        assert colocalization(sj_with_pt_clusters) == "pt_colocalization"

    def test_output_written(self, sj_with_pt_clusters):
        colocalization(sj_with_pt_clusters)
        assert sj_with_pt_clusters.maps.has("pt_colocalization")

    def test_columns(self, sj_with_pt_clusters):
        colocalization(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_colocalization"].compute()
        assert list(df.columns) == [
            "cluster_a", "cluster_b", "observed", "expected", "ratio", "log2_ratio",
        ]

    def test_has_rows(self, sj_with_pt_clusters):
        colocalization(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_colocalization"].compute()
        assert len(df) > 0

    def test_observed_nonneg(self, sj_with_pt_clusters):
        colocalization(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_colocalization"].compute()
        assert (df["observed"] >= 0).all()

    def test_force_false_skips(self, sj_with_pt_clusters):
        colocalization(sj_with_pt_clusters)
        colocalization(sj_with_pt_clusters, force=False)


class TestMoransI:
    def test_returns_key(self, sj_with_pt_clusters):
        assert morans_i(sj_with_pt_clusters) == "pt_morans_i"

    def test_columns(self, sj_with_pt_clusters):
        morans_i(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_morans_i"].compute()
        assert list(df.columns) == ["feature", "I", "expected_I", "z_score", "p_value"]

    def test_numeric_produces_one_row(self, sj_with_pt_clusters):
        morans_i(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_morans_i"].compute()
        assert len(df) == 1

    def test_spatially_clustered_high_I(self, sj_with_pt_clusters):
        morans_i(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_morans_i"].compute()
        assert (df["I"] > 0).any()

    def test_force_false_skips(self, sj_with_pt_clusters):
        morans_i(sj_with_pt_clusters)
        morans_i(sj_with_pt_clusters, force=False)


class TestGearysC:
    def test_returns_key(self, sj_with_pt_clusters):
        assert gearys_c(sj_with_pt_clusters) == "pt_gearys_c"

    def test_columns(self, sj_with_pt_clusters):
        gearys_c(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_gearys_c"].compute()
        assert list(df.columns) == ["feature", "C", "expected_C", "z_score", "p_value"]

    def test_spatially_clustered_low_C(self, sj_with_pt_clusters):
        gearys_c(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_gearys_c"].compute()
        assert (df["C"] < 1.0).any()

    def test_force_false_skips(self, sj_with_pt_clusters):
        gearys_c(sj_with_pt_clusters)
        gearys_c(sj_with_pt_clusters, force=False)


class TestClusteringCoefficient:
    def test_returns_key(self, sj_with_knn_graph):
        assert clustering_coefficient(sj_with_knn_graph) == "pt_clustering_coeff"

    def test_columns(self, sj_with_knn_graph):
        clustering_coefficient(sj_with_knn_graph)
        df = sj_with_knn_graph.maps["pt_clustering_coeff"].compute()
        assert list(df.columns) == ["cell_id", "clustering_coeff"]

    def test_shape(self, sj_with_knn_graph):
        clustering_coefficient(sj_with_knn_graph)
        df = sj_with_knn_graph.maps["pt_clustering_coeff"].compute()
        assert df.shape[0] == 200

    def test_range(self, sj_with_knn_graph):
        clustering_coefficient(sj_with_knn_graph)
        df = sj_with_knn_graph.maps["pt_clustering_coeff"].compute()
        assert (df["clustering_coeff"] >= 0).all()
        assert (df["clustering_coeff"] <= 1.0).all()

    def test_force_false_skips(self, sj_with_knn_graph):
        clustering_coefficient(sj_with_knn_graph)
        clustering_coefficient(sj_with_knn_graph, force=False)


class TestGetisOrdGi:
    def test_returns_key(self, sj_with_pt_clusters):
        assert getis_ord_gi(sj_with_pt_clusters, feature_key="leiden") == "pt_getis_ord"

    def test_columns(self, sj_with_pt_clusters):
        getis_ord_gi(sj_with_pt_clusters, feature_key="leiden")
        df = sj_with_pt_clusters.maps["pt_getis_ord"].compute()
        assert list(df.columns) == ["cell_id", "gi_stat", "p_value"]

    def test_shape(self, sj_with_pt_clusters):
        getis_ord_gi(sj_with_pt_clusters, feature_key="leiden")
        df = sj_with_pt_clusters.maps["pt_getis_ord"].compute()
        assert df.shape[0] == 200

    def test_p_values_valid(self, sj_with_pt_clusters):
        getis_ord_gi(sj_with_pt_clusters, feature_key="leiden")
        df = sj_with_pt_clusters.maps["pt_getis_ord"].compute()
        assert (df["p_value"] >= 0).all()
        assert (df["p_value"] <= 1.0).all()

    def test_categorical_raises(self, sj_with_pt_clusters):
        """Categorical feature should raise ValueError."""
        # Write a categorical leiden
        cells_df = sj_with_pt_clusters.cells.df.compute()
        cat_df = pd.DataFrame({
            "cell_id": cells_df["cell_id"].values,
            "leiden_cat": pd.Categorical(["A", "B"] * 100),
        })
        from s_spatioloji.compute import _atomic_write_parquet
        maps_dir = sj_with_pt_clusters.config.root / "maps"
        _atomic_write_parquet(cat_df, maps_dir, "leiden_cat")
        with pytest.raises(ValueError, match="numeric feature"):
            getis_ord_gi(sj_with_pt_clusters, feature_key="leiden_cat")

    def test_force_false_skips(self, sj_with_pt_clusters):
        getis_ord_gi(sj_with_pt_clusters, feature_key="leiden")
        getis_ord_gi(sj_with_pt_clusters, feature_key="leiden", force=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_spatial_point_patterns.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `patterns.py`**

```python
"""Spatial pattern analysis functions for point-based spatial data.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.

Uses sparse matrix operations for scalability to 50–100M cells.
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from s_spatioloji.compute import _atomic_write_parquet
from s_spatioloji.spatial.point.graph import _load_point_graph, _load_point_graph_sparse

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def colocalization(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "knn_graph",
    output_key: str = "pt_colocalization",
    force: bool = True,
) -> str:
    """Observed vs expected contact frequency for cluster pairs.

    Args:
        sj: Dataset instance.
        cluster_key: Key for cluster labels in maps/.
        graph_key: Key for the graph edge list.
        output_key: Key to write colocalization table under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> colocalization(sj, cluster_key="leiden")
        'pt_colocalization'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    adj, graph_cell_ids = _load_point_graph_sparse(sj, graph_key, weighted=False)

    cluster_df = sj.maps[cluster_key].compute()
    cell_to_cluster = dict(zip(cluster_df["cell_id"], cluster_df[cluster_key], strict=True))

    # Map graph cell IDs to cluster labels
    id_to_idx = {cid: i for i, cid in enumerate(graph_cell_ids)}
    labels = np.array([str(cell_to_cluster.get(cid, "")) for cid in graph_cell_ids])
    unique_labels = sorted(set(labels) - {""}, key=str)

    # Count edges per cluster pair from sparse matrix
    edge_df = sj.maps[graph_key].compute()
    from collections import defaultdict

    observed: dict = defaultdict(int)
    cluster_sizes: dict = defaultdict(int)

    for label in labels:
        if label:
            cluster_sizes[label] += 1

    n_total = sum(cluster_sizes.values())
    total_edges = len(edge_df)

    for _, row in edge_df.iterrows():
        a_id, b_id = row["cell_id_a"], row["cell_id_b"]
        if a_id in cell_to_cluster and b_id in cell_to_cluster:
            la, lb = str(cell_to_cluster[a_id]), str(cell_to_cluster[b_id])
            pair = tuple(sorted([la, lb]))
            observed[pair] += 1

    records = []
    for a, b in combinations_with_replacement(unique_labels, 2):
        pair = tuple(sorted([str(a), str(b)]))
        obs = observed.get(pair, 0)
        n_a = cluster_sizes[a]
        n_b = cluster_sizes[b]

        if a == b:
            exp = n_a * (n_a - 1) * total_edges / max(n_total * (n_total - 1), 1)
        else:
            exp = 2 * n_a * n_b * total_edges / max(n_total * (n_total - 1), 1)

        ratio = obs / exp if exp > 0 else 0.0
        log2_ratio = np.log2(ratio) if ratio > 0 else float("nan")

        records.append({
            "cluster_a": pair[0],
            "cluster_b": pair[1],
            "observed": obs,
            "expected": exp,
            "ratio": ratio,
            "log2_ratio": log2_ratio,
        })

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def _compute_morans_i_sparse(values: np.ndarray, adj, n: int) -> dict:
    """Compute Moran's I using sparse adjacency matrix.

    Args:
        values: 1-D numeric array (length n).
        adj: Sparse binary adjacency matrix (n x n).
        n: Number of cells.

    Returns:
        Dict with I, expected_I, z_score, p_value.
    """
    if n < 3:
        return {"I": 0.0, "expected_I": 0.0, "z_score": 0.0, "p_value": 1.0}

    x_bar = values.mean()
    z = values - x_bar
    denom = np.sum(z**2)
    if denom == 0:
        return {"I": 0.0, "expected_I": 0.0, "z_score": 0.0, "p_value": 1.0}

    W = float(adj.nnz)  # total weight (binary symmetric, each direction counted)
    if W == 0:
        return {"I": 0.0, "expected_I": 0.0, "z_score": 0.0, "p_value": 1.0}

    # I = (N/W) * z^T W z / z^T z
    Wz = adj @ z
    numerator = z @ Wz
    moran_I = (n / W) * numerator / denom

    expected_I = -1.0 / (n - 1)

    # Randomization assumption variance
    ones = np.ones(n)
    degree = np.asarray(adj @ ones).ravel()
    S1 = float(adj.nnz)  # For binary symmetric: S1 = 0.5 * sum (w_ij + w_ji)^2 = 2 * nnz/2 = nnz
    S2 = float(np.sum((degree + degree) ** 2))  # sum_i (sum_j w_ij + sum_j w_ji)^2
    k = n * np.sum(z**4) / denom**2

    num1 = n * (S1 * (n**2 - 3 * n + 3) - n * S2 + 3 * W**2)
    num2 = k * (S1 * (n**2 - n) - 2 * n * S2 + 6 * W**2)
    var_I = (num1 - num2) / ((n - 1) * (n - 2) * (n - 3) * W**2) - expected_I**2
    var_I = max(var_I, 1e-15)

    z_score = (moran_I - expected_I) / np.sqrt(var_I)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

    return {"I": float(moran_I), "expected_I": float(expected_I), "z_score": float(z_score), "p_value": float(p_value)}


def morans_i(
    sj: s_spatioloji,
    feature_key: str = "leiden",
    graph_key: str = "knn_graph",
    output_key: str = "pt_morans_i",
    force: bool = True,
) -> str:
    """Moran's I spatial autocorrelation using sparse matrix operations.

    For categorical features (dtype object/category), computes one-hot
    indicators and returns one row per category.

    Args:
        sj: Dataset instance.
        feature_key: Key for the feature in maps/.
        graph_key: Key for the graph edge list.
        output_key: Key to write Moran's I results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> morans_i(sj, feature_key="leiden")
        'pt_morans_i'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    adj, graph_cell_ids = _load_point_graph_sparse(sj, graph_key, weighted=False)

    feat_df = sj.maps[feature_key].compute()
    # Align feature values to graph cell order
    feat_map = dict(zip(feat_df["cell_id"], feat_df.iloc[:, 1], strict=True))
    value_cols = [c for c in feat_df.columns if c != "cell_id"]
    if len(value_cols) != 1:
        raise ValueError(f"Expected exactly one feature column besides cell_id, got {len(value_cols)}")

    col = value_cols[0]
    values_series = feat_df.set_index("cell_id")[col]
    aligned = values_series.reindex(graph_cell_ids)

    n = len(graph_cell_ids)
    records = []

    if aligned.dtype == object or hasattr(aligned, "cat"):
        categories = sorted(aligned.dropna().unique(), key=str)
        for cat in categories:
            indicator = (aligned == cat).astype(float).values
            result = _compute_morans_i_sparse(indicator, adj, n)
            result["feature"] = str(cat)
            records.append(result)
    else:
        vals = aligned.values.astype(float)
        result = _compute_morans_i_sparse(vals, adj, n)
        result["feature"] = col
        records.append(result)

    df = pd.DataFrame(records)[["feature", "I", "expected_I", "z_score", "p_value"]]
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def _compute_gearys_c_sparse(values: np.ndarray, adj, edge_df: pd.DataFrame, id_to_idx: dict, n: int) -> dict:
    """Compute Geary's C using edge list for pairwise differences.

    Args:
        values: 1-D numeric array (length n).
        adj: Sparse binary adjacency matrix.
        edge_df: Edge list DataFrame with cell_id_a, cell_id_b.
        id_to_idx: Mapping from cell_id to matrix index.
        n: Number of cells.

    Returns:
        Dict with C, expected_C, z_score, p_value.
    """
    if n < 3:
        return {"C": 1.0, "expected_C": 1.0, "z_score": 0.0, "p_value": 1.0}

    x_bar = values.mean()
    dev = values - x_bar
    denom = np.sum(dev**2)
    if denom == 0:
        return {"C": 1.0, "expected_C": 1.0, "z_score": 0.0, "p_value": 1.0}

    W = float(adj.nnz)
    if W == 0:
        return {"C": 1.0, "expected_C": 1.0, "z_score": 0.0, "p_value": 1.0}

    # Sum (x_i - x_j)^2 over edges — vectorized via edge list
    rows_a = edge_df["cell_id_a"].map(id_to_idx).values
    rows_b = edge_df["cell_id_b"].map(id_to_idx).values
    valid = ~(np.isnan(rows_a) | np.isnan(rows_b))
    rows_a = rows_a[valid].astype(int)
    rows_b = rows_b[valid].astype(int)

    sq_diffs = (values[rows_a] - values[rows_b]) ** 2
    numerator = float(sq_diffs.sum()) * 2  # both directions

    C = ((n - 1) / (2 * W)) * numerator / denom
    expected_C = 1.0

    # Variance
    ones = np.ones(n)
    degree = np.asarray(adj @ ones).ravel()
    S1 = float(adj.nnz)
    S2 = float(np.sum((2 * degree) ** 2))
    k = n * np.sum(dev**4) / denom**2

    num1 = (n - 1) * S1 * (n**2 - 3 * n + 3 - (n - 1) * k)
    num2 = (1 / 4) * (n - 1) * S2 * (n**2 + 3 * n - 6 - (n**2 - n + 2) * k)
    num3 = W**2 * (n**2 - 3 - (n - 1) ** 2 * k)
    var_C = (num1 - num2 + num3) / ((n - 1) * (n - 2) * (n - 3) * W**2)
    var_C = max(var_C, 1e-15)

    z_score = (C - expected_C) / np.sqrt(var_C)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

    return {"C": float(C), "expected_C": expected_C, "z_score": float(z_score), "p_value": float(p_value)}


def gearys_c(
    sj: s_spatioloji,
    feature_key: str = "leiden",
    graph_key: str = "knn_graph",
    output_key: str = "pt_gearys_c",
    force: bool = True,
) -> str:
    """Geary's C spatial autocorrelation.

    Same interface and categorical handling as :func:`morans_i`.

    Args:
        sj: Dataset instance.
        feature_key: Key for the feature in maps/.
        graph_key: Key for the graph edge list.
        output_key: Key to write Geary's C results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> gearys_c(sj, feature_key="leiden")
        'pt_gearys_c'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    adj, graph_cell_ids = _load_point_graph_sparse(sj, graph_key, weighted=False)
    id_to_idx = {cid: i for i, cid in enumerate(graph_cell_ids)}
    n = len(graph_cell_ids)

    edge_df = sj.maps[graph_key].compute()

    feat_df = sj.maps[feature_key].compute()
    value_cols = [c for c in feat_df.columns if c != "cell_id"]
    if len(value_cols) != 1:
        raise ValueError(f"Expected exactly one feature column besides cell_id, got {len(value_cols)}")

    col = value_cols[0]
    values_series = feat_df.set_index("cell_id")[col]
    aligned = values_series.reindex(graph_cell_ids)

    records = []
    if aligned.dtype == object or hasattr(aligned, "cat"):
        categories = sorted(aligned.dropna().unique(), key=str)
        for cat in categories:
            indicator = (aligned == cat).astype(float).values
            result = _compute_gearys_c_sparse(indicator, adj, edge_df, id_to_idx, n)
            result["feature"] = str(cat)
            records.append(result)
    else:
        vals = aligned.values.astype(float)
        result = _compute_gearys_c_sparse(vals, adj, edge_df, id_to_idx, n)
        result["feature"] = col
        records.append(result)

    df = pd.DataFrame(records)[["feature", "C", "expected_C", "z_score", "p_value"]]
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def clustering_coefficient(
    sj: s_spatioloji,
    graph_key: str = "knn_graph",
    output_key: str = "pt_clustering_coeff",
    force: bool = True,
) -> str:
    """Per-cell local clustering coefficient.

    Requires networkx. Not scalable beyond ~5M cells.

    Args:
        sj: Dataset instance.
        graph_key: Key for the graph edge list.
        output_key: Key to write clustering coefficients under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Raises:
        ValueError: If dataset has more than 5 million cells.

    Example:
        >>> clustering_coefficient(sj)
        'pt_clustering_coeff'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    n_cells = sj.cells.n_cells
    if n_cells > 5_000_000:
        raise ValueError(
            "clustering_coefficient requires networkx and is not scalable beyond ~5M cells. "
            "Consider using morans_i or getis_ord_gi for large datasets."
        )

    maps_dir.mkdir(exist_ok=True)
    G = _load_point_graph(sj, graph_key)

    cells_df = sj.cells.df.compute()
    all_cell_ids = list(cells_df["cell_id"])

    cc = nx.clustering(G)
    records = [{"cell_id": cid, "clustering_coeff": cc.get(cid, 0.0)} for cid in all_cell_ids]

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def getis_ord_gi(
    sj: s_spatioloji,
    feature_key: str,
    graph_key: str = "knn_graph",
    star: bool = True,
    output_key: str = "pt_getis_ord",
    force: bool = True,
) -> str:
    """Getis-Ord Gi* (or Gi) statistic per cell.

    Args:
        sj: Dataset instance.
        feature_key: Key for a numeric feature in maps/ (required, no default).
        graph_key: Key for the graph edge list.
        star: If ``True``, compute Gi* (includes self). If ``False``, Gi.
        output_key: Key to write results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Raises:
        ValueError: If feature is categorical.

    Example:
        >>> getis_ord_gi(sj, feature_key="leiden")
        'pt_getis_ord'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    adj, graph_cell_ids = _load_point_graph_sparse(sj, graph_key, weighted=False)

    feat_df = sj.maps[feature_key].compute()
    value_cols = [c for c in feat_df.columns if c != "cell_id"]
    if len(value_cols) != 1:
        raise ValueError(f"Expected exactly one feature column besides cell_id, got {len(value_cols)}")

    col = value_cols[0]
    values_series = feat_df.set_index("cell_id")[col]
    aligned = values_series.reindex(graph_cell_ids)

    if aligned.dtype == object or hasattr(aligned, "cat"):
        raise ValueError("Getis-Ord Gi* requires a numeric feature.")

    import scipy.sparse

    x = aligned.values.astype(np.float64)
    n = len(x)

    if star:
        adj_work = adj + scipy.sparse.eye(n, format="csr")
    else:
        adj_work = adj.copy()

    x_bar = x.mean()
    S = x.std()

    if S == 0:
        df = pd.DataFrame({
            "cell_id": graph_cell_ids,
            "gi_stat": np.zeros(n),
            "p_value": np.ones(n),
        })
        _atomic_write_parquet(df, maps_dir, output_key)
        return output_key

    # Wi = sum_j w_ij per cell
    ones = np.ones(n)
    Wi = np.asarray(adj_work @ ones).ravel()

    # sum_j w_ij * x_j
    Wx = np.asarray(adj_work @ x).ravel()

    # Wi_sq = sum_j w_ij^2 (binary weights, so = Wi)
    Wi_sq = Wi

    numerator = Wx - x_bar * Wi
    denominator = S * np.sqrt((n * Wi_sq - Wi**2) / (n - 1))
    denominator[denominator == 0] = 1e-15

    gi_stat = numerator / denominator
    p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(gi_stat)))

    df = pd.DataFrame({
        "cell_id": graph_cell_ids,
        "gi_stat": gi_stat,
        "p_value": p_value,
    })
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_spatial_point_patterns.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/s_spatioloji/spatial/point/patterns.py tests/unit/test_spatial_point_patterns.py
git commit -m "feat(spatial/point): add colocalization, Moran's I, Geary's C, clustering coeff, Getis-Ord Gi*"
```

---

## Chunk 4: Ripley's Functions (Task 5)

### Task 5: ripley.py

**Files:**
- Create: `tests/unit/test_spatial_point_ripley.py`
- Create: `src/s_spatioloji/spatial/point/ripley.py`

- [ ] **Step 1: Write Ripley tests**

Create `tests/unit/test_spatial_point_ripley.py`:

```python
"""Unit tests for s_spatioloji.spatial.point.ripley."""

from __future__ import annotations

import numpy as np

from s_spatioloji.spatial.point.ripley import ripley_f, ripley_g, ripley_k, ripley_l


class TestRipleyK:
    def test_returns_key(self, sj):
        assert ripley_k(sj) == "pt_ripley_k"

    def test_output_written(self, sj):
        ripley_k(sj)
        assert sj.maps.has("pt_ripley_k")

    def test_columns(self, sj):
        ripley_k(sj)
        df = sj.maps["pt_ripley_k"].compute()
        assert list(df.columns) == ["cluster", "r", "K", "K_theo"]

    def test_cluster_all_when_no_key(self, sj):
        ripley_k(sj)
        df = sj.maps["pt_ripley_k"].compute()
        assert (df["cluster"] == "all").all()

    def test_n_radii(self, sj):
        ripley_k(sj, n_radii=20)
        df = sj.maps["pt_ripley_k"].compute()
        assert len(df) == 20

    def test_K_theo_is_pi_r_sq(self, sj):
        ripley_k(sj, n_radii=10)
        df = sj.maps["pt_ripley_k"].compute()
        np.testing.assert_allclose(df["K_theo"].values, np.pi * df["r"].values ** 2)

    def test_force_false_skips(self, sj):
        ripley_k(sj)
        ripley_k(sj, force=False)

    def test_per_cluster(self, sj_with_pt_clusters):
        ripley_k(sj_with_pt_clusters, cluster_key="leiden", n_radii=10)
        df = sj_with_pt_clusters.maps["pt_ripley_k"].compute()
        assert len(df["cluster"].unique()) == 4  # 4 clusters


class TestRipleyL:
    def test_returns_key(self, sj):
        assert ripley_l(sj) == "pt_ripley_l"

    def test_columns(self, sj):
        ripley_l(sj)
        df = sj.maps["pt_ripley_l"].compute()
        assert list(df.columns) == ["cluster", "r", "L"]

    def test_force_false_skips(self, sj):
        ripley_l(sj)
        ripley_l(sj, force=False)


class TestRipleyG:
    def test_returns_key(self, sj):
        assert ripley_g(sj) == "pt_ripley_g"

    def test_columns(self, sj):
        ripley_g(sj)
        df = sj.maps["pt_ripley_g"].compute()
        assert list(df.columns) == ["cluster", "r", "G", "G_theo"]

    def test_G_range(self, sj):
        ripley_g(sj)
        df = sj.maps["pt_ripley_g"].compute()
        assert (df["G"] >= 0).all()
        assert (df["G"] <= 1.0).all()

    def test_force_false_skips(self, sj):
        ripley_g(sj)
        ripley_g(sj, force=False)


class TestRipleyF:
    def test_returns_key(self, sj):
        assert ripley_f(sj) == "pt_ripley_f"

    def test_columns(self, sj):
        ripley_f(sj)
        df = sj.maps["pt_ripley_f"].compute()
        assert list(df.columns) == ["cluster", "r", "F", "F_theo"]

    def test_F_range(self, sj):
        ripley_f(sj)
        df = sj.maps["pt_ripley_f"].compute()
        assert (df["F"] >= 0).all()
        assert (df["F"] <= 1.0).all()

    def test_force_false_skips(self, sj):
        ripley_f(sj)
        ripley_f(sj, force=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_spatial_point_ripley.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `ripley.py`**

```python
"""Ripley's spatial statistics (K, L, G, F) from cell centroids.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.

Uses ``scipy.spatial.cKDTree`` for O(N log N) distance queries.
No graph dependency — purely coordinate-based.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from s_spatioloji.compute import _atomic_write_parquet

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def _get_coords_and_groups(
    sj: s_spatioloji, cluster_key: str | None,
) -> list[tuple[str, np.ndarray]]:
    """Extract coordinates, optionally grouped by cluster.

    Returns list of (group_name, coords_array) tuples.
    """
    cells_df = sj.cells.df.compute()
    coords = cells_df[["x", "y"]].values.astype(np.float64)

    if cluster_key is None:
        return [("all", coords)]

    cluster_df = sj.maps[cluster_key].compute()
    merged = cells_df[["cell_id", "x", "y"]].merge(cluster_df, on="cell_id")
    col = [c for c in cluster_df.columns if c != "cell_id"][0]
    groups = []
    for label in sorted(merged[col].unique(), key=str):
        mask = merged[col] == label
        group_coords = merged.loc[mask, ["x", "y"]].values.astype(np.float64)
        groups.append((str(label), group_coords))
    return groups


def _auto_radii(coords: np.ndarray, n_radii: int, max_radius: float | None) -> np.ndarray:
    """Generate evenly spaced radii up to max_radius."""
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    if max_radius is None:
        max_radius = min(x_range, y_range) / 4.0
    return np.linspace(max_radius / n_radii, max_radius, n_radii)


def _study_area(coords: np.ndarray) -> float:
    """Bounding box area."""
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    return float(max(x_range * y_range, 1e-10))


def ripley_k(
    sj: s_spatioloji,
    radii: np.ndarray | None = None,
    n_radii: int = 50,
    max_radius: float | None = None,
    cluster_key: str | None = None,
    output_key: str = "pt_ripley_k",
    force: bool = True,
) -> str:
    """Ripley's K function.

    ``K(r) = (A / (N*(N-1))) * sum_i sum_{j!=i} w_ij * 1(d_ij <= r)``

    Args:
        sj: Dataset instance.
        radii: Explicit evaluation distances. If ``None``, auto-generates.
        n_radii: Number of radii to auto-generate.
        max_radius: Upper bound. Defaults to 1/4 of shorter bounding box side.
        cluster_key: If set, compute per cluster. If ``None``, global.
        output_key: Key to write results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> ripley_k(sj, n_radii=50)
        'pt_ripley_k'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    groups = _get_coords_and_groups(sj, cluster_key)

    records = []
    for group_name, coords in groups:
        N = len(coords)
        if N < 2:
            continue

        r_vals = radii if radii is not None else _auto_radii(coords, n_radii, max_radius)
        A = _study_area(coords)
        tree = cKDTree(coords)

        for r in r_vals:
            # count_neighbors counts all pairs (i,j) where d(i,j) <= r, including i==j
            count = tree.count_neighbors(tree, r) - N  # subtract self-pairs
            K_val = A / (N * (N - 1)) * count
            K_theo = np.pi * r**2
            records.append({
                "cluster": group_name,
                "r": float(r),
                "K": float(K_val),
                "K_theo": float(K_theo),
            })

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def ripley_l(
    sj: s_spatioloji,
    radii: np.ndarray | None = None,
    n_radii: int = 50,
    max_radius: float | None = None,
    cluster_key: str | None = None,
    output_key: str = "pt_ripley_l",
    force: bool = True,
) -> str:
    """Ripley's L function (variance-stabilized K).

    ``L(r) = sqrt(K(r) / pi) - r``. Under CSR, ``L(r) = 0``.

    Args:
        sj: Dataset instance.
        radii: Explicit evaluation distances.
        n_radii: Number of radii to auto-generate.
        max_radius: Upper bound.
        cluster_key: If set, compute per cluster.
        output_key: Key to write results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> ripley_l(sj)
        'pt_ripley_l'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    groups = _get_coords_and_groups(sj, cluster_key)

    records = []
    for group_name, coords in groups:
        N = len(coords)
        if N < 2:
            continue

        r_vals = radii if radii is not None else _auto_radii(coords, n_radii, max_radius)
        A = _study_area(coords)
        tree = cKDTree(coords)

        for r in r_vals:
            count = tree.count_neighbors(tree, r) - N
            K_val = A / (N * (N - 1)) * count
            L_val = np.sqrt(max(K_val, 0) / np.pi) - r
            records.append({"cluster": group_name, "r": float(r), "L": float(L_val)})

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def ripley_g(
    sj: s_spatioloji,
    radii: np.ndarray | None = None,
    n_radii: int = 50,
    max_radius: float | None = None,
    cluster_key: str | None = None,
    output_key: str = "pt_ripley_g",
    force: bool = True,
) -> str:
    """Ripley's G function (nearest-neighbor distance distribution).

    ``G(r) = fraction of points whose NN distance <= r``.

    Args:
        sj: Dataset instance.
        radii: Explicit evaluation distances.
        n_radii: Number of radii to auto-generate.
        max_radius: Upper bound.
        cluster_key: If set, compute per cluster.
        output_key: Key to write results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> ripley_g(sj)
        'pt_ripley_g'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    groups = _get_coords_and_groups(sj, cluster_key)

    records = []
    for group_name, coords in groups:
        N = len(coords)
        if N < 2:
            continue

        r_vals = radii if radii is not None else _auto_radii(coords, n_radii, max_radius)
        A = _study_area(coords)
        lam = N / A  # intensity

        tree = cKDTree(coords)
        nn_dists, _ = tree.query(coords, k=2)  # k=2: [self=0, true NN]
        nn_dists = nn_dists[:, 1]  # true NN distances

        for r in r_vals:
            G_val = float(np.mean(nn_dists <= r))
            G_theo = 1.0 - np.exp(-lam * np.pi * r**2)
            records.append({
                "cluster": group_name,
                "r": float(r),
                "G": G_val,
                "G_theo": float(G_theo),
            })

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def ripley_f(
    sj: s_spatioloji,
    radii: np.ndarray | None = None,
    n_radii: int = 50,
    max_radius: float | None = None,
    cluster_key: str | None = None,
    n_random: int = 1000,
    output_key: str = "pt_ripley_f",
    force: bool = True,
) -> str:
    """Ripley's F function (empty-space distance distribution).

    ``F(r) = fraction of random reference points whose nearest data point <= r``.

    Args:
        sj: Dataset instance.
        radii: Explicit evaluation distances.
        n_radii: Number of radii to auto-generate.
        max_radius: Upper bound.
        cluster_key: If set, compute per cluster.
        n_random: Number of random reference points.
        output_key: Key to write results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> ripley_f(sj)
        'pt_ripley_f'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    groups = _get_coords_and_groups(sj, cluster_key)

    rng = np.random.default_rng(42)
    records = []

    for group_name, coords in groups:
        N = len(coords)
        if N < 2:
            continue

        r_vals = radii if radii is not None else _auto_radii(coords, n_radii, max_radius)
        A = _study_area(coords)
        lam = N / A

        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        random_pts = np.column_stack([
            rng.uniform(x_min, x_max, n_random),
            rng.uniform(y_min, y_max, n_random),
        ])

        tree = cKDTree(coords)
        nn_dists, _ = tree.query(random_pts, k=1)

        for r in r_vals:
            F_val = float(np.mean(nn_dists <= r))
            F_theo = 1.0 - np.exp(-lam * np.pi * r**2)
            records.append({
                "cluster": group_name,
                "r": float(r),
                "F": F_val,
                "F_theo": float(F_theo),
            })

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_spatial_point_ripley.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/s_spatioloji/spatial/point/ripley.py tests/unit/test_spatial_point_ripley.py
git commit -m "feat(spatial/point): add Ripley's K, L, G, F functions"
```

---

## Chunk 5: Statistics (Task 6)

### Task 6: statistics.py

**Files:**
- Create: `tests/unit/test_spatial_point_statistics.py`
- Create: `src/s_spatioloji/spatial/point/statistics.py`

- [ ] **Step 1: Write statistics tests**

Create `tests/unit/test_spatial_point_statistics.py`:

```python
"""Unit tests for s_spatioloji.spatial.point.statistics."""

from __future__ import annotations

import numpy as np

from s_spatioloji.spatial.point.statistics import (
    clark_evans,
    dclf_envelope,
    permutation_test,
    quadrat_density,
)


class TestPermutationTest:
    def test_returns_key(self, sj_with_pt_clusters):
        assert permutation_test(sj_with_pt_clusters, n_permutations=50) == "pt_permutation_test"

    def test_output_written(self, sj_with_pt_clusters):
        permutation_test(sj_with_pt_clusters, n_permutations=50)
        assert sj_with_pt_clusters.maps.has("pt_permutation_test")

    def test_columns(self, sj_with_pt_clusters):
        permutation_test(sj_with_pt_clusters, n_permutations=50)
        df = sj_with_pt_clusters.maps["pt_permutation_test"].compute()
        assert list(df.columns) == ["cluster_a", "cluster_b", "observed_ratio", "p_value", "z_score"]

    def test_p_values_valid(self, sj_with_pt_clusters):
        permutation_test(sj_with_pt_clusters, n_permutations=50)
        df = sj_with_pt_clusters.maps["pt_permutation_test"].compute()
        assert (df["p_value"] >= 0).all()
        assert (df["p_value"] <= 1.0).all()

    def test_force_false_skips(self, sj_with_pt_clusters):
        permutation_test(sj_with_pt_clusters, n_permutations=50)
        permutation_test(sj_with_pt_clusters, n_permutations=50, force=False)


class TestQuadratDensity:
    def test_returns_key(self, sj_with_pt_clusters):
        assert quadrat_density(sj_with_pt_clusters) == "pt_quadrat_density"

    def test_columns(self, sj_with_pt_clusters):
        quadrat_density(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_quadrat_density"].compute()
        assert list(df.columns) == ["cluster", "chi2", "p_value", "density_mean", "density_std"]

    def test_has_rows(self, sj_with_pt_clusters):
        quadrat_density(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_quadrat_density"].compute()
        assert len(df) == 4  # 4 clusters

    def test_force_false_skips(self, sj_with_pt_clusters):
        quadrat_density(sj_with_pt_clusters)
        quadrat_density(sj_with_pt_clusters, force=False)


class TestClarkEvans:
    def test_returns_key(self, sj):
        assert clark_evans(sj) == "pt_clark_evans"

    def test_columns(self, sj):
        clark_evans(sj)
        df = sj.maps["pt_clark_evans"].compute()
        assert list(df.columns) == ["cluster", "R", "r_observed", "r_expected", "z_score", "p_value"]

    def test_global_cluster_all(self, sj):
        clark_evans(sj)
        df = sj.maps["pt_clark_evans"].compute()
        assert len(df) == 1
        assert df["cluster"].iloc[0] == "all"

    def test_R_positive(self, sj):
        clark_evans(sj)
        df = sj.maps["pt_clark_evans"].compute()
        assert (df["R"] > 0).all()

    def test_regular_grid_R_near_one(self, sj):
        """20x10 grid should have R >= 1 (regular/dispersed)."""
        clark_evans(sj)
        df = sj.maps["pt_clark_evans"].compute()
        assert df["R"].iloc[0] >= 0.9  # approximately regular

    def test_per_cluster(self, sj_with_pt_clusters):
        clark_evans(sj_with_pt_clusters, cluster_key="leiden")
        df = sj_with_pt_clusters.maps["pt_clark_evans"].compute()
        assert len(df) == 4

    def test_force_false_skips(self, sj):
        clark_evans(sj)
        clark_evans(sj, force=False)


class TestDCLFEnvelope:
    def test_returns_key(self, sj):
        assert dclf_envelope(sj, n_simulations=19) == "pt_dclf_envelope"

    def test_columns(self, sj):
        dclf_envelope(sj, n_simulations=19, n_radii=10)
        df = sj.maps["pt_dclf_envelope"].compute()
        assert list(df.columns) == ["cluster", "r", "observed", "lo", "hi", "theo", "p_value"]

    def test_p_value_valid(self, sj):
        dclf_envelope(sj, n_simulations=19, n_radii=10)
        df = sj.maps["pt_dclf_envelope"].compute()
        assert (df["p_value"] >= 0).all()
        assert (df["p_value"] <= 1.0).all()

    def test_envelope_bounds(self, sj):
        dclf_envelope(sj, n_simulations=19, n_radii=10)
        df = sj.maps["pt_dclf_envelope"].compute()
        assert (df["lo"] <= df["hi"]).all()

    def test_function_L(self, sj):
        dclf_envelope(sj, function="L", n_simulations=19, n_radii=10, output_key="pt_dclf_L")
        assert sj.maps.has("pt_dclf_L")

    def test_force_false_skips(self, sj):
        dclf_envelope(sj, n_simulations=19, n_radii=10)
        dclf_envelope(sj, n_simulations=19, n_radii=10, force=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_spatial_point_statistics.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `statistics.py`**

```python
"""Statistical testing functions for point-based spatial data.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

from s_spatioloji.compute import _atomic_write_parquet
from s_spatioloji.spatial.point.graph import _load_point_graph_sparse

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def permutation_test(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "knn_graph",
    n_permutations: int = 1000,
    random_state: int = 42,
    output_key: str = "pt_permutation_test",
    force: bool = True,
) -> str:
    """Permutation test for spatial colocalization significance.

    Permutes cluster labels, recomputes colocalization observed/expected
    ratios, and compares against the null distribution.

    Args:
        sj: Dataset instance.
        cluster_key: Key for cluster labels in maps/.
        graph_key: Key for the graph edge list.
        n_permutations: Number of label permutations.
        random_state: Seed for reproducibility.
        output_key: Key to write test results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> permutation_test(sj, n_permutations=1000)
        'pt_permutation_test'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)

    # Load edge list and cluster labels
    edge_df = sj.maps[graph_key].compute()
    cluster_df = sj.maps[cluster_key].compute()
    cell_ids = list(cluster_df["cell_id"])
    labels = list(cluster_df[cluster_key])
    cell_to_cluster = dict(zip(cell_ids, labels, strict=True))

    unique_labels = sorted(set(labels), key=str)
    pairs = list(combinations_with_replacement(unique_labels, 2))
    pair_keys = [tuple(sorted([str(a), str(b)])) for a, b in pairs]

    # Pre-filter edges to cells with labels
    edge_pairs = []
    for _, row in edge_df.iterrows():
        a, b = row["cell_id_a"], row["cell_id_b"]
        if a in cell_to_cluster and b in cell_to_cluster:
            edge_pairs.append((a, b))

    def _compute_ratios(c2c):
        counts = defaultdict(int)
        cluster_sizes = defaultdict(int)
        for cid, lab in c2c.items():
            cluster_sizes[lab] += 1
        n_total = sum(cluster_sizes.values())
        total_edges = len(edge_pairs)

        for u, v in edge_pairs:
            la, lb = c2c[u], c2c[v]
            pair = tuple(sorted([str(la), str(lb)]))
            counts[pair] += 1

        ratios = {}
        for pk in pair_keys:
            obs = counts.get(pk, 0)
            a_lab, b_lab = pk
            # Find matching label (str back to original)
            n_a = sum(1 for l in c2c.values() if str(l) == a_lab)
            n_b = sum(1 for l in c2c.values() if str(l) == b_lab)
            if a_lab == b_lab:
                exp = n_a * (n_a - 1) * total_edges / max(n_total * (n_total - 1), 1)
            else:
                exp = 2 * n_a * n_b * total_edges / max(n_total * (n_total - 1), 1)
            ratios[pk] = obs / exp if exp > 0 else 0.0
        return ratios

    observed_ratios = _compute_ratios(cell_to_cluster)

    rng = np.random.default_rng(random_state)
    perm_ratios = {pk: [] for pk in pair_keys}

    for _ in range(n_permutations):
        shuffled = rng.permutation(labels)
        perm_c2c = dict(zip(cell_ids, shuffled, strict=True))
        ratios = _compute_ratios(perm_c2c)
        for pk in pair_keys:
            perm_ratios[pk].append(ratios.get(pk, 0.0))

    records = []
    for pk in pair_keys:
        obs_ratio = observed_ratios.get(pk, 0.0)
        null_dist = np.array(perm_ratios[pk])
        null_mean = null_dist.mean()
        null_std = null_dist.std()

        z = (obs_ratio - null_mean) / null_std if null_std > 0 else 0.0
        p = (np.sum(null_dist >= obs_ratio) + 1) / (n_permutations + 1)

        records.append({
            "cluster_a": pk[0],
            "cluster_b": pk[1],
            "observed_ratio": float(obs_ratio),
            "p_value": float(p),
            "z_score": float(z),
        })

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def quadrat_density(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    n_bins: int = 10,
    output_key: str = "pt_quadrat_density",
    force: bool = True,
) -> str:
    """Quadrat-based density analysis with chi-squared test.

    Divides spatial extent into ``n_bins x n_bins`` grid quadrats.
    Does not depend on graph.

    Args:
        sj: Dataset instance.
        cluster_key: Key for cluster labels in maps/.
        n_bins: Number of bins per dimension.
        output_key: Key to write density results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> quadrat_density(sj, n_bins=10)
        'pt_quadrat_density'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)

    cluster_df = sj.maps[cluster_key].compute()
    cells_df = sj.cells.df.compute()
    merged = cells_df[["cell_id", "x", "y"]].merge(cluster_df[["cell_id", cluster_key]], on="cell_id")

    x = merged["x"].values
    y = merged["y"].values
    labels = merged[cluster_key].values

    x_edges = np.linspace(x.min() - 1e-6, x.max() + 1e-6, n_bins + 1)
    y_edges = np.linspace(y.min() - 1e-6, y.max() + 1e-6, n_bins + 1)

    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, n_bins - 1)
    y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, n_bins - 1)

    unique_labels = sorted(set(labels), key=str)
    records = []

    for label in unique_labels:
        mask = labels == label
        counts = np.zeros((n_bins, n_bins))
        for xi, yi in zip(x_bin[mask], y_bin[mask], strict=True):
            counts[xi, yi] += 1

        flat = counts.flatten()
        expected = np.full_like(flat, flat.mean())

        if expected[0] > 0:
            chi2, p = stats.chisquare(flat, f_exp=expected)
        else:
            chi2, p = 0.0, 1.0

        records.append({
            "cluster": label,
            "chi2": float(chi2),
            "p_value": float(p),
            "density_mean": float(flat.mean()),
            "density_std": float(flat.std()),
        })

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def clark_evans(
    sj: s_spatioloji,
    cluster_key: str | None = None,
    output_key: str = "pt_clark_evans",
    force: bool = True,
) -> str:
    """Clark-Evans nearest-neighbor index.

    ``R = r_obs / r_exp`` where ``R < 1`` = clustered, ``R = 1`` = random,
    ``R > 1`` = dispersed.

    Args:
        sj: Dataset instance.
        cluster_key: If set, compute per cluster. If ``None``, global.
        output_key: Key to write results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> clark_evans(sj)
        'pt_clark_evans'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    cells_df = sj.cells.df.compute()

    if cluster_key is None:
        groups = [("all", cells_df)]
    else:
        cluster_df = sj.maps[cluster_key].compute()
        merged = cells_df[["cell_id", "x", "y"]].merge(cluster_df, on="cell_id")
        col = [c for c in cluster_df.columns if c != "cell_id"][0]
        groups = [(str(label), merged[merged[col] == label]) for label in sorted(merged[col].unique(), key=str)]

    records = []
    for group_name, group_df in groups:
        coords = group_df[["x", "y"]].values.astype(np.float64)
        N = len(coords)
        if N < 2:
            records.append({
                "cluster": group_name, "R": 0.0, "r_observed": 0.0,
                "r_expected": 0.0, "z_score": 0.0, "p_value": 1.0,
            })
            continue

        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        A = max(x_range * y_range, 1e-10)
        lam = N / A

        tree = cKDTree(coords)
        nn_dists, _ = tree.query(coords, k=2)
        r_obs = float(nn_dists[:, 1].mean())

        r_exp = 1.0 / (2.0 * np.sqrt(lam))
        sigma_r = 0.26136 / np.sqrt(N * lam)

        R = r_obs / r_exp if r_exp > 0 else 0.0
        z = (r_obs - r_exp) / sigma_r if sigma_r > 0 else 0.0
        p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

        records.append({
            "cluster": group_name,
            "R": float(R),
            "r_observed": r_obs,
            "r_expected": float(r_exp),
            "z_score": float(z),
            "p_value": float(p),
        })

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def dclf_envelope(
    sj: s_spatioloji,
    function: str = "K",
    cluster_key: str | None = None,
    n_simulations: int = 199,
    radii: np.ndarray | None = None,
    n_radii: int = 50,
    max_radius: float | None = None,
    max_cells: int = 100_000,
    random_state: int = 42,
    output_key: str = "pt_dclf_envelope",
    force: bool = True,
) -> str:
    """DCLF global envelope test against CSR.

    Generates CSR point patterns and builds pointwise min/max envelope.

    Args:
        sj: Dataset instance.
        function: One of ``"K"``, ``"L"``, ``"G"``, ``"F"``.
        cluster_key: If set, compute per cluster.
        n_simulations: Number of CSR simulations.
        radii: Explicit evaluation distances.
        n_radii: Number of radii to auto-generate.
        max_radius: Upper bound for radii.
        max_cells: Subsampling threshold.
        random_state: Seed for reproducibility.
        output_key: Key to write results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> dclf_envelope(sj, function="K", n_simulations=199)
        'pt_dclf_envelope'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(random_state)

    from s_spatioloji.spatial.point.ripley import _auto_radii, _get_coords_and_groups, _study_area

    groups = _get_coords_and_groups(sj, cluster_key)

    records = []
    for group_name, coords in groups:
        N = len(coords)
        if N < 2:
            continue

        # Subsample if needed
        if N > max_cells:
            idx = rng.choice(N, max_cells, replace=False)
            coords = coords[idx]
            N = max_cells

        r_vals = radii if radii is not None else _auto_radii(coords, n_radii, max_radius)
        A = _study_area(coords)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        def _compute_function(pts, r_vals):
            """Compute the chosen Ripley function for a point pattern."""
            tree = cKDTree(pts)
            n = len(pts)
            area = max((pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min()), 1e-10)
            lam = n / area

            result = np.zeros(len(r_vals))
            if function == "K":
                for i, r in enumerate(r_vals):
                    count = tree.count_neighbors(tree, r) - n
                    result[i] = area / (n * (n - 1)) * count
            elif function == "L":
                for i, r in enumerate(r_vals):
                    count = tree.count_neighbors(tree, r) - n
                    K = area / (n * (n - 1)) * count
                    result[i] = np.sqrt(max(K, 0) / np.pi) - r
            elif function == "G":
                nn_dists, _ = tree.query(pts, k=2)
                nn_dists = nn_dists[:, 1]
                for i, r in enumerate(r_vals):
                    result[i] = np.mean(nn_dists <= r)
            elif function == "F":
                random_pts = np.column_stack([
                    rng.uniform(pts[:, 0].min(), pts[:, 0].max(), min(1000, n)),
                    rng.uniform(pts[:, 1].min(), pts[:, 1].max(), min(1000, n)),
                ])
                nn_dists, _ = tree.query(random_pts, k=1)
                for i, r in enumerate(r_vals):
                    result[i] = np.mean(nn_dists <= r)
            return result

        # Observed
        observed = _compute_function(coords, r_vals)

        # Theoretical CSR expectation
        lam = N / A
        if function == "K":
            theo = np.pi * r_vals**2
        elif function == "L":
            theo = np.zeros(len(r_vals))  # L = 0 under CSR
        elif function in ("G", "F"):
            theo = 1.0 - np.exp(-lam * np.pi * r_vals**2)
        else:
            raise ValueError(f"Unknown function: {function!r}. Must be one of 'K', 'L', 'G', 'F'.")

        # Simulations
        sim_results = np.zeros((n_simulations, len(r_vals)))
        for s in range(n_simulations):
            sim_pts = np.column_stack([
                rng.uniform(x_min, x_max, N),
                rng.uniform(y_min, y_max, N),
            ])
            sim_results[s] = _compute_function(sim_pts, r_vals)

        lo = sim_results.min(axis=0)
        hi = sim_results.max(axis=0)

        # DCLF p-value: rank of observed max absolute deviation
        obs_max_dev = np.max(np.abs(observed - theo))
        sim_max_devs = np.max(np.abs(sim_results - theo[np.newaxis, :]), axis=1)
        p_value = float((np.sum(sim_max_devs >= obs_max_dev) + 1) / (n_simulations + 1))

        for i, r in enumerate(r_vals):
            records.append({
                "cluster": group_name,
                "r": float(r),
                "observed": float(observed[i]),
                "lo": float(lo[i]),
                "hi": float(hi[i]),
                "theo": float(theo[i]),
                "p_value": p_value,
            })

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_spatial_point_statistics.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/unit/ -v`
Expected: ALL PASS (existing polygon tests unaffected)

- [ ] **Step 6: Lint and format**

Run: `ruff check src/s_spatioloji/spatial/point/ tests/unit/test_spatial_point_*.py --fix && ruff format src/s_spatioloji/spatial/point/ tests/unit/test_spatial_point_*.py`

- [ ] **Step 7: Commit**

```bash
git add src/s_spatioloji/spatial/point/statistics.py tests/unit/test_spatial_point_statistics.py
git commit -m "feat(spatial/point): add permutation test, quadrat density, Clark-Evans, DCLF envelope"
```
