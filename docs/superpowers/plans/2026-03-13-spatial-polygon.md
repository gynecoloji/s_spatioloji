# Spatial Polygon Module Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement polygon-based spatial analysis (contact graph, morphology, neighborhoods, patterns, statistics) for the s_spatioloji spatial transcriptomics package.

**Architecture:** All 12 public functions (+ 1 internal helper) follow the compute layer contract — accept `s_spatioloji`, write Parquet to `maps/`, return key string. Contact graph is the foundational structure; morphology and quadrat density are independent. All writes use `_atomic_write_parquet` from `s_spatioloji.compute`.

**Tech Stack:** shapely (STRtree, geometry ops), networkx (graph algorithms), scipy (chi-squared), scikit-learn (NearestNeighbors not needed — graph is geometry-based), opencv-python (fitEllipse for eccentricity), numpy, pandas, geopandas.

**Spec:** `docs/superpowers/specs/2026-03-13-spatial-polygon-design.md`

---

## File Structure

| Action | Path | Responsibility |
|--------|------|---------------|
| Create | `src/s_spatioloji/spatial/__init__.py` | Empty package init |
| Create | `src/s_spatioloji/spatial/polygon/__init__.py` | Re-exports all 12 public functions |
| Create | `src/s_spatioloji/spatial/polygon/graph.py` | `build_contact_graph`, `_load_contact_graph` |
| Create | `src/s_spatioloji/spatial/polygon/morphology.py` | `cell_morphology` (13 metrics) |
| Create | `src/s_spatioloji/spatial/polygon/neighborhoods.py` | `neighborhood_composition`, `nth_order_neighbors`, `neighborhood_diversity` |
| Create | `src/s_spatioloji/spatial/polygon/patterns.py` | `colocalization`, `morans_i`, `gearys_c`, `clustering_coefficient`, `border_enrichment` |
| Create | `src/s_spatioloji/spatial/polygon/statistics.py` | `permutation_test`, `quadrat_density` |
| Modify | `tests/unit/conftest.py` | Add `sj_with_boundaries` and `sj_with_graph` fixtures |
| Create | `tests/unit/test_spatial_polygon_graph.py` | Tests for graph.py |
| Create | `tests/unit/test_spatial_polygon_morphology.py` | Tests for morphology.py |
| Create | `tests/unit/test_spatial_polygon_neighborhoods.py` | Tests for neighborhoods.py |
| Create | `tests/unit/test_spatial_polygon_patterns.py` | Tests for patterns.py |
| Create | `tests/unit/test_spatial_polygon_statistics.py` | Tests for statistics.py |

**Test command:** `PYTHONPATH=src /c/Users/YJ071/anaconda3/python.exe -m pytest tests/unit/ -v`
**Lint command:** `/c/Users/YJ071/anaconda3/envs/spatioloji/Scripts/ruff.exe check src/s_spatioloji/spatial/ tests/unit/test_spatial_polygon_*.py --fix && /c/Users/YJ071/anaconda3/envs/spatioloji/Scripts/ruff.exe format src/s_spatioloji/spatial/ tests/unit/test_spatial_polygon_*.py`

---

## Chunk 1: Foundation (fixtures, package init, contact graph)

### Task 1: Test Fixtures — `sj_with_boundaries` and `sj_with_graph`

**Files:**
- Modify: `tests/unit/conftest.py`

The fixtures create synthetic Voronoi polygons from existing cell coordinates, producing a realistic tessellation where neighboring cells share boundaries.

- [ ] **Step 1: Add `sj_with_boundaries` fixture to conftest.py**

Add after the existing `sj` fixture:

```python
@pytest.fixture()
def sj_with_boundaries(sj):
    """sj with synthetic Voronoi polygon boundaries.

    Creates a Voronoi tessellation from cell centroids, clips to a bounding
    rectangle, and writes boundaries.parquet as GeoParquet.
    """
    import geopandas as gpd
    from scipy.spatial import Voronoi
    from shapely.geometry import Polygon, box
    from shapely.ops import clip_by_rect

    from s_spatioloji.data.boundaries import BoundaryStore

    cells_df = sj.cells.df.compute()
    coords = cells_df[["x", "y"]].values

    # Add mirror points so edge cells get finite Voronoi regions
    x_min, y_min = coords.min(axis=0) - 100
    x_max, y_max = coords.max(axis=0) + 100
    mirror_pts = np.concatenate([
        coords,
        np.column_stack([2 * x_min - coords[:, 0], coords[:, 1]]),
        np.column_stack([2 * x_max - coords[:, 0], coords[:, 1]]),
        np.column_stack([coords[:, 0], 2 * y_min - coords[:, 1]]),
        np.column_stack([coords[:, 0], 2 * y_max - coords[:, 1]]),
    ])

    vor = Voronoi(mirror_pts)
    clip_box = (x_min, y_min, x_max, y_max)

    geometries = []
    for i in range(len(coords)):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            # Fallback: small square around centroid
            cx, cy = coords[i]
            geometries.append(box(cx - 5, cy - 5, cx + 5, cy + 5))
        else:
            verts = [vor.vertices[v] for v in region]
            poly = Polygon(verts)
            poly = clip_by_rect(poly, *clip_box)
            if poly.is_empty or not poly.is_valid:
                cx, cy = coords[i]
                geometries.append(box(cx - 5, cy - 5, cx + 5, cy + 5))
            else:
                geometries.append(poly)

    gdf = gpd.GeoDataFrame({
        "cell_id": cells_df["cell_id"].values,
        "geometry": geometries,
    })
    BoundaryStore.create(sj.config.paths.boundaries, gdf)
    return sj
```

- [ ] **Step 2: Add `sj_with_graph` fixture**

Add after `sj_with_boundaries`:

```python
@pytest.fixture()
def sj_with_graph(sj_with_boundaries):
    """sj_with_boundaries + pre-built contact graph."""
    from s_spatioloji.spatial.polygon.graph import build_contact_graph

    build_contact_graph(sj_with_boundaries)
    return sj_with_boundaries
```

- [ ] **Step 3: Add `sj_with_clusters` fixture**

This fixture adds fake cluster labels that downstream neighborhood/pattern tests need:

```python
@pytest.fixture()
def sj_with_clusters(sj_with_graph):
    """sj_with_graph + synthetic cluster labels in maps/leiden.parquet."""
    cells_df = sj_with_graph.cells.df.compute()
    # Assign 4 clusters based on spatial position (spatially coherent)
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
    maps_dir = sj_with_graph.config.root / "maps"
    maps_dir.mkdir(exist_ok=True)
    _atomic_write_parquet(df, maps_dir, "leiden")
    return sj_with_graph
```

- [ ] **Step 4: Verify fixtures load without error**

Run: `PYTHONPATH=src /c/Users/YJ071/anaconda3/python.exe -m pytest tests/unit/conftest.py --collect-only`
Expected: Collection succeeds, fixtures are recognized.

---

### Task 2: Package Init Files

**Files:**
- Create: `src/s_spatioloji/spatial/__init__.py`
- Create: `src/s_spatioloji/spatial/polygon/__init__.py`

- [ ] **Step 1: Create empty spatial package init**

```python
"""Spatial analysis modules for s_spatioloji."""
```

- [ ] **Step 2: Create polygon package init with re-exports**

```python
"""Polygon-based spatial analysis.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.
"""

from s_spatioloji.spatial.polygon.graph import build_contact_graph
from s_spatioloji.spatial.polygon.morphology import cell_morphology
from s_spatioloji.spatial.polygon.neighborhoods import (
    neighborhood_composition,
    neighborhood_diversity,
    nth_order_neighbors,
)
from s_spatioloji.spatial.polygon.patterns import (
    border_enrichment,
    clustering_coefficient,
    colocalization,
    gearys_c,
    morans_i,
)
from s_spatioloji.spatial.polygon.statistics import permutation_test, quadrat_density

__all__ = [
    "build_contact_graph",
    "cell_morphology",
    "neighborhood_composition",
    "nth_order_neighbors",
    "neighborhood_diversity",
    "colocalization",
    "morans_i",
    "gearys_c",
    "clustering_coefficient",
    "border_enrichment",
    "permutation_test",
    "quadrat_density",
]
```

Note: This will fail to import until all submodules exist. That's fine — we'll create stub files in each task.

---

### Task 3: Contact Graph — `build_contact_graph` and `_load_contact_graph`

**Files:**
- Create: `src/s_spatioloji/spatial/polygon/graph.py`
- Create: `tests/unit/test_spatial_polygon_graph.py`

- [ ] **Step 1: Write tests for graph.py**

```python
"""Unit tests for s_spatioloji.spatial.polygon.graph."""

from __future__ import annotations

import networkx as nx
import pytest

from s_spatioloji.spatial.polygon.graph import _load_contact_graph, build_contact_graph


class TestBuildContactGraph:
    def test_returns_key(self, sj_with_boundaries):
        assert build_contact_graph(sj_with_boundaries) == "contact_graph"

    def test_output_written(self, sj_with_boundaries):
        build_contact_graph(sj_with_boundaries)
        assert sj_with_boundaries.maps.has("contact_graph")

    def test_columns(self, sj_with_boundaries):
        build_contact_graph(sj_with_boundaries)
        df = sj_with_boundaries.maps["contact_graph"].compute()
        assert list(df.columns) == ["cell_id_a", "cell_id_b", "shared_length"]

    def test_edges_exist(self, sj_with_boundaries):
        """Voronoi tessellation should produce many touching pairs."""
        build_contact_graph(sj_with_boundaries)
        df = sj_with_boundaries.maps["contact_graph"].compute()
        assert len(df) > 50  # 200 cells in a grid → many edges

    def test_edge_ordering(self, sj_with_boundaries):
        """cell_id_a < cell_id_b lexicographically."""
        build_contact_graph(sj_with_boundaries)
        df = sj_with_boundaries.maps["contact_graph"].compute()
        assert (df["cell_id_a"] < df["cell_id_b"]).all()

    def test_shared_length_nonneg(self, sj_with_boundaries):
        build_contact_graph(sj_with_boundaries)
        df = sj_with_boundaries.maps["contact_graph"].compute()
        assert (df["shared_length"] >= 0).all()

    def test_buffer_distance(self, sj_with_boundaries):
        """Larger buffer → more edges."""
        build_contact_graph(sj_with_boundaries, buffer_distance=0.0, output_key="g0")
        build_contact_graph(sj_with_boundaries, buffer_distance=10.0, output_key="g10")
        df0 = sj_with_boundaries.maps["g0"].compute()
        df10 = sj_with_boundaries.maps["g10"].compute()
        assert len(df10) >= len(df0)

    def test_force_false_skips(self, sj_with_boundaries):
        build_contact_graph(sj_with_boundaries)
        build_contact_graph(sj_with_boundaries, force=False)  # should not crash

    def test_custom_output_key(self, sj_with_boundaries):
        assert build_contact_graph(sj_with_boundaries, output_key="my_graph") == "my_graph"
        assert sj_with_boundaries.maps.has("my_graph")


class TestLoadContactGraph:
    def test_returns_networkx_graph(self, sj_with_graph):
        G = _load_contact_graph(sj_with_graph)
        assert isinstance(G, nx.Graph)

    def test_node_count(self, sj_with_graph):
        G = _load_contact_graph(sj_with_graph)
        # Not all 200 cells may be in the graph if some are isolated, but most should be
        assert len(G.nodes) > 100

    def test_edge_weights(self, sj_with_graph):
        G = _load_contact_graph(sj_with_graph)
        for _, _, data in G.edges(data=True):
            assert "shared_length" in data
            assert data["shared_length"] >= 0

    def test_missing_graph_raises(self, sj_with_boundaries):
        with pytest.raises(FileNotFoundError, match="Contact graph not found"):
            _load_contact_graph(sj_with_boundaries)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src /c/Users/YJ071/anaconda3/python.exe -m pytest tests/unit/test_spatial_polygon_graph.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement graph.py**

```python
"""Contact graph construction from polygon boundaries.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
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

            # Compute shared boundary length from original (unbuffered) geometries
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src /c/Users/YJ071/anaconda3/python.exe -m pytest tests/unit/test_spatial_polygon_graph.py -v`
Expected: All PASS

- [ ] **Step 5: Lint**

Run: `/c/Users/YJ071/anaconda3/envs/spatioloji/Scripts/ruff.exe check src/s_spatioloji/spatial/ tests/unit/test_spatial_polygon_graph.py --fix && /c/Users/YJ071/anaconda3/envs/spatioloji/Scripts/ruff.exe format src/s_spatioloji/spatial/ tests/unit/test_spatial_polygon_graph.py`

- [ ] **Step 6: Commit**

```bash
git add src/s_spatioloji/spatial/ tests/unit/conftest.py tests/unit/test_spatial_polygon_graph.py
git commit -m "feat(spatial/polygon): add contact graph with STRtree index"
```

---

## Chunk 2: Morphology

### Task 4: Cell Morphology — `cell_morphology`

**Files:**
- Create: `src/s_spatioloji/spatial/polygon/morphology.py`
- Create: `tests/unit/test_spatial_polygon_morphology.py`

- [ ] **Step 1: Write tests for morphology.py**

```python
"""Unit tests for s_spatioloji.spatial.polygon.morphology."""

from __future__ import annotations

import numpy as np

from s_spatioloji.spatial.polygon.morphology import cell_morphology


EXPECTED_COLUMNS = [
    "cell_id", "area", "perimeter", "centroid_x", "centroid_y",
    "circularity", "elongation", "solidity", "eccentricity", "aspect_ratio",
    "fractal_dimension", "vertex_count", "convexity_defects", "rectangularity",
]


class TestCellMorphology:
    def test_returns_key(self, sj_with_boundaries):
        assert cell_morphology(sj_with_boundaries) == "morphology"

    def test_output_written(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        assert sj_with_boundaries.maps.has("morphology")

    def test_columns(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        df = sj_with_boundaries.maps["morphology"].compute()
        assert list(df.columns) == EXPECTED_COLUMNS

    def test_shape(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        df = sj_with_boundaries.maps["morphology"].compute()
        assert df.shape == (200, 14)  # 200 cells, 13 metrics + cell_id

    def test_area_positive(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        df = sj_with_boundaries.maps["morphology"].compute()
        assert (df["area"] > 0).all()

    def test_perimeter_positive(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        df = sj_with_boundaries.maps["morphology"].compute()
        assert (df["perimeter"] > 0).all()

    def test_circularity_range(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        df = sj_with_boundaries.maps["morphology"].compute()
        assert (df["circularity"] > 0).all()
        assert (df["circularity"] <= 1.0 + 1e-6).all()

    def test_solidity_range(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        df = sj_with_boundaries.maps["morphology"].compute()
        assert (df["solidity"] > 0).all()
        assert (df["solidity"] <= 1.0 + 1e-6).all()

    def test_elongation_range(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        df = sj_with_boundaries.maps["morphology"].compute()
        assert (df["elongation"] >= 0).all()
        assert (df["elongation"] < 1.0).all()

    def test_vertex_count_positive(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        df = sj_with_boundaries.maps["morphology"].compute()
        assert (df["vertex_count"] >= 3).all()

    def test_rectangularity_range(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        df = sj_with_boundaries.maps["morphology"].compute()
        assert (df["rectangularity"] > 0).all()
        assert (df["rectangularity"] <= 1.0 + 1e-6).all()

    def test_force_false_skips(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        cell_morphology(sj_with_boundaries, force=False)

    def test_custom_output_key(self, sj_with_boundaries):
        assert cell_morphology(sj_with_boundaries, output_key="morph2") == "morph2"
        assert sj_with_boundaries.maps.has("morph2")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src /c/Users/YJ071/anaconda3/python.exe -m pytest tests/unit/test_spatial_polygon_morphology.py -v`
Expected: FAIL

- [ ] **Step 3: Implement morphology.py**

```python
"""Cell morphology metrics from polygon boundaries.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd

from s_spatioloji.compute import _atomic_write_parquet

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def _fractal_dimension(coords: np.ndarray) -> float:
    """Box-counting fractal dimension of a 2D point set.

    Args:
        coords: Array of shape (N, 2) with boundary coordinates.

    Returns:
        Estimated fractal dimension (typically 1.0-1.5 for cell boundaries).
    """
    if len(coords) < 4:
        return 1.0

    # Normalize to unit square
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = maxs - mins
    span[span == 0] = 1.0
    normed = (coords - mins) / span

    sizes = []
    counts = []
    for k in range(1, 8):
        cell_size = 1.0 / (2**k)
        bins = set()
        for pt in normed:
            bx = int(pt[0] / cell_size)
            by = int(pt[1] / cell_size)
            bins.add((bx, by))
        if len(bins) > 0:
            sizes.append(cell_size)
            counts.append(len(bins))

    if len(sizes) < 2:
        return 1.0

    log_sizes = np.log(1.0 / np.array(sizes))
    log_counts = np.log(np.array(counts, dtype=float))
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return float(max(1.0, coeffs[0]))


def cell_morphology(
    sj: s_spatioloji,
    output_key: str = "morphology",
    force: bool = True,
) -> str:
    """Compute 13 morphology metrics per cell from polygon boundaries.

    Metrics include area, perimeter, centroid, circularity, elongation,
    solidity, eccentricity, aspect ratio, fractal dimension, vertex count,
    convexity defects, and rectangularity.

    Args:
        sj: Dataset instance.
        output_key: Key to write morphology table under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> cell_morphology(sj)
        'morphology'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    gdf = sj.boundaries.load()

    records = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        cell_id = row["cell_id"]

        # Basic
        area = geom.area
        perimeter = geom.length
        centroid = geom.centroid
        centroid_x = centroid.x
        centroid_y = centroid.y

        # Shape descriptors
        circularity = (4.0 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0

        # Minimum rotated rectangle for elongation and aspect_ratio
        mrr = geom.minimum_rotated_rectangle
        mrr_coords = list(mrr.exterior.coords)
        edge_lengths = []
        for k in range(4):
            dx = mrr_coords[k + 1][0] - mrr_coords[k][0]
            dy = mrr_coords[k + 1][1] - mrr_coords[k][1]
            edge_lengths.append(np.sqrt(dx**2 + dy**2))
        major = max(edge_lengths)
        minor = min(edge_lengths)
        elongation = 1.0 - (minor / major) if major > 0 else 0.0
        aspect_ratio = major / minor if minor > 0 else 1.0

        # Solidity
        convex_hull = geom.convex_hull
        hull_area = convex_hull.area
        solidity = area / hull_area if hull_area > 0 else 1.0

        # Eccentricity via OpenCV fitEllipse
        ext_coords = np.array(geom.exterior.coords[:-1], dtype=np.float32)
        if len(ext_coords) >= 5:
            contour = ext_coords.reshape(-1, 1, 2)
            (_, _), (ma, MA), _ = cv2.fitEllipse(contour)
            if MA > 0:
                ratio = min(ma, MA) / max(ma, MA)
                eccentricity = np.sqrt(1.0 - ratio**2)
            else:
                eccentricity = 0.0
        else:
            eccentricity = 0.0

        # Boundary complexity
        fd = _fractal_dimension(ext_coords)
        vertex_count = len(ext_coords)
        convexity_defects = hull_area - area

        # Rectangularity
        bbox = geom.bounds  # (minx, miny, maxx, maxy)
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        rectangularity = area / bbox_area if bbox_area > 0 else 0.0

        records.append({
            "cell_id": cell_id,
            "area": area,
            "perimeter": perimeter,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "circularity": circularity,
            "elongation": elongation,
            "solidity": solidity,
            "eccentricity": eccentricity,
            "aspect_ratio": aspect_ratio,
            "fractal_dimension": fd,
            "vertex_count": vertex_count,
            "convexity_defects": convexity_defects,
            "rectangularity": rectangularity,
        })

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src /c/Users/YJ071/anaconda3/python.exe -m pytest tests/unit/test_spatial_polygon_morphology.py -v`
Expected: All PASS

- [ ] **Step 5: Lint and commit**

```bash
ruff check + format
git add src/s_spatioloji/spatial/polygon/morphology.py tests/unit/test_spatial_polygon_morphology.py
git commit -m "feat(spatial/polygon): add cell morphology metrics (13 descriptors)"
```

---

## Chunk 3: Neighborhoods

### Task 5: Neighborhood Functions

**Files:**
- Create: `src/s_spatioloji/spatial/polygon/neighborhoods.py`
- Create: `tests/unit/test_spatial_polygon_neighborhoods.py`

- [ ] **Step 1: Write tests for neighborhoods.py**

```python
"""Unit tests for s_spatioloji.spatial.polygon.neighborhoods."""

from __future__ import annotations

import numpy as np
import pytest

from s_spatioloji.spatial.polygon.neighborhoods import (
    neighborhood_composition,
    neighborhood_diversity,
    nth_order_neighbors,
)


class TestNeighborhoodComposition:
    def test_returns_key(self, sj_with_clusters):
        assert neighborhood_composition(sj_with_clusters) == "nhood_composition"

    def test_output_written(self, sj_with_clusters):
        neighborhood_composition(sj_with_clusters)
        assert sj_with_clusters.maps.has("nhood_composition")

    def test_cell_id_column(self, sj_with_clusters):
        neighborhood_composition(sj_with_clusters)
        df = sj_with_clusters.maps["nhood_composition"].compute()
        assert df.columns[0] == "cell_id"

    def test_shape(self, sj_with_clusters):
        neighborhood_composition(sj_with_clusters)
        df = sj_with_clusters.maps["nhood_composition"].compute()
        assert df.shape[0] == 200  # all cells
        assert df.shape[1] >= 2  # cell_id + at least 1 cluster column

    def test_counts_nonneg(self, sj_with_clusters):
        neighborhood_composition(sj_with_clusters)
        df = sj_with_clusters.maps["nhood_composition"].compute()
        num_cols = [c for c in df.columns if c != "cell_id"]
        assert (df[num_cols].values >= 0).all()

    def test_force_false_skips(self, sj_with_clusters):
        neighborhood_composition(sj_with_clusters)
        neighborhood_composition(sj_with_clusters, force=False)

    def test_missing_graph_raises(self, sj_with_boundaries):
        with pytest.raises(FileNotFoundError):
            neighborhood_composition(sj_with_boundaries)


class TestNthOrderNeighbors:
    def test_returns_key(self, sj_with_graph):
        assert nth_order_neighbors(sj_with_graph) == "nhood_nth_order"

    def test_columns_order2(self, sj_with_graph):
        nth_order_neighbors(sj_with_graph, order=2)
        df = sj_with_graph.maps["nhood_nth_order"].compute()
        assert list(df.columns) == ["cell_id", "n_order_1", "n_order_2"]

    def test_columns_order3(self, sj_with_graph):
        nth_order_neighbors(sj_with_graph, order=3, output_key="nth3")
        df = sj_with_graph.maps["nth3"].compute()
        assert list(df.columns) == ["cell_id", "n_order_1", "n_order_2", "n_order_3"]

    def test_shape(self, sj_with_graph):
        nth_order_neighbors(sj_with_graph)
        df = sj_with_graph.maps["nhood_nth_order"].compute()
        assert df.shape[0] == 200

    def test_counts_nonneg(self, sj_with_graph):
        nth_order_neighbors(sj_with_graph)
        df = sj_with_graph.maps["nhood_nth_order"].compute()
        assert (df["n_order_1"] >= 0).all()
        assert (df["n_order_2"] >= 0).all()

    def test_force_false_skips(self, sj_with_graph):
        nth_order_neighbors(sj_with_graph)
        nth_order_neighbors(sj_with_graph, force=False)


class TestNeighborhoodDiversity:
    def test_returns_key(self, sj_with_clusters):
        assert neighborhood_diversity(sj_with_clusters) == "nhood_diversity"

    def test_columns(self, sj_with_clusters):
        neighborhood_diversity(sj_with_clusters)
        df = sj_with_clusters.maps["nhood_diversity"].compute()
        assert list(df.columns) == ["cell_id", "shannon", "simpson"]

    def test_shape(self, sj_with_clusters):
        neighborhood_diversity(sj_with_clusters)
        df = sj_with_clusters.maps["nhood_diversity"].compute()
        assert df.shape == (200, 3)

    def test_shannon_nonneg(self, sj_with_clusters):
        neighborhood_diversity(sj_with_clusters)
        df = sj_with_clusters.maps["nhood_diversity"].compute()
        assert (df["shannon"] >= 0).all()

    def test_simpson_range(self, sj_with_clusters):
        neighborhood_diversity(sj_with_clusters)
        df = sj_with_clusters.maps["nhood_diversity"].compute()
        assert (df["simpson"] >= 0).all()
        assert (df["simpson"] <= 1.0).all()

    def test_force_false_skips(self, sj_with_clusters):
        neighborhood_diversity(sj_with_clusters)
        neighborhood_diversity(sj_with_clusters, force=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src /c/Users/YJ071/anaconda3/python.exe -m pytest tests/unit/test_spatial_polygon_neighborhoods.py -v`
Expected: FAIL

- [ ] **Step 3: Implement neighborhoods.py**

```python
"""Neighborhood analysis functions for polygon-based spatial data.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.
"""

from __future__ import annotations

from collections import defaultdict, deque
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
    cell_to_cluster = dict(zip(cluster_df["cell_id"], cluster_df[cluster_key]))
    all_clusters = sorted(set(cell_to_cluster.values()), key=str)

    # All cells from the cells store (not just graph nodes)
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
    cell_to_cluster = dict(zip(cluster_df["cell_id"], cluster_df[cluster_key]))

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

        # Count frequencies
        counts = defaultdict(int)
        for lab in neighbor_labels:
            counts[lab] += 1
        total = len(neighbor_labels)
        probs = np.array([c / total for c in counts.values()])

        # Shannon entropy: -sum(p * log(p))
        shannon = -np.sum(probs * np.log(probs + 1e-15))

        # Gini-Simpson: 1 - sum(p^2)
        simpson = 1.0 - np.sum(probs**2)

        records.append({"cell_id": cell_id, "shannon": float(shannon), "simpson": float(simpson)})

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src /c/Users/YJ071/anaconda3/python.exe -m pytest tests/unit/test_spatial_polygon_neighborhoods.py -v`
Expected: All PASS

- [ ] **Step 5: Lint and commit**

```bash
ruff check + format
git add src/s_spatioloji/spatial/polygon/neighborhoods.py tests/unit/test_spatial_polygon_neighborhoods.py
git commit -m "feat(spatial/polygon): add neighborhood composition, nth-order, and diversity"
```

---

## Chunk 4: Patterns

### Task 6: Pattern Functions

**Files:**
- Create: `src/s_spatioloji/spatial/polygon/patterns.py`
- Create: `tests/unit/test_spatial_polygon_patterns.py`

- [ ] **Step 1: Write tests for patterns.py**

```python
"""Unit tests for s_spatioloji.spatial.polygon.patterns."""

from __future__ import annotations

import numpy as np
import pytest

from s_spatioloji.spatial.polygon.patterns import (
    border_enrichment,
    clustering_coefficient,
    colocalization,
    gearys_c,
    morans_i,
)


class TestColocalization:
    def test_returns_key(self, sj_with_clusters):
        assert colocalization(sj_with_clusters) == "colocalization"

    def test_output_written(self, sj_with_clusters):
        colocalization(sj_with_clusters)
        assert sj_with_clusters.maps.has("colocalization")

    def test_columns(self, sj_with_clusters):
        colocalization(sj_with_clusters)
        df = sj_with_clusters.maps["colocalization"].compute()
        assert list(df.columns) == [
            "cluster_a", "cluster_b", "observed", "expected", "ratio", "log2_ratio",
        ]

    def test_has_rows(self, sj_with_clusters):
        colocalization(sj_with_clusters)
        df = sj_with_clusters.maps["colocalization"].compute()
        # 4 clusters → 4+6=10 pairs (including self-pairs)
        assert len(df) > 0

    def test_observed_nonneg(self, sj_with_clusters):
        colocalization(sj_with_clusters)
        df = sj_with_clusters.maps["colocalization"].compute()
        assert (df["observed"] >= 0).all()

    def test_force_false_skips(self, sj_with_clusters):
        colocalization(sj_with_clusters)
        colocalization(sj_with_clusters, force=False)


class TestMoransI:
    def test_returns_key(self, sj_with_clusters):
        assert morans_i(sj_with_clusters) == "morans_i"

    def test_columns(self, sj_with_clusters):
        morans_i(sj_with_clusters)
        df = sj_with_clusters.maps["morans_i"].compute()
        assert list(df.columns) == ["feature", "I", "expected_I", "z_score", "p_value"]

    def test_categorical_produces_rows(self, sj_with_clusters):
        """Categorical feature (leiden) → one row per category."""
        morans_i(sj_with_clusters)
        df = sj_with_clusters.maps["morans_i"].compute()
        assert len(df) == 4  # 4 clusters

    def test_spatially_clustered_high_I(self, sj_with_clusters):
        """Spatially coherent clusters should have positive Moran's I."""
        morans_i(sj_with_clusters)
        df = sj_with_clusters.maps["morans_i"].compute()
        # At least some categories should have I > 0
        assert (df["I"] > 0).any()

    def test_force_false_skips(self, sj_with_clusters):
        morans_i(sj_with_clusters)
        morans_i(sj_with_clusters, force=False)


class TestGearysC:
    def test_returns_key(self, sj_with_clusters):
        assert gearys_c(sj_with_clusters) == "gearys_c"

    def test_columns(self, sj_with_clusters):
        gearys_c(sj_with_clusters)
        df = sj_with_clusters.maps["gearys_c"].compute()
        assert list(df.columns) == ["feature", "C", "expected_C", "z_score", "p_value"]

    def test_spatially_clustered_low_C(self, sj_with_clusters):
        """Spatially coherent clusters should have C < 1."""
        gearys_c(sj_with_clusters)
        df = sj_with_clusters.maps["gearys_c"].compute()
        assert (df["C"] < 1.0).any()

    def test_force_false_skips(self, sj_with_clusters):
        gearys_c(sj_with_clusters)
        gearys_c(sj_with_clusters, force=False)


class TestClusteringCoefficient:
    def test_returns_key(self, sj_with_graph):
        assert clustering_coefficient(sj_with_graph) == "clustering_coeff"

    def test_columns(self, sj_with_graph):
        clustering_coefficient(sj_with_graph)
        df = sj_with_graph.maps["clustering_coeff"].compute()
        assert list(df.columns) == ["cell_id", "clustering_coeff"]

    def test_shape(self, sj_with_graph):
        clustering_coefficient(sj_with_graph)
        df = sj_with_graph.maps["clustering_coeff"].compute()
        assert df.shape[0] == 200

    def test_range(self, sj_with_graph):
        clustering_coefficient(sj_with_graph)
        df = sj_with_graph.maps["clustering_coeff"].compute()
        assert (df["clustering_coeff"] >= 0).all()
        assert (df["clustering_coeff"] <= 1.0).all()

    def test_force_false_skips(self, sj_with_graph):
        clustering_coefficient(sj_with_graph)
        clustering_coefficient(sj_with_graph, force=False)


class TestBorderEnrichment:
    def test_returns_key(self, sj_with_clusters):
        assert border_enrichment(sj_with_clusters) == "border_enrichment"

    def test_columns(self, sj_with_clusters):
        border_enrichment(sj_with_clusters)
        df = sj_with_clusters.maps["border_enrichment"].compute()
        assert list(df.columns) == [
            "cluster", "n_cells", "n_border", "border_fraction", "enrichment",
        ]

    def test_has_rows(self, sj_with_clusters):
        border_enrichment(sj_with_clusters)
        df = sj_with_clusters.maps["border_enrichment"].compute()
        assert len(df) == 4  # 4 clusters

    def test_n_cells_positive(self, sj_with_clusters):
        border_enrichment(sj_with_clusters)
        df = sj_with_clusters.maps["border_enrichment"].compute()
        assert (df["n_cells"] > 0).all()

    def test_border_fraction_range(self, sj_with_clusters):
        border_enrichment(sj_with_clusters)
        df = sj_with_clusters.maps["border_enrichment"].compute()
        assert (df["border_fraction"] >= 0).all()
        assert (df["border_fraction"] <= 1.0).all()

    def test_force_false_skips(self, sj_with_clusters):
        border_enrichment(sj_with_clusters)
        border_enrichment(sj_with_clusters, force=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src /c/Users/YJ071/anaconda3/python.exe -m pytest tests/unit/test_spatial_polygon_patterns.py -v`
Expected: FAIL

- [ ] **Step 3: Implement patterns.py**

```python
"""Spatial pattern analysis functions for polygon-based spatial data.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from s_spatioloji.compute import _atomic_write_parquet
from s_spatioloji.spatial.polygon.graph import _load_contact_graph

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def colocalization(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "contact_graph",
    output_key: str = "colocalization",
    force: bool = True,
) -> str:
    """Observed vs expected contact frequency for cluster pairs.

    Args:
        sj: Dataset instance.
        cluster_key: Key for cluster labels in maps/.
        graph_key: Key for the contact graph edge list.
        output_key: Key to write colocalization table under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> colocalization(sj, cluster_key="leiden")
        'colocalization'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    G = _load_contact_graph(sj, graph_key)

    cluster_df = sj.maps[cluster_key].compute()
    cell_to_cluster = dict(zip(cluster_df["cell_id"], cluster_df[cluster_key]))
    cluster_labels = sorted(set(cell_to_cluster.values()), key=str)

    # Count cluster sizes
    cluster_sizes = defaultdict(int)
    for label in cell_to_cluster.values():
        cluster_sizes[label] += 1
    n_total = sum(cluster_sizes.values())
    total_edges = G.number_of_edges()

    # Count observed edges per cluster pair
    observed = defaultdict(int)
    for u, v in G.edges():
        if u in cell_to_cluster and v in cell_to_cluster:
            a, b = cell_to_cluster[u], cell_to_cluster[v]
            pair = tuple(sorted([str(a), str(b)]))
            observed[pair] += 1

    records = []
    for a, b in combinations_with_replacement(cluster_labels, 2):
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


def _compute_morans_i(values: np.ndarray, G: nx.Graph, nodes: list[str]) -> dict:
    """Compute Moran's I for a single numeric vector.

    Args:
        values: 1-D array of numeric values aligned with ``nodes``.
        G: Contact graph.
        nodes: Cell IDs corresponding to ``values``.

    Returns:
        Dict with keys: I, expected_I, z_score, p_value.
    """
    N = len(values)
    if N < 3:
        return {"I": 0.0, "expected_I": 0.0, "z_score": 0.0, "p_value": 1.0}

    x_bar = values.mean()
    dev = values - x_bar
    denom = np.sum(dev**2)
    if denom == 0:
        return {"I": 0.0, "expected_I": 0.0, "z_score": 0.0, "p_value": 1.0}

    node_idx = {n: i for i, n in enumerate(nodes)}
    W = 0.0
    numerator = 0.0
    S1 = 0.0
    degree_sums = np.zeros(N)

    for u, v in G.edges():
        if u in node_idx and v in node_idx:
            i, j = node_idx[u], node_idx[v]
            W += 2  # binary symmetric: w_ij = w_ji = 1
            numerator += dev[i] * dev[j]
            S1 += 2  # (w_ij + w_ji)^2 = 4, but S1 = 0.5 * sum = 2 per edge
            degree_sums[i] += 1
            degree_sums[j] += 1

    if W == 0:
        return {"I": 0.0, "expected_I": 0.0, "z_score": 0.0, "p_value": 1.0}

    numerator *= 2  # count both directions
    I = (N / W) * numerator / denom
    expected_I = -1.0 / (N - 1)

    # Randomization assumption variance
    S1_total = S1  # accumulated from the loop: 2 per edge with both endpoints in node_idx
    S2 = np.sum((2 * degree_sums)**2)  # sum_i (sum_j w_ij + sum_j w_ji)^2
    k = N * np.sum(dev**4) / denom**2  # kurtosis

    num1 = N * (S1_total * (N**2 - 3 * N + 3) - N * S2 + 3 * W**2)
    num2 = k * (S1_total * (N**2 - N) - 2 * N * S2 + 6 * W**2)
    var_I = (num1 - num2) / ((N - 1) * (N - 2) * (N - 3) * W**2) - expected_I**2
    var_I = max(var_I, 1e-15)

    z = (I - expected_I) / np.sqrt(var_I)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    return {"I": float(I), "expected_I": float(expected_I), "z_score": float(z), "p_value": float(p)}


def morans_i(
    sj: s_spatioloji,
    feature_key: str = "leiden",
    graph_key: str = "contact_graph",
    output_key: str = "morans_i",
    force: bool = True,
) -> str:
    """Moran's I spatial autocorrelation.

    For categorical features (dtype object/category), computes one-hot
    indicators and returns one row per category.

    Args:
        sj: Dataset instance.
        feature_key: Key for the feature in maps/.
        graph_key: Key for the contact graph edge list.
        output_key: Key to write Moran's I results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> morans_i(sj, feature_key="leiden")
        'morans_i'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    G = _load_contact_graph(sj, graph_key)

    feat_df = sj.maps[feature_key].compute()
    cell_ids = list(feat_df["cell_id"])
    value_cols = [c for c in feat_df.columns if c != "cell_id"]

    if len(value_cols) != 1:
        raise ValueError(f"Expected exactly one feature column besides cell_id, got {len(value_cols)}")

    col = value_cols[0]
    values = feat_df[col]

    records = []
    if values.dtype == object or hasattr(values, "cat"):
        # Categorical: one-hot encoding
        categories = sorted(values.unique(), key=str)
        for cat in categories:
            indicator = (values == cat).astype(float).values
            result = _compute_morans_i(indicator, G, cell_ids)
            result["feature"] = str(cat)
            records.append(result)
    else:
        result = _compute_morans_i(values.values.astype(float), G, cell_ids)
        result["feature"] = col
        records.append(result)

    df = pd.DataFrame(records)[["feature", "I", "expected_I", "z_score", "p_value"]]
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def _compute_gearys_c(values: np.ndarray, G: nx.Graph, nodes: list[str]) -> dict:
    """Compute Geary's C for a single numeric vector.

    Args:
        values: 1-D array of numeric values aligned with ``nodes``.
        G: Contact graph.
        nodes: Cell IDs corresponding to ``values``.

    Returns:
        Dict with keys: C, expected_C, z_score, p_value.
    """
    N = len(values)
    if N < 3:
        return {"C": 1.0, "expected_C": 1.0, "z_score": 0.0, "p_value": 1.0}

    x_bar = values.mean()
    dev = values - x_bar
    denom = np.sum(dev**2)
    if denom == 0:
        return {"C": 1.0, "expected_C": 1.0, "z_score": 0.0, "p_value": 1.0}

    node_idx = {n: i for i, n in enumerate(nodes)}
    W = 0.0
    numerator = 0.0

    for u, v in G.edges():
        if u in node_idx and v in node_idx:
            i, j = node_idx[u], node_idx[v]
            W += 2  # binary symmetric
            numerator += (values[i] - values[j]) ** 2

    if W == 0:
        return {"C": 1.0, "expected_C": 1.0, "z_score": 0.0, "p_value": 1.0}

    C = ((N - 1) / (2 * W)) * numerator / denom
    expected_C = 1.0

    # Randomization assumption variance (parallels Moran's I approach)
    node_idx_map = {n: i for i, n in enumerate(nodes)}
    degree_sums = np.zeros(N)
    edge_count = 0
    for u, v in G.edges():
        if u in node_idx_map and v in node_idx_map:
            degree_sums[node_idx_map[u]] += 1
            degree_sums[node_idx_map[v]] += 1
            edge_count += 1

    S1 = 2.0 * edge_count  # 0.5 * sum (w_ij + w_ji)^2 = 2 per edge for binary
    S2 = np.sum((2 * degree_sums)**2)
    k = N * np.sum((values - values.mean())**4) / denom**2  # kurtosis

    num1 = (N - 1) * S1 * (N**2 - 3 * N + 3 - (N - 1) * k)
    num2 = (1 / 4) * (N - 1) * S2 * (N**2 + 3 * N - 6 - (N**2 - N + 2) * k)
    num3 = W**2 * (N**2 - 3 - (N - 1)**2 * k)
    var_C = (num1 - num2 + num3) / ((N - 1) * (N - 2) * (N - 3) * W**2)
    var_C = max(var_C, 1e-15)

    z = (C - expected_C) / np.sqrt(var_C)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    return {"C": float(C), "expected_C": expected_C, "z_score": float(z), "p_value": float(p)}


def gearys_c(
    sj: s_spatioloji,
    feature_key: str = "leiden",
    graph_key: str = "contact_graph",
    output_key: str = "gearys_c",
    force: bool = True,
) -> str:
    """Geary's C spatial autocorrelation.

    Same interface and categorical handling as :func:`morans_i`.

    Args:
        sj: Dataset instance.
        feature_key: Key for the feature in maps/.
        graph_key: Key for the contact graph edge list.
        output_key: Key to write Geary's C results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> gearys_c(sj, feature_key="leiden")
        'gearys_c'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    G = _load_contact_graph(sj, graph_key)

    feat_df = sj.maps[feature_key].compute()
    cell_ids = list(feat_df["cell_id"])
    value_cols = [c for c in feat_df.columns if c != "cell_id"]

    if len(value_cols) != 1:
        raise ValueError(f"Expected exactly one feature column besides cell_id, got {len(value_cols)}")

    col = value_cols[0]
    values = feat_df[col]

    records = []
    if values.dtype == object or hasattr(values, "cat"):
        categories = sorted(values.unique(), key=str)
        for cat in categories:
            indicator = (values == cat).astype(float).values
            result = _compute_gearys_c(indicator, G, cell_ids)
            result["feature"] = str(cat)
            records.append(result)
    else:
        result = _compute_gearys_c(values.values.astype(float), G, cell_ids)
        result["feature"] = col
        records.append(result)

    df = pd.DataFrame(records)[["feature", "C", "expected_C", "z_score", "p_value"]]
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def clustering_coefficient(
    sj: s_spatioloji,
    graph_key: str = "contact_graph",
    output_key: str = "clustering_coeff",
    force: bool = True,
) -> str:
    """Per-cell local clustering coefficient.

    Args:
        sj: Dataset instance.
        graph_key: Key for the contact graph edge list.
        output_key: Key to write clustering coefficients under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> clustering_coefficient(sj)
        'clustering_coeff'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    G = _load_contact_graph(sj, graph_key)

    cells_df = sj.cells.df.compute()
    all_cell_ids = list(cells_df["cell_id"])

    cc = nx.clustering(G)
    records = [
        {"cell_id": cid, "clustering_coeff": cc.get(cid, 0.0)}
        for cid in all_cell_ids
    ]

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def border_enrichment(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "contact_graph",
    output_key: str = "border_enrichment",
    force: bool = True,
) -> str:
    """Identify border cells and compute enrichment per cluster.

    A cell is a border cell if at least one neighbor belongs to a
    different cluster.

    Args:
        sj: Dataset instance.
        cluster_key: Key for cluster labels in maps/.
        graph_key: Key for the contact graph edge list.
        output_key: Key to write border enrichment under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> border_enrichment(sj, cluster_key="leiden")
        'border_enrichment'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    G = _load_contact_graph(sj, graph_key)

    cluster_df = sj.maps[cluster_key].compute()
    cell_to_cluster = dict(zip(cluster_df["cell_id"], cluster_df[cluster_key]))

    # Count border cells per cluster
    cluster_cells = defaultdict(list)
    for cid, label in cell_to_cluster.items():
        cluster_cells[label].append(cid)

    n_total = len(cell_to_cluster)
    k_avg = 2.0 * G.number_of_edges() / max(n_total, 1)

    records = []
    for label in sorted(cluster_cells.keys(), key=str):
        cells = cluster_cells[label]
        n_cells = len(cells)
        n_border = 0
        for cid in cells:
            if G.has_node(cid):
                for neighbor in G.neighbors(cid):
                    if neighbor in cell_to_cluster and cell_to_cluster[neighbor] != label:
                        n_border += 1
                        break

        border_fraction = n_border / n_cells if n_cells > 0 else 0.0
        expected_fraction = 1.0 - (n_cells / max(n_total, 1)) ** k_avg if k_avg > 0 else 0.0
        enrichment = border_fraction / expected_fraction if expected_fraction > 0 else 0.0

        records.append({
            "cluster": label,
            "n_cells": n_cells,
            "n_border": n_border,
            "border_fraction": border_fraction,
            "enrichment": enrichment,
        })

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src /c/Users/YJ071/anaconda3/python.exe -m pytest tests/unit/test_spatial_polygon_patterns.py -v`
Expected: All PASS

- [ ] **Step 5: Lint and commit**

```bash
ruff check + format
git add src/s_spatioloji/spatial/polygon/patterns.py tests/unit/test_spatial_polygon_patterns.py
git commit -m "feat(spatial/polygon): add colocalization, Moran's I, Geary's C, clustering coeff, border enrichment"
```

---

## Chunk 5: Statistics and Final Integration

### Task 7: Statistics Functions

**Files:**
- Create: `src/s_spatioloji/spatial/polygon/statistics.py`
- Create: `tests/unit/test_spatial_polygon_statistics.py`

- [ ] **Step 1: Write tests for statistics.py**

```python
"""Unit tests for s_spatioloji.spatial.polygon.statistics."""

from __future__ import annotations

import numpy as np
import pytest

from s_spatioloji.spatial.polygon.statistics import permutation_test, quadrat_density


class TestPermutationTest:
    def test_returns_key(self, sj_with_clusters):
        assert permutation_test(sj_with_clusters, n_permutations=50) == "permutation_test"

    def test_output_written(self, sj_with_clusters):
        permutation_test(sj_with_clusters, n_permutations=50)
        assert sj_with_clusters.maps.has("permutation_test")

    def test_columns(self, sj_with_clusters):
        permutation_test(sj_with_clusters, n_permutations=50)
        df = sj_with_clusters.maps["permutation_test"].compute()
        assert list(df.columns) == [
            "cluster_a", "cluster_b", "observed_ratio", "p_value", "z_score",
        ]

    def test_has_rows(self, sj_with_clusters):
        permutation_test(sj_with_clusters, n_permutations=50)
        df = sj_with_clusters.maps["permutation_test"].compute()
        assert len(df) > 0

    def test_p_values_range(self, sj_with_clusters):
        permutation_test(sj_with_clusters, n_permutations=50)
        df = sj_with_clusters.maps["permutation_test"].compute()
        assert (df["p_value"] >= 0).all()
        assert (df["p_value"] <= 1.0).all()

    def test_reproducible(self, sj_with_clusters):
        permutation_test(sj_with_clusters, n_permutations=50, output_key="pt1")
        permutation_test(sj_with_clusters, n_permutations=50, output_key="pt2")
        df1 = sj_with_clusters.maps["pt1"].compute()
        df2 = sj_with_clusters.maps["pt2"].compute()
        np.testing.assert_array_equal(df1["p_value"].values, df2["p_value"].values)

    def test_force_false_skips(self, sj_with_clusters):
        permutation_test(sj_with_clusters, n_permutations=50)
        permutation_test(sj_with_clusters, n_permutations=50, force=False)


class TestQuadratDensity:
    def test_returns_key(self, sj_with_clusters):
        assert quadrat_density(sj_with_clusters) == "quadrat_density"

    def test_output_written(self, sj_with_clusters):
        quadrat_density(sj_with_clusters)
        assert sj_with_clusters.maps.has("quadrat_density")

    def test_columns(self, sj_with_clusters):
        quadrat_density(sj_with_clusters)
        df = sj_with_clusters.maps["quadrat_density"].compute()
        assert list(df.columns) == ["cluster", "chi2", "p_value", "density_mean", "density_std"]

    def test_has_rows(self, sj_with_clusters):
        quadrat_density(sj_with_clusters)
        df = sj_with_clusters.maps["quadrat_density"].compute()
        assert len(df) == 4  # 4 clusters

    def test_chi2_nonneg(self, sj_with_clusters):
        quadrat_density(sj_with_clusters)
        df = sj_with_clusters.maps["quadrat_density"].compute()
        assert (df["chi2"] >= 0).all()

    def test_p_values_range(self, sj_with_clusters):
        quadrat_density(sj_with_clusters)
        df = sj_with_clusters.maps["quadrat_density"].compute()
        assert (df["p_value"] >= 0).all()
        assert (df["p_value"] <= 1.0).all()

    def test_no_graph_needed(self, sj_with_clusters):
        """quadrat_density does NOT depend on contact graph."""
        # Just needs clusters and cell coords — would work even without a graph
        quadrat_density(sj_with_clusters)

    def test_force_false_skips(self, sj_with_clusters):
        quadrat_density(sj_with_clusters)
        quadrat_density(sj_with_clusters, force=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src /c/Users/YJ071/anaconda3/python.exe -m pytest tests/unit/test_spatial_polygon_statistics.py -v`
Expected: FAIL

- [ ] **Step 3: Implement statistics.py**

```python
"""Statistical testing functions for polygon-based spatial data.

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

from s_spatioloji.compute import _atomic_write_parquet
from s_spatioloji.spatial.polygon.graph import _load_contact_graph

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def permutation_test(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "contact_graph",
    n_permutations: int = 1000,
    random_state: int = 42,
    output_key: str = "permutation_test",
    force: bool = True,
) -> str:
    """Permutation test for spatial colocalization significance.

    Permutes cluster labels across cells, recomputes colocalization
    ratios each time, and compares observed ratios against the null
    distribution.

    Args:
        sj: Dataset instance.
        cluster_key: Key for cluster labels in maps/.
        graph_key: Key for the contact graph edge list.
        n_permutations: Number of label permutations.
        random_state: Seed for reproducibility.
        output_key: Key to write test results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> permutation_test(sj, n_permutations=1000)
        'permutation_test'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    G = _load_contact_graph(sj, graph_key)

    cluster_df = sj.maps[cluster_key].compute()
    cell_ids = list(cluster_df["cell_id"])
    labels = list(cluster_df[cluster_key])
    cell_to_cluster = dict(zip(cell_ids, labels))

    unique_labels = sorted(set(labels), key=str)
    pairs = list(combinations_with_replacement(unique_labels, 2))
    pair_keys = [tuple(sorted([str(a), str(b)])) for a, b in pairs]

    # Edges as index pairs for fast lookup
    edges = []
    for u, v in G.edges():
        if u in cell_to_cluster and v in cell_to_cluster:
            edges.append((u, v))

    def _count_edges(c2c):
        counts = defaultdict(int)
        for u, v in edges:
            a, b = c2c[u], c2c[v]
            pair = tuple(sorted([str(a), str(b)]))
            counts[pair] += 1
        return counts

    # Observed
    observed = _count_edges(cell_to_cluster)

    # Permutations
    rng = np.random.default_rng(random_state)
    perm_counts = {pk: [] for pk in pair_keys}

    for _ in range(n_permutations):
        shuffled_labels = rng.permutation(labels)
        perm_c2c = dict(zip(cell_ids, shuffled_labels))
        counts = _count_edges(perm_c2c)
        for pk in pair_keys:
            perm_counts[pk].append(counts.get(pk, 0))

    records = []
    for pk in pair_keys:
        obs = observed.get(pk, 0)
        null_dist = np.array(perm_counts[pk])
        null_mean = null_dist.mean()
        null_std = null_dist.std()

        if null_std > 0:
            z = (obs - null_mean) / null_std
        else:
            z = 0.0

        # p-value: fraction of permutations with count >= observed
        p = (np.sum(null_dist >= obs) + 1) / (n_permutations + 1)

        records.append({
            "cluster_a": pk[0],
            "cluster_b": pk[1],
            "observed_ratio": float(obs),
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
    output_key: str = "quadrat_density",
    force: bool = True,
) -> str:
    """Quadrat-based density analysis with chi-squared test.

    Divides the spatial extent into ``n_bins x n_bins`` grid quadrats,
    counts cells per cluster per quadrat, and tests for spatial uniformity
    using a chi-squared test.

    Args:
        sj: Dataset instance.
        cluster_key: Key for cluster labels in maps/.
        n_bins: Number of bins in each spatial dimension.
        output_key: Key to write density results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> quadrat_density(sj, n_bins=10)
        'quadrat_density'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)

    cluster_df = sj.maps[cluster_key].compute()
    cells_df = sj.cells.df.compute()

    # Merge cluster labels with cell coordinates
    merged = cells_df[["cell_id", "x", "y"]].merge(
        cluster_df[["cell_id", cluster_key]], on="cell_id"
    )

    x = merged["x"].values
    y = merged["y"].values
    labels = merged[cluster_key].values

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    # Avoid zero-width bins
    x_edges = np.linspace(x_min - 1e-6, x_max + 1e-6, n_bins + 1)
    y_edges = np.linspace(y_min - 1e-6, y_max + 1e-6, n_bins + 1)

    x_bin = np.digitize(x, x_edges) - 1
    y_bin = np.digitize(y, y_edges) - 1
    x_bin = np.clip(x_bin, 0, n_bins - 1)
    y_bin = np.clip(y_bin, 0, n_bins - 1)

    unique_labels = sorted(set(labels), key=str)
    records = []

    for label in unique_labels:
        mask = labels == label
        x_b = x_bin[mask]
        y_b = y_bin[mask]

        # Count cells per quadrat
        counts = np.zeros((n_bins, n_bins))
        for xi, yi in zip(x_b, y_b):
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src /c/Users/YJ071/anaconda3/python.exe -m pytest tests/unit/test_spatial_polygon_statistics.py -v`
Expected: All PASS

- [ ] **Step 5: Lint and commit**

```bash
ruff check + format
git add src/s_spatioloji/spatial/polygon/statistics.py tests/unit/test_spatial_polygon_statistics.py
git commit -m "feat(spatial/polygon): add permutation test and quadrat density"
```

---

### Task 8: Final Integration — Run All Tests

- [ ] **Step 1: Run full test suite**

Run: `PYTHONPATH=src /c/Users/YJ071/anaconda3/python.exe -m pytest tests/unit/ -v`
Expected: All tests PASS (existing + new spatial polygon tests)

- [ ] **Step 2: Lint all new code**

Run: `/c/Users/YJ071/anaconda3/envs/spatioloji/Scripts/ruff.exe check src/s_spatioloji/spatial/ tests/unit/test_spatial_polygon_*.py --fix && /c/Users/YJ071/anaconda3/envs/spatioloji/Scripts/ruff.exe format src/s_spatioloji/spatial/ tests/unit/test_spatial_polygon_*.py`

- [ ] **Step 3: Verify __init__.py imports work**

Run: `PYTHONPATH=src /c/Users/YJ071/anaconda3/python.exe -c "from s_spatioloji.spatial.polygon import build_contact_graph, cell_morphology, neighborhood_composition, nth_order_neighbors, neighborhood_diversity, colocalization, morans_i, gearys_c, clustering_coefficient, border_enrichment, permutation_test, quadrat_density; print('All 12 functions imported successfully')"`

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "chore(spatial/polygon): lint fixes and integration verification"
```
