# Spatial Polygon Module Design Spec

**Date:** 2026-03-13
**Module:** `src/s_spatioloji/spatial/polygon/`
**Status:** Approved

## Goal

Implement polygon-based spatial analysis for image-based spatial transcriptomics. Uses cell boundary geometries (from `BoundaryStore`) to build contact graphs and compute morphology, neighborhood, pattern, and statistical analyses.

## Architecture

All functions follow the compute layer contract: accept an `s_spatioloji` object, write results to `maps/` as Parquet, and return the output key string. All Parquet writes use the atomic write pattern (`_atomic_write_parquet` from `s_spatioloji.compute`) — write to temp file then rename. All functions call `maps_dir.mkdir(exist_ok=True)` before writing.

The contact graph is the foundational data structure — built once, persisted as an edge list Parquet, and loaded into networkx internally by downstream functions.

**Output types:** Most functions write per-cell tables (keyed by `cell_id`). Some spatial statistics functions write aggregate tables keyed by cluster pair or feature name (e.g., colocalization, permutation test). This is an intentional extension of the compute layer contract for spatial statistics that are inherently pairwise or per-cluster.

**Edge list convention:** The contact graph edge list stores each edge once as an unordered pair (`cell_id_a < cell_id_b` lexicographically). Downstream consumers must account for this when iterating.

**Dependency flow:**
```
graph.py (build_contact_graph)
    ├── neighborhoods.py (composition, nth_order, diversity)
    ├── patterns.py (colocalization, morans_i, gearys_c, clustering_coeff, border_enrichment)
    └── statistics.py (permutation_test)

morphology.py (independent — reads boundaries directly)
statistics.py/quadrat_density (independent — reads cell coordinates directly)
```

## Module Structure

```
src/s_spatioloji/spatial/
├── __init__.py
└── polygon/
    ├── __init__.py          # re-exports all public functions
    ├── graph.py             # build_contact_graph + _load_contact_graph
    ├── morphology.py        # cell_morphology
    ├── neighborhoods.py     # neighborhood_composition, nth_order_neighbors, neighborhood_diversity
    ├── patterns.py          # colocalization, morans_i, gearys_c, clustering_coefficient, border_enrichment
    └── statistics.py        # permutation_test, quadrat_density
```

## Function Specifications

### graph.py

#### `build_contact_graph`

```python
def build_contact_graph(
    sj: s_spatioloji,
    buffer_distance: float = 0.0,
    output_key: str = "contact_graph",
    force: bool = True,
) -> str:
```

- Loads boundaries from `sj.boundaries` (GeoParquet with Shapely geometries).
- Uses STRtree spatial index for efficient pairwise intersection testing.
- Buffers each geometry by `buffer_distance` before testing (0 = strict touching).
- Writes edge list Parquet to `maps/contact_graph.parquet` with columns: `cell_id_a`, `cell_id_b`, `shared_length` (length of shared boundary, 0 if buffer-only contact).
- Returns `output_key`.

#### `_load_contact_graph` (private, defined in `graph.py`, imported by other polygon modules)

```python
def _load_contact_graph(sj: s_spatioloji, graph_key: str = "contact_graph") -> nx.Graph:
```

- Loads edge list from `maps/<graph_key>.parquet` via `sj.maps[graph_key].compute()`.
- Constructs and returns `networkx.Graph` with `cell_id` string nodes and `shared_length` float edge weights.
- Raises `FileNotFoundError` with message "Contact graph not found. Run build_contact_graph() first." if the Parquet file does not exist.
- Import pattern: `from s_spatioloji.spatial.polygon.graph import _load_contact_graph`.

---

### morphology.py

#### `cell_morphology`

```python
def cell_morphology(
    sj: s_spatioloji,
    output_key: str = "morphology",
    force: bool = True,
) -> str:
```

- Loads boundaries from `sj.boundaries`.
- Computes 13 per-cell metrics:
  - **Basic (4):** `area`, `perimeter`, `centroid_x`, `centroid_y`
  - **Shape descriptors (5):** `circularity` (4pi*area/perimeter^2), `elongation` (1 - minor/major axis of minimum rotated rectangle), `solidity` (area/convex hull area), `eccentricity` (from fitted ellipse via OpenCV `fitEllipse`), `aspect_ratio` (major/minor axis ratio of minimum rotated rectangle)
  - **Boundary complexity (4):** `fractal_dimension` (box-counting on boundary coords), `vertex_count` (polygon vertex count), `convexity_defects` (convex hull area - polygon area), `rectangularity` (area / bounding box area)
- OpenCV `fitEllipse` requires converting Shapely polygon exterior coordinates to a numpy contour array `(N, 1, 2)` of dtype `float32`.
- Writes `maps/morphology.parquet`: `cell_id` + all metric columns.
- Returns `output_key`.

---

### neighborhoods.py

#### `neighborhood_composition`

```python
def neighborhood_composition(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "contact_graph",
    output_key: str = "nhood_composition",
    force: bool = True,
) -> str:
```

- Loads contact graph via `_load_contact_graph`.
- Loads cluster labels from `sj.maps[cluster_key]`.
- For each cell, counts neighbor cluster types.
- Writes `maps/nhood_composition.parquet`: `cell_id`, one column per cluster label (counts).
- Raises `FileNotFoundError` if contact graph not built.

#### `nth_order_neighbors`

```python
def nth_order_neighbors(
    sj: s_spatioloji,
    order: int = 2,
    graph_key: str = "contact_graph",
    output_key: str = "nhood_nth_order",
    force: bool = True,
) -> str:
```

- BFS up to `order` hops on contact graph. Excludes the cell itself from all counts. Counts at each order are exclusive (neighbors at order 2 excludes order 1 neighbors).
- Writes `maps/nhood_nth_order.parquet`: `cell_id`, columns `n_order_1` through `n_order_{order}` (one column per hop level, all computed in a single BFS pass).

#### `neighborhood_diversity`

```python
def neighborhood_diversity(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "contact_graph",
    output_key: str = "nhood_diversity",
    force: bool = True,
) -> str:
```

- Computes Shannon entropy (`-sum(p_i * log(p_i))`) and Gini-Simpson index (`1 - sum(p_i^2)`) of neighbor cluster composition, where `p_i` is the fraction of neighbors belonging to cluster `i`.
- Writes `maps/nhood_diversity.parquet`: `cell_id`, `shannon`, `simpson`.

---

### patterns.py

#### `colocalization`

```python
def colocalization(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "contact_graph",
    output_key: str = "colocalization",
    force: bool = True,
) -> str:
```

- For each pair of cluster labels, counts observed contact frequency vs expected.
- Expected value formula: `expected_ij = 2 * n_i * n_j * total_edges / (n_total * (n_total - 1))` for `i != j`, where `n_i` and `n_j` are cluster sizes, `n_total` is total cells, and `total_edges` is total edge count. For `i == j`: `expected_ii = n_i * (n_i - 1) * total_edges / (n_total * (n_total - 1))`.
- `ratio = observed / expected` (clamped: if expected == 0, ratio = 0). `log2_ratio = log2(ratio)` (nan if ratio == 0).
- Writes `maps/colocalization.parquet`: `cluster_a`, `cluster_b`, `observed`, `expected`, `ratio`, `log2_ratio`.

#### `morans_i`

```python
def morans_i(
    sj: s_spatioloji,
    feature_key: str = "leiden",
    graph_key: str = "contact_graph",
    output_key: str = "morans_i",
    force: bool = True,
) -> str:
```

- Spatial autocorrelation on a numeric feature. If `feature_key` points to a categorical column (detected by dtype object/category), computes one-hot indicators and returns one row per category.
- Uses contact graph as **binary** spatial weight matrix (0/1 adjacency, ignoring `shared_length`).
- Moran's I formula: `I = (N / W) * (sum_ij w_ij (x_i - x_bar)(x_j - x_bar)) / (sum_i (x_i - x_bar)^2)` where `N` = number of cells, `W` = sum of all weights, `w_ij` = binary adjacency.
- `expected_I = -1 / (N - 1)`. Z-score uses **randomization assumption** variance: `Var(I) = (N * (S1*(N^2-3N+3) - N*S2 + 3*W^2) - k*(S1*(N^2-N) - 2*N*S2 + 6*W^2)) / ((N-1)*(N-2)*(N-3)*W^2) - expected_I^2` where `S1`, `S2` are standard weight matrix sums and `k` is kurtosis. P-value is two-sided from normal approximation.
- `feature` column = column name for numeric, category name for categorical.
- Writes `maps/morans_i.parquet`: `feature`, `I`, `expected_I`, `z_score`, `p_value`.

#### `gearys_c`

```python
def gearys_c(
    sj: s_spatioloji,
    feature_key: str = "leiden",
    graph_key: str = "contact_graph",
    output_key: str = "gearys_c",
    force: bool = True,
) -> str:
```

- Same interface and categorical handling as `morans_i`. Uses binary weight matrix.
- Geary's C formula: `C = ((N-1) / (2*W)) * (sum_ij w_ij (x_i - x_j)^2) / (sum_i (x_i - x_bar)^2)`. `expected_C = 1.0`. Z-score uses randomization assumption variance. P-value two-sided from normal approximation.
- Writes `maps/gearys_c.parquet`: `feature`, `C`, `expected_C`, `z_score`, `p_value`.

#### `clustering_coefficient`

```python
def clustering_coefficient(
    sj: s_spatioloji,
    graph_key: str = "contact_graph",
    output_key: str = "clustering_coeff",
    force: bool = True,
) -> str:
```

- Per-cell local clustering coefficient from networkx.
- Writes `maps/clustering_coeff.parquet`: `cell_id`, `clustering_coeff`.

#### `border_enrichment`

```python
def border_enrichment(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "contact_graph",
    output_key: str = "border_enrichment",
    force: bool = True,
) -> str:
```

- Identifies cells at cluster boundaries (at least one neighbor belongs to a different cluster).
- Per-cluster: fraction of cells that are border cells, enrichment vs random expectation.
- Null model: under random label assignment, expected border fraction for cluster `i` = `1 - (n_i / N)^k_avg` where `n_i` is cluster size, `N` is total cells, `k_avg` is mean node degree. `enrichment = observed_border_fraction / expected_border_fraction`.
- Writes `maps/border_enrichment.parquet`: `cluster`, `n_cells`, `n_border`, `border_fraction`, `enrichment`.

---

### statistics.py

#### `permutation_test`

```python
def permutation_test(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "contact_graph",
    n_permutations: int = 1000,
    random_state: int = 42,
    output_key: str = "permutation_test",
    force: bool = True,
) -> str:
```

- Permutes cluster labels across cells using `np.random.default_rng(random_state)`, recomputes colocalization ratios each time.
- Compares observed ratios against null distribution.
- Writes `maps/permutation_test.parquet`: `cluster_a`, `cluster_b`, `observed_ratio`, `p_value`, `z_score`.

#### `quadrat_density`

```python
def quadrat_density(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    n_bins: int = 10,
    output_key: str = "quadrat_density",
    force: bool = True,
) -> str:
```

- Divides spatial extent into `n_bins x n_bins` grid quadrats.
- Counts cells per cluster per quadrat, computes chi-squared test for spatial uniformity.
- Writes `maps/quadrat_density.parquet`: `cluster`, `chi2`, `p_value`, `density_mean`, `density_std`.
- Does NOT depend on contact graph (uses cell coordinates from `sj.cells`).

---

## Design Decisions

1. **All outputs to `maps/` as Parquet** — consistent with compute layer contract, no mutation of cells.parquet.
2. **Contact graph as edge list Parquet** — persisted for caching, loaded into networkx internally via `_load_contact_graph` helper.
3. **Configurable `buffer_distance`** — default 0 (strict touching), positive values allow gap tolerance for segmentation imperfections.
4. **Graph-dependent functions raise `FileNotFoundError`** if contact graph not built — explicit over implicit, no silent graph building.
5. **All functions use `force` parameter** — consistent with compute layer skip-if-exists pattern.
6. **Core dependencies only** where possible — shapely, networkx, scipy, scikit-learn, opencv-python are all core deps. No optional dependency guards needed for this module.

## Testing Strategy

- Shared fixture: `sj_with_boundaries` — extends existing `sj` fixture with synthetic polygon boundaries (e.g., Voronoi tessellation of cell centroids).
- Contact graph fixture: `sj_with_graph` — `sj_with_boundaries` + pre-built contact graph.
- Each function gets: return key test, output shape test, column presence test, force=False skip test.
- Morphology: validate metric ranges (circularity in [0,1], area > 0, etc.).
- Patterns: validate against known synthetic arrangements (e.g., spatially clustered labels should have high Moran's I).
