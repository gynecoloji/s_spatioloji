# Spatial Point Module Design Spec

**Date:** 2026-03-13
**Module:** `src/s_spatioloji/spatial/point/`
**Status:** Approved

## Goal

Implement point-based (centroid) spatial analysis for image-based spatial transcriptomics. Uses cell centroid coordinates (x, y from `CellStore`) to build KNN and radius graphs and compute neighborhood, pattern, Ripley's, and statistical analyses. Designed to scale to 50–100 million cells with 5000 genes.

## Architecture

All functions follow the compute layer contract: accept an `s_spatioloji` object, write results to `maps/` as Parquet, and return the output key string. All Parquet writes use the atomic write pattern (`_atomic_write_parquet` from `s_spatioloji.compute`) — write to temp file then rename. All functions call `maps_dir.mkdir(exist_ok=True)` before writing.

The KNN graph is the foundational data structure — built once, persisted as an edge list Parquet, and loaded as a sparse matrix or networkx graph internally by downstream functions.

**Output key prefix:** All point module outputs use `pt_` prefix to avoid collision with polygon module outputs (e.g., `pt_nhood_composition` vs `nhood_composition`).

**Edge list convention:** The graph edge list stores each edge once as an unordered pair (`cell_id_a < cell_id_b` lexicographically). Downstream consumers must account for this when iterating.

**Scalability contract (50–100M cells):**
- Sparse matrices (`scipy.sparse.csr_matrix`) over networkx wherever possible.
- Tree-based counting (`scipy.spatial.cKDTree`) for Ripley's and distance-based statistics.
- Vectorized numpy/scipy throughout — zero Python for-loops over cells.
- Networkx used only where graph algorithms are required (BFS for nth_order, clustering coefficient).
- Subsampling parameter for inherently O(N^2) operations (Ripley's simulations).

**Dependency flow:**
```
graph.py (build_knn_graph, build_radius_graph)
    ├── neighborhoods.py (composition, nth_order, diversity)
    ├── patterns.py (colocalization, morans_i, gearys_c, clustering_coeff, getis_ord_gi)
    └── statistics.py/permutation_test

ripley.py (independent — uses coordinates + cKDTree directly)
statistics.py/clark_evans (independent — uses coordinates + cKDTree directly)
statistics.py/quadrat_density (independent — uses coordinates directly)
statistics.py/dclf_envelope (independent — uses coordinates + cKDTree directly)
```

## Module Structure

```
src/s_spatioloji/spatial/point/
├── __init__.py          # re-exports all public functions (lazy import pattern)
├── graph.py             # build_knn_graph, build_radius_graph, _load_point_graph, _load_point_graph_sparse
├── neighborhoods.py     # neighborhood_composition, nth_order_neighbors, neighborhood_diversity
├── patterns.py          # colocalization, morans_i, gearys_c, clustering_coefficient, getis_ord_gi
├── ripley.py            # ripley_k, ripley_l, ripley_g, ripley_f
└── statistics.py        # permutation_test, quadrat_density, clark_evans, dclf_envelope
```

## Function Specifications

### graph.py

#### `build_knn_graph`

```python
def build_knn_graph(
    sj: s_spatioloji,
    k: int = 10,
    output_key: str = "knn_graph",
    force: bool = True,
) -> str:
```

- Reads cell centroids from `sj.cells` (x, y columns).
- Uses `scipy.spatial.cKDTree` for efficient KNN query — O(N log N).
- Writes edge list Parquet to `maps/knn_graph.parquet` with columns: `cell_id_a`, `cell_id_b`, `distance` (Euclidean).
- Edge convention: `cell_id_a < cell_id_b` lexicographically. Duplicate edges from mutual KNN collapsed to single edge.
- Returns `output_key`.

#### `build_radius_graph`

```python
def build_radius_graph(
    sj: s_spatioloji,
    radius: float,
    output_key: str = "radius_graph",
    force: bool = True,
) -> str:
```

- Uses `cKDTree.query_ball_point(r=radius)`.
- Same edge list format: `cell_id_a`, `cell_id_b`, `distance`.
- `radius` is required (no default) — forces the user to consider their dataset's scale.
- Returns `output_key`.

#### `_load_point_graph` (private)

```python
def _load_point_graph(sj: s_spatioloji, graph_key: str = "knn_graph") -> nx.Graph:
```

- Loads edge list from `maps/<graph_key>.parquet` via `sj.maps[graph_key].compute()`.
- Constructs and returns `networkx.Graph` with `cell_id` string nodes and `distance` float edge weights.
- Raises `FileNotFoundError` with message "Point graph not found. Run build_knn_graph() or build_radius_graph() first." if not found.
- **Use only** for functions requiring graph algorithms (BFS, clustering coefficient). Not suitable at 100M cells for vectorized operations.

#### `_load_point_graph_sparse` (private)

```python
def _load_point_graph_sparse(
    sj: s_spatioloji, graph_key: str = "knn_graph"
) -> tuple[scipy.sparse.csr_matrix, np.ndarray]:
```

- Loads edge list from `maps/<graph_key>.parquet`.
- Returns `(adjacency_matrix, cell_ids)` where `adjacency_matrix` is a symmetric binary `csr_matrix` and `cell_ids` is a 1D array mapping matrix indices to cell_id strings.
- Raises `FileNotFoundError` if not found.
- **Primary loader** for scalable operations (Moran's I, Geary's C, neighborhoods, etc.).

---

### neighborhoods.py

#### `neighborhood_composition`

```python
def neighborhood_composition(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "knn_graph",
    weighted: bool = False,
    output_key: str = "pt_nhood_composition",
    force: bool = True,
) -> str:
```

- Loads graph via `_load_point_graph_sparse`.
- Loads cluster labels from `sj.maps[cluster_key]`.
- When `weighted=False`: sparse matrix multiplication `adjacency @ cluster_indicator_matrix` gives counts per cluster per cell in one operation.
- When `weighted=True`: uses inverse-distance weighted adjacency matrix (`1/distance` weights) instead of binary. Closer neighbors contribute more.
- Writes `maps/pt_nhood_composition.parquet`: `cell_id`, one column per cluster label.
- Raises `FileNotFoundError` if graph not built.

#### `nth_order_neighbors`

```python
def nth_order_neighbors(
    sj: s_spatioloji,
    order: int = 2,
    graph_key: str = "knn_graph",
    output_key: str = "pt_nhood_nth_order",
    force: bool = True,
) -> str:
```

- BFS up to `order` hops on graph. Excludes the cell itself. Counts at each order are exclusive (order 2 excludes order 1 neighbors).
- Uses `_load_point_graph` (networkx) — BFS requires graph traversal.
- Writes `maps/pt_nhood_nth_order.parquet`: `cell_id`, columns `n_order_1` through `n_order_{order}`.

#### `neighborhood_diversity`

```python
def neighborhood_diversity(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "knn_graph",
    weighted: bool = False,
    output_key: str = "pt_nhood_diversity",
    force: bool = True,
) -> str:
```

- Computes Shannon entropy (`-sum(p_i * log(p_i))`) and Gini-Simpson index (`1 - sum(p_i^2)`) of neighbor cluster composition.
- When `weighted=True`: proportions computed from inverse-distance weights instead of raw counts.
- Writes `maps/pt_nhood_diversity.parquet`: `cell_id`, `shannon`, `simpson`.

---

### patterns.py

#### `colocalization`

```python
def colocalization(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "knn_graph",
    output_key: str = "pt_colocalization",
    force: bool = True,
) -> str:
```

- For each pair of cluster labels, counts observed contact frequency vs expected.
- Expected value formula: `expected_ij = 2 * n_i * n_j * total_edges / (n_total * (n_total - 1))` for `i != j`. For `i == j`: `expected_ii = n_i * (n_i - 1) * total_edges / (n_total * (n_total - 1))`.
- `ratio = observed / expected` (clamped: if expected == 0, ratio = 0). `log2_ratio = log2(ratio)` (nan if ratio == 0).
- Uses `_load_point_graph_sparse` for vectorized edge counting.
- Writes `maps/pt_colocalization.parquet`: `cluster_a`, `cluster_b`, `observed`, `expected`, `ratio`, `log2_ratio`.

#### `morans_i`

```python
def morans_i(
    sj: s_spatioloji,
    feature_key: str = "leiden",
    graph_key: str = "knn_graph",
    output_key: str = "pt_morans_i",
    force: bool = True,
) -> str:
```

- Spatial autocorrelation on a numeric feature. If `feature_key` points to a categorical column (detected by dtype object/category), computes one-hot indicators and returns one row per category.
- Uses **binary** adjacency from `_load_point_graph_sparse` (0/1, ignoring distance).
- Moran's I formula: `I = (N / W) * (sum_ij w_ij (x_i - x_bar)(x_j - x_bar)) / (sum_i (x_i - x_bar)^2)`. Computed as sparse matrix-vector multiplication: `z.T @ W @ z / (z.T @ z)` scaled by `N/W`, where `z = x - x_bar`.
- `expected_I = -1 / (N - 1)`. Z-score uses randomization assumption variance. P-value two-sided from normal approximation.
- Writes `maps/pt_morans_i.parquet`: `feature`, `I`, `expected_I`, `z_score`, `p_value`.

#### `gearys_c`

```python
def gearys_c(
    sj: s_spatioloji,
    feature_key: str = "leiden",
    graph_key: str = "knn_graph",
    output_key: str = "pt_gearys_c",
    force: bool = True,
) -> str:
```

- Same interface and categorical handling as `morans_i`. Uses binary adjacency.
- Geary's C formula: `C = ((N-1) / (2*W)) * (sum_ij w_ij (x_i - x_j)^2) / (sum_i (x_i - x_bar)^2)`. Vectorized via sparse operations: for each edge, compute `(x_i - x_j)^2` using edge list arrays.
- `expected_C = 1.0`. Z-score uses randomization assumption variance. P-value two-sided from normal approximation.
- Writes `maps/pt_gearys_c.parquet`: `feature`, `C`, `expected_C`, `z_score`, `p_value`.

#### `clustering_coefficient`

```python
def clustering_coefficient(
    sj: s_spatioloji,
    graph_key: str = "knn_graph",
    output_key: str = "pt_clustering_coeff",
    force: bool = True,
) -> str:
```

- Per-cell local clustering coefficient from networkx.
- Uses `_load_point_graph` (networkx required for this algorithm).
- Writes `maps/pt_clustering_coeff.parquet`: `cell_id`, `clustering_coeff`.

#### `getis_ord_gi`

```python
def getis_ord_gi(
    sj: s_spatioloji,
    feature_key: str,
    graph_key: str = "knn_graph",
    star: bool = True,
    output_key: str = "pt_getis_ord",
    force: bool = True,
) -> str:
```

- Computes Gi* (default) or Gi statistic per cell for a numeric feature.
- `feature_key` is required (no default) — must point to a numeric maps key.
- If `feature_key` points to a categorical column, raises `ValueError` ("Getis-Ord Gi* requires a numeric feature.").
- Gi* formula: `Gi*(i) = (sum_j w_ij * x_j - x_bar * sum_j w_ij) / (S * sqrt((N * sum_j w_ij^2 - (sum_j w_ij)^2) / (N-1)))` where `S` is global std, `x_bar` is global mean, and `w_ij` includes `j=i` for star variant.
- `star=True`: includes the cell itself in the local sum (Gi*). `star=False`: excludes it (Gi).
- Z-scores are the statistic itself. P-value two-sided from normal approximation.
- Fully vectorized via sparse matrix operations from `_load_point_graph_sparse`.
- Writes `maps/pt_getis_ord.parquet`: `cell_id`, `gi_stat`, `p_value`.

---

### ripley.py

All four functions are coordinate-based (no graph dependency). Each supports per-cluster computation via `cluster_key`. Uses `scipy.spatial.cKDTree` for O(N log N) distance queries.

**Common parameters:**
- `radii: np.ndarray | None = None` — explicit evaluation distances. If `None`, auto-generates `n_radii` evenly spaced values.
- `n_radii: int = 50` — number of radii to auto-generate.
- `max_radius: float | None = None` — if `None`, defaults to 1/4 of the shorter bounding box side (standard convention to limit edge effects).
- `cluster_key: str | None = None` — if `None`, computes globally. If provided, computes per cluster label.

**Edge correction:** Ripley's border correction — each point's contribution weighted by `1 / fraction_of_circle_inside_study_area`. Vectorized: rectangular bounding box clipping is min/max arithmetic on numpy arrays.

#### `ripley_k`

```python
def ripley_k(
    sj: s_spatioloji,
    radii: np.ndarray | None = None,
    n_radii: int = 50,
    max_radius: float | None = None,
    cluster_key: str | None = None,
    output_key: str = "pt_ripley_k",
    force: bool = True,
) -> str:
```

- `K(r) = (A / N^2) * sum_i sum_{j!=i} w_ij * 1(d_ij <= r)` where `A` is study area, `N` is point count, `w_ij` is edge correction weight.
- Uses `cKDTree.count_neighbors(other, r)` for O(N log N) pair counting per radius.
- Writes `maps/pt_ripley_k.parquet`: `cluster` (or `"all"`), `r`, `K`, `K_theo` (CSR expectation: `pi * r^2`).

#### `ripley_l`

```python
def ripley_l(
    sj: s_spatioloji,
    radii: np.ndarray | None = None,
    n_radii: int = 50,
    max_radius: float | None = None,
    cluster_key: str | None = None,
    output_key: str = "pt_ripley_l",
    force: bool = True,
) -> str:
```

- `L(r) = sqrt(K(r) / pi) - r`. Variance-stabilized K. Under CSR, L(r) = 0.
- Computes K internally then transforms.
- Writes `maps/pt_ripley_l.parquet`: `cluster`, `r`, `L`.

#### `ripley_g`

```python
def ripley_g(
    sj: s_spatioloji,
    radii: np.ndarray | None = None,
    n_radii: int = 50,
    max_radius: float | None = None,
    cluster_key: str | None = None,
    output_key: str = "pt_ripley_g",
    force: bool = True,
) -> str:
```

- Nearest-neighbor distance function. `G(r) = fraction of points whose NN distance <= r`.
- Computed from `cKDTree.query(k=2)` (k=2 because k=1 is self).
- Writes `maps/pt_ripley_g.parquet`: `cluster`, `r`, `G`, `G_theo` (CSR: `1 - exp(-lambda * pi * r^2)` where `lambda = N/A`).

#### `ripley_f`

```python
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
```

- Empty-space function. `F(r) = fraction of random reference points whose nearest data point is within r`.
- Generates `n_random` uniform random points within the bounding box, queries nearest data point distance via cKDTree.
- Writes `maps/pt_ripley_f.parquet`: `cluster`, `r`, `F`, `F_theo` (same theoretical as G under CSR).

**Output convention:** Each function writes one Parquet with multiple rows per radius value (long format). When `cluster_key` is set, the `cluster` column distinguishes groups; when `None`, `cluster = "all"`.

---

### statistics.py

#### `permutation_test`

```python
def permutation_test(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "knn_graph",
    n_permutations: int = 1000,
    random_state: int = 42,
    output_key: str = "pt_permutation_test",
    force: bool = True,
) -> str:
```

- Permutes cluster labels using `np.random.default_rng(random_state)`, recomputes colocalization ratios each time.
- Uses `_load_point_graph_sparse` for O(N*k) per permutation.
- Writes `maps/pt_permutation_test.parquet`: `cluster_a`, `cluster_b`, `observed_ratio`, `p_value`, `z_score`.

#### `quadrat_density`

```python
def quadrat_density(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    n_bins: int = 10,
    output_key: str = "pt_quadrat_density",
    force: bool = True,
) -> str:
```

- Divides spatial extent into `n_bins x n_bins` grid quadrats.
- Counts cells per cluster per quadrat using `np.histogram2d`. Chi-squared test for spatial uniformity.
- Writes `maps/pt_quadrat_density.parquet`: `cluster`, `chi2`, `p_value`, `density_mean`, `density_std`.
- Does NOT depend on graph (uses cell coordinates from `sj.cells`).

#### `clark_evans`

```python
def clark_evans(
    sj: s_spatioloji,
    cluster_key: str | None = None,
    output_key: str = "pt_clark_evans",
    force: bool = True,
) -> str:
```

- Nearest-neighbor index: `R = r_obs / r_exp` where `r_obs` is mean NN distance and `r_exp = 1 / (2 * sqrt(lambda))` with `lambda = N/A` (intensity).
- R < 1 = clustered, R = 1 = random, R > 1 = dispersed.
- Z-test: `z = (r_obs - r_exp) / sigma_r` where `sigma_r = 0.26136 / sqrt(N * lambda)`.
- When `cluster_key` is provided, computes per cluster. When `None`, computes globally.
- Uses `cKDTree.query(k=2)` — O(N log N).
- Writes `maps/pt_clark_evans.parquet`: `cluster` (or `"all"`), `R`, `r_observed`, `r_expected`, `z_score`, `p_value`.

#### `dclf_envelope`

```python
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
```

- `function`: one of `"K"`, `"L"`, `"G"`, `"F"` — which Ripley function to test against CSR.
- Generates `n_simulations` CSR point patterns (uniform random in bounding box, same N), computes the chosen Ripley function for each.
- Builds pointwise min/max envelope. Global envelope test p-value (DCLF: rank of observed max absolute deviation among simulation max absolute deviations).
- `max_cells`: subsampling threshold. If N > `max_cells`, randomly samples `max_cells` cells before computing. Ripley's functions are statistical summaries that converge well under subsampling.
- Uses `cKDTree.count_neighbors` for K/L simulations, `cKDTree.query` for G/F — all O(N log N) per simulation.
- Writes `maps/pt_dclf_envelope.parquet`: `cluster`, `r`, `observed`, `lo` (envelope lower), `hi` (envelope upper), `theo` (CSR expectation), `p_value` (one global p-value repeated per row for convenience).

---

## Design Decisions

1. **All outputs to `maps/` as Parquet** — consistent with compute layer contract, no mutation of cells.parquet.
2. **`pt_` prefix on all output keys** — avoids collision with polygon module outputs that share similar function names.
3. **KNN graph as edge list Parquet** — persisted for caching, loaded as sparse matrix or networkx depending on downstream need.
4. **Dual graph loaders** — `_load_point_graph` (networkx) for algorithms requiring graph traversal; `_load_point_graph_sparse` (scipy CSR) for vectorized linear algebra at scale.
5. **`weighted` parameter on neighborhoods** — distance weighting is natural for point-based analysis where distance is a continuous variable (unlike polygon where contact is binary).
6. **`radius` required (no default) in `build_radius_graph`** — forces users to think about spatial scale of their dataset.
7. **Ripley's functions use cKDTree directly** — no graph dependency, pure coordinate-based statistics.
8. **`max_cells` subsampling in `dclf_envelope`** — Monte Carlo simulations at 100M cells are prohibitive; subsampling is statistically valid for summary functions.
9. **All functions use `force` parameter** — consistent with compute layer skip-if-exists pattern.
10. **Core dependencies only** — scipy, networkx, numpy are all core deps. No optional dependency guards needed.

## Scalability Notes

At 50–100M cells:
- **Graph building:** `cKDTree` KNN query is O(N log N). Edge list at 100M × k=10 = ~1B edges fits in Parquet.
- **Sparse adjacency:** `csr_matrix` with 1B nonzeros uses ~12 GB RAM (8 bytes data + 4 bytes index per entry). Feasible on research workstations.
- **Moran's/Geary's/Gi*:** Sparse matrix-vector multiplication O(N*k). Fully vectorized.
- **Neighborhoods:** Sparse matrix × indicator matrix in one operation. No per-cell loops.
- **Ripley's:** `cKDTree.count_neighbors` is O(N log N) per radius. Edge correction vectorized.
- **DCLF:** Subsampled to `max_cells` (default 100K). 199 simulations × O(100K log 100K) is fast.
- **Permutation test:** O(N*k) per permutation via sparse matrix. 1000 permutations at 100M cells is feasible.
- **nth_order / clustering_coefficient:** These require networkx (graph algorithms). At 100M cells, networkx graph construction will be slow and memory-heavy. Consider warning or chunking for very large datasets.

## Testing Strategy

- **Shared fixtures:** Reuse existing `sj` fixture (200 cells, 20×10 grid). Add `sj_with_knn_graph` fixture with pre-built KNN graph.
- **`sj_with_knn_graph`:** `sj` + `maps/knn_graph.parquet` pre-built with k=6.
- Each function gets: return key test, output written test, column presence test, force=False skip test.
- **Graph tests:** verify edge count, symmetry (no directed edges), distance > 0.
- **Neighborhood tests:** verify weighted vs unweighted produce different values.
- **Ripley's tests:** validate against known arrangements — regular grid should show L(r) < 0 (dispersed), clustered points should show L(r) > 0.
- **Statistics tests:** Clark-Evans R ≈ 1 for regular grid fixture, permutation test p-values in [0, 1].
- **Getis-Ord:** validate that cells in high-value clusters get positive Gi* z-scores.
