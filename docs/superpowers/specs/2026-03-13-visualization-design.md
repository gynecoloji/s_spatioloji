# Visualization Module Design Spec

**Date:** 2026-03-13
**Module:** `src/s_spatioloji/visualization/`
**Status:** Approved

## Goal

Implement a comprehensive visualization module for image-based spatial transcriptomics. Covers embedding plots, spatial maps (with optional tissue image overlay), gene expression summaries, and spatial analysis result plots.

## Architecture

All visualization functions follow a hybrid convention:

- Accept `sj: s_spatioloji` as first argument.
- Return `matplotlib.axes.Axes` (or `matplotlib.figure.Figure` for multi-panel plots like `violin`).
- Auto-save figure to `{dataset_root}/figures/{function_name}.png` when `save=True`.
- Optional `ax` parameter to plot on existing axes (skip figure creation).
- Optional `save_path: Path | str | None` to override save location.
- `save: bool = True` controls auto-save behavior.

**Figure saving:** All functions call `_save_figure()` from `_common.py`. This creates `figures/` directory in the dataset root if it doesn't exist, saves as `{name}.{fmt}` with configurable dpi and format.

**Large dataset handling:** Two strategies, both optional:
- `max_cells: int = 100_000` — random subsampling if N exceeds threshold.
- `xlim: tuple | None`, `ylim: tuple | None` — bounding box crop (spatial functions only).
Both are applied before plotting. Subsampling uses `np.random.default_rng(42)` for reproducibility.

**Color conventions:**
- Categorical: curated publication-friendly palette (distinguishable, colorblind-accessible). Not `tab20`.
- Continuous expression: `"magma"` (default, overridable via `cmap`).
- Diverging statistics: `"RdBu_r"` (for log2 ratios, z-scores).
- All continuous plots support `vmin`/`vmax` for manual color scale control.
- All categorical plots support `palette` override (list of hex colors).

**Dependencies:** matplotlib, seaborn (for violin), tifffile (for multi-page TIF reading). These should be added to `pyproject.toml` as core dependencies.

## Module Structure

```
src/s_spatioloji/visualization/
├── __init__.py          # Re-exports all public functions (lazy import pattern)
├── _common.py           # Shared helpers (figure management, subsampling, palettes)
├── embedding.py         # Embedding scatter plots (UMAP, PCA, tSNE)
├── spatial.py           # Spatial coordinate plots (dot, polygon, image overlay)
├── expression.py        # Gene expression summary plots (heatmap, violin, dotplot)
└── analysis.py          # Spatial analysis result plots (Ripley, colocalization, etc.)
```

## Function Specifications

### _common.py (private helpers)

#### `_setup_ax`

```python
def _setup_ax(ax: Axes | None, figsize: tuple[float, float] = (8, 6)) -> tuple[Figure, Axes]:
```

- If `ax` is None, creates `fig, ax = plt.subplots(figsize=figsize)`.
- If `ax` is provided, gets its parent figure via `ax.get_figure()`.
- Returns `(fig, ax)`.

#### `_save_figure`

```python
def _save_figure(
    fig: Figure,
    sj: s_spatioloji,
    name: str,
    save: bool = True,
    save_path: Path | str | None = None,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
```

- If `save` is False, does nothing.
- If `save_path` is provided, saves to that exact path.
- Otherwise, saves to `{sj.config.root}/figures/{name}.{fmt}`.
- Creates `figures/` directory with `mkdir(exist_ok=True)`.
- Calls `fig.savefig(path, dpi=dpi, bbox_inches="tight")`.

#### `_subsample`

```python
def _subsample(
    df: pd.DataFrame,
    max_cells: int = 100_000,
    random_state: int = 42,
) -> pd.DataFrame:
```

- If `len(df) <= max_cells`, returns `df` unchanged.
- Otherwise, returns `df.sample(n=max_cells, random_state=random_state)`.

#### `_filter_bbox`

```python
def _filter_bbox(
    df: pd.DataFrame,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> pd.DataFrame:
```

- Filters rows where `x` is within `xlim` and `y` is within `ylim`.
- If both are None, returns `df` unchanged.

#### `_get_feature_values`

```python
def _get_feature_values(
    sj: s_spatioloji,
    color_by: str,
    cell_ids: np.ndarray,
) -> tuple[np.ndarray, bool]:
```

- Resolves `color_by` to an array of values aligned to `cell_ids`.
- Lookup order: (1) check `sj.maps[color_by]` — load the non-cell_id column, (2) check if `color_by` is a gene name in `sj.expression.gene_names` — extract expression vector, (3) check if `color_by` is a column in `sj.cells.df` — extract that column.
- Returns `(values, is_categorical)` where `is_categorical` is True if dtype is object/category or integer with fewer than 20 unique values.
- Raises `ValueError` if `color_by` cannot be resolved.

#### `_categorical_palette`

```python
def _categorical_palette(n: int) -> list[str]:
```

- Returns a list of `n` hex color strings from a curated publication-friendly palette.
- Uses a hand-picked set of ~20 distinguishable, colorblind-accessible colors.
- If `n > len(palette)`, cycles through the palette.

#### `_continuous_cmap`

```python
def _continuous_cmap(name: str = "magma") -> Colormap:
```

- Returns `matplotlib.colormaps[name]`.
- Convenience wrapper for consistency.

---

### embedding.py

#### `scatter`

```python
def scatter(
    sj: s_spatioloji,
    basis: str = "X_umap",
    color_by: str = "leiden",
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    palette: list[str] | None = None,
    max_cells: int = 100_000,
    figsize: tuple[float, float] = (8, 6),
    point_size: float = 5,
    alpha: float = 1.0,
    ax: Axes | None = None,
    save: bool = True,
    save_path: Path | str | None = None,
) -> Axes:
```

- Loads 2D coordinates from `sj.maps[basis]` — expects columns like `{basis}_1`, `{basis}_2` or first two non-cell_id columns.
- Resolves `color_by` via `_get_feature_values`.
- Categorical: `ax.scatter()` with one call per category, using `palette` or `_categorical_palette(n)`. Adds legend outside plot (`bbox_to_anchor=(1.05, 1)`).
- Continuous: single `ax.scatter()` with colormap. Adds colorbar.
- Subsamples if N > `max_cells`.
- Axis labels: column names from the basis DataFrame (e.g., "UMAP_1", "UMAP_2").
- Removes axis ticks and spines for clean embedding look.
- Saves as `scatter_{basis}_{color_by}.{fmt}`.

---

### spatial.py

#### `spatial_scatter`

```python
def spatial_scatter(
    sj: s_spatioloji,
    color_by: str = "leiden",
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    palette: list[str] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    max_cells: int = 100_000,
    image_path: Path | str | None = None,
    image_alpha: float = 0.5,
    figsize: tuple[float, float] = (10, 10),
    point_size: float = 3,
    alpha: float = 1.0,
    ax: Axes | None = None,
    save: bool = True,
    save_path: Path | str | None = None,
) -> Axes:
```

- Reads cell coordinates from `sj.cells.df` (x, y columns).
- Applies `_filter_bbox` then `_subsample`.
- If `image_path` is provided: loads TIF via `tifffile.imread`, displays with `ax.imshow(image, alpha=image_alpha, extent=[x_min, x_max, y_max, y_min])`. The extent maps image pixels to spatial coordinates. For multi-page TIFs, reads the first page.
- Overlays scatter plot on top.
- Categorical/continuous coloring same as `scatter`.
- Sets `ax.set_aspect("equal")` for correct spatial proportions.
- Inverts y-axis (`ax.invert_yaxis()`) to match image convention (origin top-left).
- Saves as `spatial_scatter_{color_by}.{fmt}`.

#### `spatial_polygons`

```python
def spatial_polygons(
    sj: s_spatioloji,
    color_by: str = "leiden",
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    palette: list[str] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    max_cells: int = 100_000,
    image_path: Path | str | None = None,
    image_alpha: float = 0.5,
    edgecolor: str = "black",
    linewidth: float = 0.3,
    figsize: tuple[float, float] = (10, 10),
    alpha: float = 0.7,
    ax: Axes | None = None,
    save: bool = True,
    save_path: Path | str | None = None,
) -> Axes:
```

- Loads boundaries from `sj.boundaries.load()` (GeoDataFrame).
- Filters by bounding box using `sj.boundaries.query_bbox(xlim, ylim)` if provided, otherwise uses full extent then subsamples.
- Resolves `color_by` values, maps to face colors.
- Uses `matplotlib.collections.PatchCollection` with `shapely` geometry conversion for efficient polygon rendering, or `geopandas.GeoDataFrame.plot()`.
- Image overlay same as `spatial_scatter`.
- Saves as `spatial_polygons_{color_by}.{fmt}`.

#### `spatial_expression`

```python
def spatial_expression(
    sj: s_spatioloji,
    gene: str,
    mode: str = "scatter",
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    max_cells: int = 100_000,
    image_path: Path | str | None = None,
    image_alpha: float = 0.5,
    figsize: tuple[float, float] = (10, 10),
    point_size: float = 3,
    ax: Axes | None = None,
    save: bool = True,
    save_path: Path | str | None = None,
) -> Axes:
```

- Convenience wrapper: calls `spatial_scatter(sj, color_by=gene, ...)` if `mode="scatter"`, or `spatial_polygons(sj, color_by=gene, ...)` if `mode="polygon"`.
- `gene` is required (no default). Raises `ValueError` if gene not found in `sj.expression.gene_names`.
- Saves as `spatial_expression_{gene}.{fmt}`.

---

### expression.py

#### `heatmap`

```python
def heatmap(
    sj: s_spatioloji,
    genes: list[str],
    cluster_key: str = "leiden",
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    standardize: bool = True,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    save: bool = True,
    save_path: Path | str | None = None,
) -> Axes:
```

- Computes mean expression per cluster for each gene.
- Loads expression for `genes` from `sj.expression`, loads cluster labels from `sj.maps[cluster_key]`.
- If `standardize=True`, z-scores each gene across clusters (subtract mean, divide by std).
- Plots with `ax.imshow()` or `ax.pcolormesh()`. X-axis = clusters, Y-axis = genes.
- Adds colorbar. Tick labels for both axes.
- Auto-sizes figure if `figsize=None`: width based on n_clusters, height based on n_genes.
- Saves as `heatmap_{cluster_key}.{fmt}`.

#### `violin`

```python
def violin(
    sj: s_spatioloji,
    genes: list[str],
    cluster_key: str = "leiden",
    palette: list[str] | None = None,
    ncols: int = 4,
    figsize: tuple[float, float] | None = None,
    save: bool = True,
    save_path: Path | str | None = None,
) -> Figure:
```

- Returns `Figure` (not Axes) because it creates a multi-panel grid.
- One subplot per gene, arranged in `ceil(n_genes / ncols)` rows × `ncols` columns.
- Uses `seaborn.violinplot` within each subplot.
- X-axis = cluster labels, Y-axis = expression values.
- Auto-sizes figure if `figsize=None`.
- No `ax` parameter (creates its own figure).
- Saves as `violin_{cluster_key}.{fmt}`.

#### `dotplot`

```python
def dotplot(
    sj: s_spatioloji,
    genes: list[str],
    cluster_key: str = "leiden",
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    max_dot_size: float = 200,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    save: bool = True,
    save_path: Path | str | None = None,
) -> Axes:
```

- X-axis = genes, Y-axis = clusters.
- For each gene × cluster: dot size = fraction of cells with expression > 0, dot color = mean expression of expressing cells.
- Uses `ax.scatter()` with computed sizes and colors.
- Adds size legend (e.g., 25%, 50%, 75%, 100%) and colorbar.
- Auto-sizes figure if `figsize=None`.
- Saves as `dotplot_{cluster_key}.{fmt}`.

---

### analysis.py

#### `ripley_plot`

```python
def ripley_plot(
    sj: s_spatioloji,
    function: str = "K",
    maps_key: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    ax: Axes | None = None,
    save: bool = True,
    save_path: Path | str | None = None,
) -> Axes:
```

- If `maps_key` is None, resolves to `pt_ripley_{function.lower()}`.
- Loads long-format DataFrame from `sj.maps[maps_key]`.
- Plots one line per cluster (or single line if `cluster="all"`).
- For K: also plots `K_theo` as dashed line. For L: horizontal line at 0. For G/F: plots `G_theo`/`F_theo` as dashed.
- Legend shows cluster names. X-axis = "r", Y-axis = function name.
- Saves as `ripley_{function.lower()}.{fmt}`.

#### `colocalization_heatmap`

```python
def colocalization_heatmap(
    sj: s_spatioloji,
    maps_key: str = "pt_colocalization",
    metric: str = "log2_ratio",
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    save: bool = True,
    save_path: Path | str | None = None,
) -> Axes:
```

- Loads colocalization table, pivots `cluster_a` × `cluster_b` into a symmetric matrix using `metric` column.
- Fills diagonal with the self-colocalization values.
- Plots with `ax.imshow()`. Adds colorbar.
- Tick labels = cluster names.
- Auto-sizes figure based on number of clusters.
- Saves as `colocalization_{metric}.{fmt}`.

#### `neighborhood_bar`

```python
def neighborhood_bar(
    sj: s_spatioloji,
    maps_key: str = "pt_nhood_composition",
    cluster_key: str = "leiden",
    normalize: bool = True,
    palette: list[str] | None = None,
    figsize: tuple[float, float] | None = None,
    ax: Axes | None = None,
    save: bool = True,
    save_path: Path | str | None = None,
) -> Axes:
```

- Loads neighborhood composition table. Groups by cluster (from `cluster_key`), computes mean composition per cluster.
- If `normalize=True`, divides each row by its sum to show proportions.
- Stacked horizontal bar chart: Y-axis = source cluster, segments = neighbor cluster proportions.
- Legend for neighbor cluster colors.
- Saves as `neighborhood_bar.{fmt}`.

#### `envelope_plot`

```python
def envelope_plot(
    sj: s_spatioloji,
    maps_key: str = "pt_dclf_envelope",
    figsize: tuple[float, float] = (8, 6),
    ax: Axes | None = None,
    save: bool = True,
    save_path: Path | str | None = None,
) -> Axes:
```

- Loads envelope DataFrame. Plots per cluster (or single if `cluster="all"`).
- Observed line (solid), theoretical line (dashed), simulation band as `ax.fill_between(r, lo, hi, alpha=0.2)`.
- Shows p-value in plot title or annotation.
- Saves as `envelope_{maps_key}.{fmt}`.

---

### `__init__.py`

Lazy import pattern matching polygon/point modules:

```python
_LAZY_IMPORTS = {
    "scatter": ("s_spatioloji.visualization.embedding", "scatter"),
    "spatial_scatter": ("s_spatioloji.visualization.spatial", "spatial_scatter"),
    "spatial_polygons": ("s_spatioloji.visualization.spatial", "spatial_polygons"),
    "spatial_expression": ("s_spatioloji.visualization.spatial", "spatial_expression"),
    "heatmap": ("s_spatioloji.visualization.expression", "heatmap"),
    "violin": ("s_spatioloji.visualization.expression", "violin"),
    "dotplot": ("s_spatioloji.visualization.expression", "dotplot"),
    "ripley_plot": ("s_spatioloji.visualization.analysis", "ripley_plot"),
    "colocalization_heatmap": ("s_spatioloji.visualization.analysis", "colocalization_heatmap"),
    "neighborhood_bar": ("s_spatioloji.visualization.analysis", "neighborhood_bar"),
    "envelope_plot": ("s_spatioloji.visualization.analysis", "envelope_plot"),
}
```

---

## Design Decisions

1. **Hybrid return convention** — return Axes/Figure AND auto-save to `figures/`. Supports both interactive Jupyter use and batch pipeline use.
2. **`figures/` directory** — separate from `maps/` because these are visual outputs, not data. Not managed by the Maps accessor.
3. **Publication-friendly palette** — not `tab20` (too garish). Hand-picked distinguishable colors.
4. **`max_cells` + bbox** — two complementary strategies for large datasets. Bbox first (spatial locality), then subsample (global).
5. **`_get_feature_values` resolution order** — maps → gene name → cells column. Most specific first.
6. **`tifffile` for image reading** — handles multi-page TIFs, 16-bit images, and large files better than `plt.imread`.
7. **`seaborn` only for violin** — minimal dependency surface. Everything else uses raw matplotlib.
8. **`spatial_expression` as convenience wrapper** — avoids duplicating spatial plot logic. Just delegates to scatter/polygon with gene as `color_by`.
9. **`violin` returns Figure** — multi-panel layout requires Figure-level control. All other functions return Axes.
10. **Lazy imports in `__init__.py`** — consistent with polygon/point modules. Avoids importing matplotlib at package import time.

## Dependencies

Add to `pyproject.toml` under `[project] dependencies`:
- `matplotlib >= 3.7`
- `seaborn >= 0.13`
- `tifffile >= 2023.1`

## Testing Strategy

- **Shared fixtures:** Reuse existing `sj` fixture (200 cells, 20×10 grid). Add `sj_with_embeddings` fixture with synthetic 2D UMAP coordinates.
- **`sj_with_embeddings`:** `sj` + `maps/X_umap.parquet` with two columns (`UMAP_1`, `UMAP_2`) from random 2D coordinates.
- Each function gets: returns Axes/Figure test, figure saved test, force save=False skips test.
- **Spatial tests:** verify `ax.get_xlim()`/`ax.get_ylim()` when xlim/ylim set.
- **Expression tests:** verify heatmap shape matches n_genes × n_clusters.
- **Analysis tests:** verify lines plotted on axes (check `ax.get_lines()` count).
- **Image overlay test:** create a small synthetic numpy array, save as TIF, verify `imshow` called.
- Use `matplotlib.use("Agg")` backend in tests (no display needed).
- Close all figures after each test with `plt.close("all")` to prevent memory leaks.
