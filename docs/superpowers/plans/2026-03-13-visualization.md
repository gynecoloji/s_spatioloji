# Visualization Module Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a visualization module with embedding plots, spatial maps (dot/polygon/image overlay), expression summaries (heatmap/violin/dotplot), and spatial analysis result plots.

**Architecture:** All functions accept `sj: s_spatioloji`, return `matplotlib.axes.Axes` (or `Figure` for violin), auto-save to `{dataset_root}/figures/`. Shared helpers in `_common.py` handle figure setup, saving, subsampling, bbox filtering, feature resolution, and color palettes.

**Tech Stack:** matplotlib, tifffile, seaborn (optional, guard-imported for violin only), numpy, pandas, geopandas

**Spec:** `docs/superpowers/specs/2026-03-13-visualization-design.md`

**Reference:** `src/s_spatioloji/spatial/polygon/__init__.py` (lazy import pattern)

---

## File Structure

```
src/s_spatioloji/visualization/
├── __init__.py          # Lazy imports of all 11 public functions
├── _common.py           # _setup_ax, _save_figure, _subsample, _filter_bbox, _get_feature_values, _categorical_palette, _continuous_cmap
├── embedding.py         # scatter
├── spatial.py           # spatial_scatter, spatial_polygons, spatial_expression
├── expression.py        # heatmap, violin, dotplot
└── analysis.py          # ripley_plot, colocalization_heatmap, neighborhood_bar, envelope_plot

tests/unit/
├── conftest.py          # Add sj_with_embeddings, sj_with_viz_clusters fixtures
├── test_visualization_common.py
├── test_visualization_embedding.py
├── test_visualization_spatial.py
├── test_visualization_expression.py
└── test_visualization_analysis.py
```

---

## Chunk 1: Foundation + Embedding (Tasks 1–3)

### Task 1: Add matplotlib dependency + test fixtures

**Files:**
- Modify: `pyproject.toml` (add matplotlib)
- Modify: `tests/unit/conftest.py` (add visualization fixtures)

- [ ] **Step 1: Add matplotlib to pyproject.toml dependencies**

Add `"matplotlib>=3.7"` to the `dependencies` list in `[project]`.

- [ ] **Step 2: Add visualization fixtures to conftest.py**

Add at end of file:

```python
@pytest.fixture()
def sj_with_embeddings(sj_with_pt_clusters):
    """sj_with_pt_clusters + synthetic UMAP coordinates in maps/X_umap.parquet."""
    rng = np.random.default_rng(42)
    cells_df = sj_with_pt_clusters.cells.df.compute()
    n = len(cells_df)
    umap_df = pd.DataFrame({
        "cell_id": cells_df["cell_id"].values,
        "UMAP_1": rng.standard_normal(n),
        "UMAP_2": rng.standard_normal(n),
    })
    from s_spatioloji.compute import _atomic_write_parquet

    maps_dir = sj_with_pt_clusters.config.root / "maps"
    maps_dir.mkdir(exist_ok=True)
    _atomic_write_parquet(umap_df, maps_dir, "X_umap")
    return sj_with_pt_clusters
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml tests/unit/conftest.py
git commit -m "feat(visualization): add matplotlib dep and visualization test fixtures"
```

### Task 2: _common.py + __init__.py + tests

**Files:**
- Create: `src/s_spatioloji/visualization/__init__.py`
- Create: `src/s_spatioloji/visualization/_common.py`
- Create: `tests/unit/test_visualization_common.py`

- [ ] **Step 1: Write _common.py tests**

Create `tests/unit/test_visualization_common.py`:

```python
"""Unit tests for s_spatioloji.visualization._common."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from s_spatioloji.visualization._common import (
    _categorical_palette,
    _continuous_cmap,
    _filter_bbox,
    _get_feature_values,
    _save_figure,
    _setup_ax,
    _subsample,
)


class TestSetupAx:
    def test_creates_figure_when_none(self):
        fig, ax = _setup_ax(None)
        assert fig is not None
        assert ax is not None
        plt.close("all")

    def test_uses_existing_ax(self):
        _, existing_ax = plt.subplots()
        fig, ax = _setup_ax(existing_ax)
        assert ax is existing_ax
        plt.close("all")

    def test_custom_figsize(self):
        fig, ax = _setup_ax(None, figsize=(12, 4))
        w, h = fig.get_size_inches()
        assert abs(w - 12) < 0.1
        assert abs(h - 4) < 0.1
        plt.close("all")


class TestSaveFigure:
    def test_saves_to_figures_dir(self, sj):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        _save_figure(fig, sj, "test_plot")
        figures_dir = sj.config.root / "figures"
        assert (figures_dir / "test_plot.png").exists()
        plt.close("all")

    def test_save_false_skips(self, sj):
        fig, ax = plt.subplots()
        _save_figure(fig, sj, "test_plot", save=False)
        figures_dir = sj.config.root / "figures"
        assert not (figures_dir / "test_plot.png").exists()
        plt.close("all")

    def test_custom_save_path(self, sj, tmp_path):
        fig, ax = plt.subplots()
        custom = tmp_path / "custom.png"
        _save_figure(fig, sj, "test_plot", save_path=custom)
        assert custom.exists()
        plt.close("all")

    def test_custom_fmt(self, sj):
        fig, ax = plt.subplots()
        _save_figure(fig, sj, "test_plot", fmt="pdf")
        figures_dir = sj.config.root / "figures"
        assert (figures_dir / "test_plot.pdf").exists()
        plt.close("all")


class TestSubsample:
    def test_no_change_when_small(self):
        df = pd.DataFrame({"x": range(10)})
        result = _subsample(df, max_cells=100)
        assert len(result) == 10

    def test_subsamples_when_large(self):
        df = pd.DataFrame({"x": range(1000)})
        result = _subsample(df, max_cells=100)
        assert len(result) == 100

    def test_reproducible(self):
        df = pd.DataFrame({"x": range(1000)})
        r1 = _subsample(df, max_cells=100)
        r2 = _subsample(df, max_cells=100)
        assert r1["x"].tolist() == r2["x"].tolist()


class TestFilterBbox:
    def test_no_filter_when_none(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]})
        result = _filter_bbox(df)
        assert len(result) == 3

    def test_filters_x(self):
        df = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 0, 0, 0]})
        result = _filter_bbox(df, xlim=(0.5, 2.5))
        assert len(result) == 2

    def test_filters_both(self):
        df = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]})
        result = _filter_bbox(df, xlim=(0.5, 2.5), ylim=(0.5, 2.5))
        assert len(result) == 2


class TestGetFeatureValues:
    def test_resolves_maps_key(self, sj_with_embeddings):
        cells_df = sj_with_embeddings.cells.df.compute()
        cell_ids = cells_df["cell_id"].values
        values, is_cat = _get_feature_values(sj_with_embeddings, "leiden", cell_ids)
        assert len(values) == len(cell_ids)

    def test_resolves_gene_name(self, sj_with_embeddings):
        cells_df = sj_with_embeddings.cells.df.compute()
        cell_ids = cells_df["cell_id"].values
        values, is_cat = _get_feature_values(sj_with_embeddings, "gene_0", cell_ids)
        assert len(values) == len(cell_ids)
        assert not is_cat

    def test_unknown_raises(self, sj_with_embeddings):
        cells_df = sj_with_embeddings.cells.df.compute()
        cell_ids = cells_df["cell_id"].values
        with pytest.raises(ValueError, match="Cannot resolve"):
            _get_feature_values(sj_with_embeddings, "nonexistent_thing_xyz", cell_ids)


class TestCategoricalPalette:
    def test_returns_n_colors(self):
        colors = _categorical_palette(5)
        assert len(colors) == 5

    def test_cycles_when_large(self):
        colors = _categorical_palette(30)
        assert len(colors) == 30

    def test_returns_hex_strings(self):
        colors = _categorical_palette(3)
        for c in colors:
            assert c.startswith("#")


class TestContinuousCmap:
    def test_returns_colormap(self):
        cmap = _continuous_cmap("magma")
        assert cmap is not None

    def test_custom_name(self):
        cmap = _continuous_cmap("viridis")
        assert cmap.name == "viridis"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_visualization_common.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Create `__init__.py`**

```python
"""Visualization functions for s_spatioloji.

All plotting functions accept an ``s_spatioloji`` object, return
``matplotlib.axes.Axes`` (or ``Figure`` for multi-panel plots),
and auto-save to ``{dataset_root}/figures/``.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
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

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

- [ ] **Step 4: Create `_common.py`**

```python
"""Shared helpers for visualization functions.

Private module — not exported from ``s_spatioloji.visualization``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colormap import Colormap
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji

# Publication-friendly categorical palette (colorblind-accessible)
_PALETTE = [
    "#4477AA",  # blue
    "#EE6677",  # red
    "#228833",  # green
    "#CCBB44",  # yellow
    "#66CCEE",  # cyan
    "#AA3377",  # purple
    "#BBBBBB",  # grey
    "#EE8866",  # orange
    "#44BB99",  # teal
    "#FFAABB",  # pink
    "#332288",  # indigo
    "#882255",  # wine
    "#117733",  # forest
    "#999933",  # olive
    "#CC6677",  # rose
    "#661100",  # brown
    "#6699CC",  # steel
    "#AA4499",  # magenta
    "#DDCC77",  # sand
    "#88CCEE",  # sky
]


def _setup_ax(
    ax: Axes | None, figsize: tuple[float, float] = (8, 6)
) -> tuple[Figure, Axes]:
    """Create or reuse a matplotlib Axes.

    Args:
        ax: Existing axes to reuse, or None to create new.
        figsize: Figure size if creating new.

    Returns:
        Tuple of (Figure, Axes).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


def _save_figure(
    fig: Figure,
    sj: s_spatioloji,
    name: str,
    save: bool = True,
    save_path: Path | str | None = None,
    dpi: int = 200,
    fmt: str = "png",
) -> None:
    """Save figure to the dataset's ``figures/`` directory.

    Args:
        fig: Matplotlib figure to save.
        sj: Dataset instance (used for root path).
        name: Base filename (without extension).
        save: If False, skip saving.
        save_path: Override path. If provided, saves there instead.
        dpi: Resolution in dots per inch.
        fmt: File format (png, pdf, svg, etc.).
    """
    if not save:
        return

    if save_path is not None:
        path = Path(save_path)
    else:
        figures_dir = sj.config.root / "figures"
        figures_dir.mkdir(exist_ok=True)
        path = figures_dir / f"{name}.{fmt}"

    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")


def _subsample(
    df: pd.DataFrame,
    max_cells: int = 100_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Subsample DataFrame if it exceeds max_cells.

    Args:
        df: Input DataFrame.
        max_cells: Maximum number of rows to keep.
        random_state: Seed for reproducibility.

    Returns:
        Original or subsampled DataFrame.
    """
    if len(df) <= max_cells:
        return df

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(df), max_cells, replace=False)
    indices.sort()
    return df.iloc[indices]


def _filter_bbox(
    df: pd.DataFrame,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Filter DataFrame rows by spatial bounding box.

    Expects columns named ``"x"`` and ``"y"``.

    Args:
        df: Input DataFrame with x, y columns.
        xlim: (x_min, x_max) bounds, or None for no filter.
        ylim: (y_min, y_max) bounds, or None for no filter.

    Returns:
        Filtered DataFrame.
    """
    mask = np.ones(len(df), dtype=bool)
    if xlim is not None:
        mask &= (df["x"].values >= xlim[0]) & (df["x"].values <= xlim[1])
    if ylim is not None:
        mask &= (df["y"].values >= ylim[0]) & (df["y"].values <= ylim[1])
    return df.loc[mask]


def _get_feature_values(
    sj: s_spatioloji,
    color_by: str,
    cell_ids: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Resolve a feature name to an array of values aligned to cell_ids.

    Lookup order: (1) maps key, (2) gene name, (3) cells column.

    Args:
        sj: Dataset instance.
        color_by: Feature name to resolve.
        cell_ids: Cell IDs to align values to.

    Returns:
        Tuple of (values_array, is_categorical).

    Raises:
        ValueError: If ``color_by`` cannot be resolved.
    """
    # 1. Try maps
    if sj.maps.has(color_by):
        df = sj.maps[color_by].compute()
        value_cols = [c for c in df.columns if c != "cell_id"]
        if value_cols:
            col = value_cols[0]
            id_to_val = dict(zip(df["cell_id"], df[col], strict=True))
            values = np.array([id_to_val.get(cid, np.nan) for cid in cell_ids])
            is_cat = values.dtype == object or (
                np.issubdtype(values.dtype, np.integer) and len(np.unique(values[~np.isnan(values.astype(float))])) < 20
            )
            return values, is_cat

    # 2. Try gene name
    gene_names = list(sj.expression.gene_names)
    if color_by in gene_names:
        gene_idx = gene_names.index(color_by)
        # Get cell_id order from expression store
        expr_cell_ids = list(sj.expression.cell_ids)
        expr_values = sj.expression[:, gene_idx].compute().ravel()
        id_to_val = dict(zip(expr_cell_ids, expr_values, strict=True))
        values = np.array([id_to_val.get(cid, 0.0) for cid in cell_ids], dtype=float)
        return values, False

    # 3. Try cells column
    cells_df = sj.cells.df.compute()
    if color_by in cells_df.columns:
        id_to_val = dict(zip(cells_df["cell_id"], cells_df[color_by], strict=True))
        values = np.array([id_to_val.get(cid) for cid in cell_ids])
        is_cat = values.dtype == object or (
            np.issubdtype(values.dtype, np.integer) and len(set(values)) < 20
        )
        return values, is_cat

    raise ValueError(
        f"Cannot resolve '{color_by}'. Not found in maps, gene names, or cells columns."
    )


def _categorical_palette(n: int) -> list[str]:
    """Return n publication-friendly categorical colors.

    Args:
        n: Number of colors needed.

    Returns:
        List of hex color strings.
    """
    return [_PALETTE[i % len(_PALETTE)] for i in range(n)]


def _continuous_cmap(name: str = "magma") -> Colormap:
    """Return a matplotlib colormap by name.

    Args:
        name: Colormap name.

    Returns:
        Matplotlib Colormap object.
    """
    return plt.get_cmap(name)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_visualization_common.py -v`
Expected: ALL PASS

- [ ] **Step 6: Lint and commit**

```bash
ruff check src/s_spatioloji/visualization/ tests/unit/test_visualization_common.py --fix
ruff format src/s_spatioloji/visualization/ tests/unit/test_visualization_common.py
git add src/s_spatioloji/visualization/__init__.py src/s_spatioloji/visualization/_common.py tests/unit/test_visualization_common.py
git commit -m "feat(visualization): add shared helpers and lazy-import __init__"
```

### Task 3: embedding.py (scatter)

**Files:**
- Create: `src/s_spatioloji/visualization/embedding.py`
- Create: `tests/unit/test_visualization_embedding.py`

- [ ] **Step 1: Write embedding tests**

Create `tests/unit/test_visualization_embedding.py`:

```python
"""Unit tests for s_spatioloji.visualization.embedding."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from s_spatioloji.visualization.embedding import scatter


class TestScatter:
    def test_returns_axes(self, sj_with_embeddings):
        ax = scatter(sj_with_embeddings, basis="X_umap", color_by="leiden", save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj_with_embeddings):
        scatter(sj_with_embeddings, basis="X_umap", color_by="leiden")
        figures_dir = sj_with_embeddings.config.root / "figures"
        assert (figures_dir / "scatter_X_umap_leiden.png").exists()
        plt.close("all")

    def test_save_false_skips(self, sj_with_embeddings):
        scatter(sj_with_embeddings, basis="X_umap", color_by="leiden", save=False)
        figures_dir = sj_with_embeddings.config.root / "figures"
        assert not figures_dir.exists() or not (figures_dir / "scatter_X_umap_leiden.png").exists()
        plt.close("all")

    def test_existing_ax(self, sj_with_embeddings):
        _, existing_ax = plt.subplots()
        ax = scatter(sj_with_embeddings, basis="X_umap", color_by="leiden", ax=existing_ax, save=False)
        assert ax is existing_ax
        plt.close("all")

    def test_continuous_color(self, sj_with_embeddings):
        ax = scatter(sj_with_embeddings, basis="X_umap", color_by="gene_0", save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_custom_figsize(self, sj_with_embeddings):
        ax = scatter(sj_with_embeddings, basis="X_umap", color_by="leiden", figsize=(12, 4), save=False)
        fig = ax.get_figure()
        w, h = fig.get_size_inches()
        assert abs(w - 12) < 0.1
        plt.close("all")
```

- [ ] **Step 2: Implement embedding.py**

```python
"""Embedding scatter plots (UMAP, PCA, tSNE).

All plotting functions accept an ``s_spatioloji`` object and return
``matplotlib.axes.Axes``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from s_spatioloji.visualization._common import (
    _categorical_palette,
    _continuous_cmap,
    _get_feature_values,
    _save_figure,
    _setup_ax,
    _subsample,
)

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


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
    dpi: int = 200,
    fmt: str = "png",
) -> Axes:
    """2D scatter plot of embedding coordinates.

    Args:
        sj: Dataset instance.
        basis: Maps key for 2D coordinates (e.g., ``"X_umap"``).
        color_by: Feature to color by (maps key, gene name, or cells column).
        cmap: Colormap for continuous features.
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.
        palette: Override categorical palette (list of hex colors).
        max_cells: Subsample threshold.
        figsize: Figure size.
        point_size: Marker size.
        alpha: Marker transparency.
        ax: Existing axes to plot on.
        save: If True, auto-save to figures/.
        save_path: Override save location.
        dpi: Output resolution.
        fmt: Output format.

    Returns:
        The matplotlib Axes with the scatter plot.

    Example:
        >>> scatter(sj, basis="X_umap", color_by="leiden")
    """
    fig, ax = _setup_ax(ax, figsize)

    # Load coordinates
    coord_df = sj.maps[basis].compute()
    coord_cols = [c for c in coord_df.columns if c != "cell_id"]
    x_col, y_col = coord_cols[0], coord_cols[1]

    # Subsample
    coord_df = _subsample(coord_df, max_cells)
    cell_ids = coord_df["cell_id"].values
    x = coord_df[x_col].values
    y = coord_df[y_col].values

    # Resolve feature
    values, is_cat = _get_feature_values(sj, color_by, cell_ids)

    if is_cat:
        categories = sorted(set(values), key=str)
        colors = palette or _categorical_palette(len(categories))
        for i, cat in enumerate(categories):
            mask = values == cat
            ax.scatter(x[mask], y[mask], c=colors[i % len(colors)], s=point_size,
                       alpha=alpha, label=str(cat), edgecolors="none", rasterized=True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, markerscale=2)
    else:
        vals = values.astype(float)
        sc = ax.scatter(x, y, c=vals, cmap=cmap, s=point_size, alpha=alpha,
                        vmin=vmin, vmax=vmax, edgecolors="none", rasterized=True)
        plt.colorbar(sc, ax=ax, shrink=0.6, label=color_by)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    _save_figure(fig, sj, f"scatter_{basis}_{color_by}", save=save, save_path=save_path, dpi=dpi, fmt=fmt)
    return ax
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/unit/test_visualization_embedding.py -v`
Expected: ALL PASS

- [ ] **Step 4: Lint and commit**

```bash
ruff check src/s_spatioloji/visualization/embedding.py tests/unit/test_visualization_embedding.py --fix
ruff format src/s_spatioloji/visualization/embedding.py tests/unit/test_visualization_embedding.py
git add src/s_spatioloji/visualization/embedding.py tests/unit/test_visualization_embedding.py
git commit -m "feat(visualization): add embedding scatter plot"
```

---

## Chunk 2: Spatial Plots (Task 4)

### Task 4: spatial.py (spatial_scatter, spatial_polygons, spatial_expression)

**Files:**
- Create: `src/s_spatioloji/visualization/spatial.py`
- Create: `tests/unit/test_visualization_spatial.py`

- [ ] **Step 1: Write spatial tests**

Create `tests/unit/test_visualization_spatial.py`:

```python
"""Unit tests for s_spatioloji.visualization.spatial."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tifffile

from s_spatioloji.visualization.spatial import (
    spatial_expression,
    spatial_polygons,
    spatial_scatter,
)


class TestSpatialScatter:
    def test_returns_axes(self, sj_with_embeddings):
        ax = spatial_scatter(sj_with_embeddings, color_by="leiden", save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj_with_embeddings):
        spatial_scatter(sj_with_embeddings, color_by="leiden")
        figures_dir = sj_with_embeddings.config.root / "figures"
        assert (figures_dir / "spatial_scatter_leiden.png").exists()
        plt.close("all")

    def test_save_false_skips(self, sj_with_embeddings):
        spatial_scatter(sj_with_embeddings, color_by="leiden", save=False)
        figures_dir = sj_with_embeddings.config.root / "figures"
        assert not figures_dir.exists() or not (figures_dir / "spatial_scatter_leiden.png").exists()
        plt.close("all")

    def test_existing_ax(self, sj_with_embeddings):
        _, existing_ax = plt.subplots()
        ax = spatial_scatter(sj_with_embeddings, color_by="leiden", ax=existing_ax, save=False)
        assert ax is existing_ax
        plt.close("all")

    def test_bbox_filter(self, sj_with_embeddings):
        ax = spatial_scatter(sj_with_embeddings, color_by="leiden", xlim=(0, 200), ylim=(0, 100), save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_continuous_color(self, sj_with_embeddings):
        ax = spatial_scatter(sj_with_embeddings, color_by="gene_0", save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_image_overlay(self, sj_with_embeddings, tmp_path):
        img = np.random.default_rng(42).integers(0, 255, (100, 100), dtype=np.uint8)
        img_path = tmp_path / "test.tif"
        tifffile.imwrite(str(img_path), img)
        ax = spatial_scatter(sj_with_embeddings, color_by="leiden", image_path=img_path, save=False)
        # Check that imshow was called (at least one AxesImage in the axes)
        assert len(ax.get_images()) >= 1
        plt.close("all")


class TestSpatialPolygons:
    def test_returns_axes(self, sj_with_embeddings):
        # sj_with_embeddings inherits from sj_with_pt_clusters -> sj_with_knn_graph -> sj
        # sj does NOT have boundaries, so we need sj_with_boundaries
        # However sj_with_embeddings only has sj + knn_graph + clusters + umap
        # We'll test with sj that has boundaries
        pass  # Tested below with boundary fixture

    def test_saves_figure(self, sj_with_embeddings):
        pass  # Tested below with boundary fixture


class TestSpatialPolygonsWithBoundaries:
    """Tests requiring sj_with_boundaries (polygon data)."""

    def test_returns_axes(self, sj_with_clusters):
        ax = spatial_polygons(sj_with_clusters, color_by="leiden", save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj_with_clusters):
        spatial_polygons(sj_with_clusters, color_by="leiden")
        figures_dir = sj_with_clusters.config.root / "figures"
        assert (figures_dir / "spatial_polygons_leiden.png").exists()
        plt.close("all")

    def test_existing_ax(self, sj_with_clusters):
        _, existing_ax = plt.subplots()
        ax = spatial_polygons(sj_with_clusters, color_by="leiden", ax=existing_ax, save=False)
        assert ax is existing_ax
        plt.close("all")


class TestSpatialExpression:
    def test_returns_axes_scatter(self, sj_with_embeddings):
        ax = spatial_expression(sj_with_embeddings, gene="gene_0", mode="scatter", save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj_with_embeddings):
        spatial_expression(sj_with_embeddings, gene="gene_0")
        figures_dir = sj_with_embeddings.config.root / "figures"
        assert (figures_dir / "spatial_expression_gene_0.png").exists()
        plt.close("all")

    def test_invalid_gene_raises(self, sj_with_embeddings):
        with pytest.raises(ValueError, match="not found"):
            spatial_expression(sj_with_embeddings, gene="nonexistent_gene_xyz", save=False)
        plt.close("all")
```

- [ ] **Step 2: Implement spatial.py**

```python
"""Spatial coordinate plots (dot maps, polygon maps, image overlay).

All plotting functions accept an ``s_spatioloji`` object and return
``matplotlib.axes.Axes``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from s_spatioloji.visualization._common import (
    _categorical_palette,
    _filter_bbox,
    _get_feature_values,
    _save_figure,
    _setup_ax,
    _subsample,
)

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def _load_image(image_path: Path | str) -> np.ndarray:
    """Load a TIF image, handling multi-page files.

    Args:
        image_path: Path to TIF file.

    Returns:
        2D or 3D numpy array (first page if multi-page).
    """
    import tifffile

    img = tifffile.imread(str(image_path))
    # Multi-page: take first page
    if img.ndim > 3:
        img = img[0]
    return img


def _overlay_image(ax: Axes, image_path: Path | str | None, image_alpha: float, extent: list[float]) -> None:
    """Overlay a tissue image on axes if path is provided.

    Args:
        ax: Axes to plot on.
        image_path: Path to TIF, or None to skip.
        image_alpha: Image transparency.
        extent: [x_min, x_max, y_max, y_min] for imshow extent.
    """
    if image_path is None:
        return
    img = _load_image(image_path)
    ax.imshow(img, alpha=image_alpha, extent=extent, aspect="auto", zorder=0)


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
    dpi: int = 200,
    fmt: str = "png",
) -> Axes:
    """Spatial scatter plot of cells colored by feature.

    Args:
        sj: Dataset instance.
        color_by: Feature to color by (maps key, gene name, or cells column).
        cmap: Colormap for continuous features.
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.
        palette: Override categorical palette.
        xlim: Bounding box x limits.
        ylim: Bounding box y limits.
        max_cells: Subsample threshold.
        image_path: TIF file for background image.
        image_alpha: Background image transparency.
        figsize: Figure size.
        point_size: Marker size.
        alpha: Marker transparency.
        ax: Existing axes to plot on.
        save: If True, auto-save to figures/.
        save_path: Override save location.
        dpi: Output resolution.
        fmt: Output format.

    Returns:
        The matplotlib Axes with the spatial scatter plot.

    Example:
        >>> spatial_scatter(sj, color_by="leiden")
    """
    fig, ax = _setup_ax(ax, figsize)

    # Load and filter coordinates
    cells_df = sj.cells.df[["cell_id", "x", "y"]].compute()
    cells_df = _filter_bbox(cells_df, xlim, ylim)
    cells_df = _subsample(cells_df, max_cells)
    cell_ids = cells_df["cell_id"].values
    x = cells_df["x"].values
    y = cells_df["y"].values

    # Image overlay
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    _overlay_image(ax, image_path, image_alpha, [x_min, x_max, y_max, y_min])

    # Resolve feature
    values, is_cat = _get_feature_values(sj, color_by, cell_ids)

    if is_cat:
        categories = sorted(set(values), key=str)
        colors = palette or _categorical_palette(len(categories))
        for i, cat in enumerate(categories):
            mask = values == cat
            ax.scatter(x[mask], y[mask], c=colors[i % len(colors)], s=point_size,
                       alpha=alpha, label=str(cat), edgecolors="none", rasterized=True, zorder=1)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, markerscale=2)
    else:
        vals = values.astype(float)
        sc = ax.scatter(x, y, c=vals, cmap=cmap, s=point_size, alpha=alpha,
                        vmin=vmin, vmax=vmax, edgecolors="none", rasterized=True, zorder=1)
        plt.colorbar(sc, ax=ax, shrink=0.6, label=color_by)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    _save_figure(fig, sj, f"spatial_scatter_{color_by}", save=save, save_path=save_path, dpi=dpi, fmt=fmt)
    return ax


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
    dpi: int = 200,
    fmt: str = "png",
) -> Axes:
    """Spatial polygon plot of cell boundaries colored by feature.

    Args:
        sj: Dataset instance.
        color_by: Feature to color by.
        cmap: Colormap for continuous features.
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.
        palette: Override categorical palette.
        xlim: Bounding box x limits.
        ylim: Bounding box y limits.
        max_cells: Subsample threshold.
        image_path: TIF file for background image.
        image_alpha: Background image transparency.
        edgecolor: Polygon edge color.
        linewidth: Polygon edge width.
        figsize: Figure size.
        alpha: Polygon fill transparency.
        ax: Existing axes to plot on.
        save: If True, auto-save to figures/.
        save_path: Override save location.
        dpi: Output resolution.
        fmt: Output format.

    Returns:
        The matplotlib Axes with the polygon plot.

    Example:
        >>> spatial_polygons(sj, color_by="leiden")
    """
    import geopandas as gpd

    fig, ax = _setup_ax(ax, figsize)

    # Load boundaries
    gdf = sj.boundaries.load()

    # Filter by bbox
    if xlim is not None and ylim is not None:
        gdf = sj.boundaries.query_bbox(xlim[0], ylim[0], xlim[1], ylim[1])

    # Subsample
    if len(gdf) > max_cells:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(gdf), max_cells, replace=False)
        gdf = gdf.iloc[sorted(indices)]

    cell_ids = gdf["cell_id"].values

    # Image overlay
    bounds = gdf.total_bounds  # [x_min, y_min, x_max, y_max]
    _overlay_image(ax, image_path, image_alpha, [bounds[0], bounds[2], bounds[3], bounds[1]])

    # Resolve feature
    values, is_cat = _get_feature_values(sj, color_by, cell_ids)

    if is_cat:
        categories = sorted(set(values), key=str)
        colors = palette or _categorical_palette(len(categories))
        color_map = {str(cat): colors[i % len(colors)] for i, cat in enumerate(categories)}
        gdf = gdf.copy()
        gdf["_color"] = [color_map[str(v)] for v in values]
        gdf.plot(ax=ax, color=gdf["_color"], edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, zorder=1)
        # Manual legend
        from matplotlib.patches import Patch
        handles = [Patch(facecolor=color_map[str(c)], label=str(c)) for c in categories]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
    else:
        vals = values.astype(float)
        gdf = gdf.copy()
        gdf["_value"] = vals
        gdf.plot(ax=ax, column="_value", cmap=cmap, vmin=vmin, vmax=vmax,
                 edgecolor=edgecolor, linewidth=linewidth, alpha=alpha, legend=True, zorder=1)

    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    _save_figure(fig, sj, f"spatial_polygons_{color_by}", save=save, save_path=save_path, dpi=dpi, fmt=fmt)
    return ax


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
    dpi: int = 200,
    fmt: str = "png",
) -> Axes:
    """Spatial gene expression overlay.

    Convenience wrapper that delegates to :func:`spatial_scatter` or
    :func:`spatial_polygons` with ``color_by=gene``.

    Args:
        sj: Dataset instance.
        gene: Gene name (required). Must be in ``sj.expression.gene_names``.
        mode: ``"scatter"`` or ``"polygon"``.
        cmap: Colormap for expression values.
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.
        xlim: Bounding box x limits.
        ylim: Bounding box y limits.
        max_cells: Subsample threshold.
        image_path: TIF file for background image.
        image_alpha: Background image transparency.
        figsize: Figure size.
        point_size: Marker size (scatter mode only).
        ax: Existing axes to plot on.
        save: If True, auto-save to figures/.
        save_path: Override save location.
        dpi: Output resolution.
        fmt: Output format.

    Returns:
        The matplotlib Axes.

    Raises:
        ValueError: If gene is not found.

    Example:
        >>> spatial_expression(sj, gene="Slc17a7")
    """
    gene_names = list(sj.expression.gene_names)
    if gene not in gene_names:
        raise ValueError(f"Gene '{gene}' not found in sj.expression.gene_names.")

    # Override save name
    save_path_override = save_path
    if save_path is None and save:
        figures_dir = sj.config.root / "figures"
        figures_dir.mkdir(exist_ok=True)
        save_path_override = figures_dir / f"spatial_expression_{gene}.{fmt}"

    if mode == "scatter":
        return spatial_scatter(
            sj, color_by=gene, cmap=cmap, vmin=vmin, vmax=vmax,
            xlim=xlim, ylim=ylim, max_cells=max_cells,
            image_path=image_path, image_alpha=image_alpha,
            figsize=figsize, point_size=point_size, ax=ax,
            save=save, save_path=save_path_override, dpi=dpi, fmt=fmt,
        )
    elif mode == "polygon":
        return spatial_polygons(
            sj, color_by=gene, cmap=cmap, vmin=vmin, vmax=vmax,
            xlim=xlim, ylim=ylim, max_cells=max_cells,
            image_path=image_path, image_alpha=image_alpha,
            figsize=figsize, ax=ax,
            save=save, save_path=save_path_override, dpi=dpi, fmt=fmt,
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Must be 'scatter' or 'polygon'.")
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/unit/test_visualization_spatial.py -v`
Expected: ALL PASS

- [ ] **Step 4: Lint and commit**

```bash
ruff check src/s_spatioloji/visualization/spatial.py tests/unit/test_visualization_spatial.py --fix
ruff format src/s_spatioloji/visualization/spatial.py tests/unit/test_visualization_spatial.py
git add src/s_spatioloji/visualization/spatial.py tests/unit/test_visualization_spatial.py
git commit -m "feat(visualization): add spatial scatter, polygon, and expression plots"
```

---

## Chunk 3: Expression Plots (Task 5)

### Task 5: expression.py (heatmap, violin, dotplot)

**Files:**
- Create: `src/s_spatioloji/visualization/expression.py`
- Create: `tests/unit/test_visualization_expression.py`

- [ ] **Step 1: Write expression tests**

Create `tests/unit/test_visualization_expression.py`:

```python
"""Unit tests for s_spatioloji.visualization.expression."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from s_spatioloji.visualization.expression import dotplot, heatmap, violin


class TestHeatmap:
    def test_returns_axes(self, sj_with_embeddings):
        ax = heatmap(sj_with_embeddings, genes=["gene_0", "gene_1", "gene_2"], save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj_with_embeddings):
        heatmap(sj_with_embeddings, genes=["gene_0", "gene_1"])
        figures_dir = sj_with_embeddings.config.root / "figures"
        assert (figures_dir / "heatmap_leiden.png").exists()
        plt.close("all")

    def test_standardize_uses_diverging_cmap(self, sj_with_embeddings):
        ax = heatmap(sj_with_embeddings, genes=["gene_0", "gene_1"], standardize=True, save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_existing_ax(self, sj_with_embeddings):
        _, existing_ax = plt.subplots()
        ax = heatmap(sj_with_embeddings, genes=["gene_0"], ax=existing_ax, save=False)
        assert ax is existing_ax
        plt.close("all")


class TestViolin:
    def test_returns_figure(self, sj_with_embeddings):
        fig = violin(sj_with_embeddings, genes=["gene_0", "gene_1"], save=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_saves_figure(self, sj_with_embeddings):
        violin(sj_with_embeddings, genes=["gene_0"])
        figures_dir = sj_with_embeddings.config.root / "figures"
        assert (figures_dir / "violin_leiden.png").exists()
        plt.close("all")

    def test_multi_gene_grid(self, sj_with_embeddings):
        fig = violin(sj_with_embeddings, genes=["gene_0", "gene_1", "gene_2", "gene_3", "gene_4"], ncols=3, save=False)
        axes = fig.get_axes()
        assert len(axes) >= 5
        plt.close("all")


class TestDotplot:
    def test_returns_axes(self, sj_with_embeddings):
        ax = dotplot(sj_with_embeddings, genes=["gene_0", "gene_1", "gene_2"], save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj_with_embeddings):
        dotplot(sj_with_embeddings, genes=["gene_0", "gene_1"])
        figures_dir = sj_with_embeddings.config.root / "figures"
        assert (figures_dir / "dotplot_leiden.png").exists()
        plt.close("all")

    def test_existing_ax(self, sj_with_embeddings):
        _, existing_ax = plt.subplots()
        ax = dotplot(sj_with_embeddings, genes=["gene_0"], ax=existing_ax, save=False)
        assert ax is existing_ax
        plt.close("all")
```

- [ ] **Step 2: Implement expression.py**

```python
"""Gene expression summary plots (heatmap, violin, dotplot).

All plotting functions accept an ``s_spatioloji`` object and return
``matplotlib.axes.Axes`` or ``matplotlib.figure.Figure``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from s_spatioloji.visualization._common import (
    _categorical_palette,
    _save_figure,
    _setup_ax,
)

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


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
    dpi: int = 200,
    fmt: str = "png",
) -> Axes:
    """Heatmap of mean gene expression per cluster.

    Args:
        sj: Dataset instance.
        genes: Gene names to show.
        cluster_key: Cluster labels key in maps/.
        cmap: Colormap. Defaults to ``"RdBu_r"`` when ``standardize=True``.
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.
        standardize: Z-score each gene across clusters.
        figsize: Figure size (auto-sized if None).
        ax: Existing axes to plot on.
        save: If True, auto-save to figures/.
        save_path: Override save location.
        dpi: Output resolution.
        fmt: Output format.

    Returns:
        The matplotlib Axes.

    Example:
        >>> heatmap(sj, genes=["Slc17a7", "Gad1", "Aqp4"])
    """
    # Auto-select diverging cmap for standardized data
    if standardize and cmap == "magma":
        cmap = "RdBu_r"

    # Load expression and clusters
    cluster_df = sj.maps[cluster_key].compute()
    cell_to_cluster = dict(zip(cluster_df["cell_id"], cluster_df[cluster_key], strict=True))
    clusters = sorted(set(cell_to_cluster.values()), key=str)

    gene_names = list(sj.expression.gene_names)
    gene_indices = [gene_names.index(g) for g in genes]
    expr_cell_ids = list(sj.expression.cell_ids)

    # Build mean expression matrix (genes x clusters)
    matrix = np.zeros((len(genes), len(clusters)))
    for gi, gene_idx in enumerate(gene_indices):
        expr_vals = sj.expression[:, gene_idx].compute().ravel()
        for ci, cluster in enumerate(clusters):
            mask = np.array([cell_to_cluster.get(cid) == cluster for cid in expr_cell_ids])
            if mask.sum() > 0:
                matrix[gi, ci] = expr_vals[mask].mean()

    if standardize:
        row_mean = matrix.mean(axis=1, keepdims=True)
        row_std = matrix.std(axis=1, keepdims=True)
        row_std[row_std == 0] = 1.0
        matrix = (matrix - row_mean) / row_std

    if figsize is None:
        figsize = (max(3, len(clusters) * 0.6 + 2), max(3, len(genes) * 0.4 + 1))

    fig, ax = _setup_ax(ax, figsize)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.6)

    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels([str(c) for c in clusters], rotation=45, ha="right")
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes)

    _save_figure(fig, sj, f"heatmap_{cluster_key}", save=save, save_path=save_path, dpi=dpi, fmt=fmt)
    return ax


def violin(
    sj: s_spatioloji,
    genes: list[str],
    cluster_key: str = "leiden",
    palette: list[str] | None = None,
    ncols: int = 4,
    figsize: tuple[float, float] | None = None,
    save: bool = True,
    save_path: Path | str | None = None,
    dpi: int = 200,
    fmt: str = "png",
) -> Figure:
    """Violin plot of gene expression per cluster.

    Args:
        sj: Dataset instance.
        genes: Gene names to show.
        cluster_key: Cluster labels key in maps/.
        palette: Override categorical palette.
        ncols: Number of columns in grid layout.
        figsize: Figure size (auto-sized if None).
        save: If True, auto-save to figures/.
        save_path: Override save location.
        dpi: Output resolution.
        fmt: Output format.

    Returns:
        The matplotlib Figure.

    Example:
        >>> violin(sj, genes=["Slc17a7", "Gad1"])
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError("Install seaborn for violin plots: pip install seaborn")

    import pandas as pd

    cluster_df = sj.maps[cluster_key].compute()
    cell_to_cluster = dict(zip(cluster_df["cell_id"], cluster_df[cluster_key], strict=True))
    clusters = sorted(set(cell_to_cluster.values()), key=str)

    gene_names = list(sj.expression.gene_names)
    expr_cell_ids = list(sj.expression.cell_ids)

    n_genes = len(genes)
    nrows = math.ceil(n_genes / ncols)
    if figsize is None:
        figsize = (ncols * 3.5, nrows * 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    colors = palette or _categorical_palette(len(clusters))

    for gi, gene in enumerate(genes):
        row, col = divmod(gi, ncols)
        ax = axes[row][col]

        gene_idx = gene_names.index(gene)
        expr_vals = sj.expression[:, gene_idx].compute().ravel()

        records = []
        for cid, val in zip(expr_cell_ids, expr_vals, strict=True):
            if cid in cell_to_cluster:
                records.append({"cluster": str(cell_to_cluster[cid]), "expression": float(val)})

        plot_df = pd.DataFrame(records)
        sns.violinplot(
            data=plot_df, x="cluster", y="expression", ax=ax,
            order=[str(c) for c in clusters], palette=colors[:len(clusters)],
            inner="box", linewidth=0.5, cut=0,
        )
        ax.set_title(gene)
        ax.set_xlabel("")
        if col > 0:
            ax.set_ylabel("")

    # Hide unused axes
    for gi in range(n_genes, nrows * ncols):
        row, col = divmod(gi, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    _save_figure(fig, sj, f"violin_{cluster_key}", save=save, save_path=save_path, dpi=dpi, fmt=fmt)
    return fig


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
    dpi: int = 200,
    fmt: str = "png",
) -> Axes:
    """Dot plot showing fraction expressing and mean expression.

    Args:
        sj: Dataset instance.
        genes: Gene names to show.
        cluster_key: Cluster labels key in maps/.
        cmap: Colormap for mean expression.
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.
        max_dot_size: Maximum dot size in points^2.
        figsize: Figure size (auto-sized if None).
        ax: Existing axes to plot on.
        save: If True, auto-save to figures/.
        save_path: Override save location.
        dpi: Output resolution.
        fmt: Output format.

    Returns:
        The matplotlib Axes.

    Example:
        >>> dotplot(sj, genes=["Slc17a7", "Gad1", "Aqp4"])
    """
    cluster_df = sj.maps[cluster_key].compute()
    cell_to_cluster = dict(zip(cluster_df["cell_id"], cluster_df[cluster_key], strict=True))
    clusters = sorted(set(cell_to_cluster.values()), key=str)

    gene_names = list(sj.expression.gene_names)
    expr_cell_ids = list(sj.expression.cell_ids)

    # Compute fraction expressing and mean expression
    frac_matrix = np.zeros((len(clusters), len(genes)))
    mean_matrix = np.zeros((len(clusters), len(genes)))

    for gi, gene in enumerate(genes):
        gene_idx = gene_names.index(gene)
        expr_vals = sj.expression[:, gene_idx].compute().ravel()

        for ci, cluster in enumerate(clusters):
            mask = np.array([cell_to_cluster.get(cid) == cluster for cid in expr_cell_ids])
            if mask.sum() > 0:
                cluster_vals = expr_vals[mask]
                frac_matrix[ci, gi] = (cluster_vals > 0).mean()
                expressing = cluster_vals[cluster_vals > 0]
                mean_matrix[ci, gi] = expressing.mean() if len(expressing) > 0 else 0.0

    if figsize is None:
        figsize = (max(3, len(genes) * 0.8 + 2), max(3, len(clusters) * 0.6 + 1))

    fig, ax = _setup_ax(ax, figsize)

    # Create grid coordinates
    x_coords, y_coords = np.meshgrid(range(len(genes)), range(len(clusters)))
    sizes = frac_matrix.flatten() * max_dot_size
    colors = mean_matrix.flatten()

    sc = ax.scatter(
        x_coords.flatten(), y_coords.flatten(),
        s=sizes, c=colors, cmap=cmap, vmin=vmin, vmax=vmax,
        edgecolors="gray", linewidth=0.5,
    )
    plt.colorbar(sc, ax=ax, shrink=0.6, label="Mean expression")

    # Size legend
    for frac in [0.25, 0.5, 0.75, 1.0]:
        ax.scatter([], [], s=frac * max_dot_size, c="gray", edgecolors="gray",
                   linewidth=0.5, label=f"{int(frac * 100)}%")
    ax.legend(title="Fraction\nexpressing", bbox_to_anchor=(1.25, 1), loc="upper left",
              frameon=False, labelspacing=1.2)

    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=45, ha="right")
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([str(c) for c in clusters])

    _save_figure(fig, sj, f"dotplot_{cluster_key}", save=save, save_path=save_path, dpi=dpi, fmt=fmt)
    return ax
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/unit/test_visualization_expression.py -v`
Expected: ALL PASS

- [ ] **Step 4: Lint and commit**

```bash
ruff check src/s_spatioloji/visualization/expression.py tests/unit/test_visualization_expression.py --fix
ruff format src/s_spatioloji/visualization/expression.py tests/unit/test_visualization_expression.py
git add src/s_spatioloji/visualization/expression.py tests/unit/test_visualization_expression.py
git commit -m "feat(visualization): add heatmap, violin, and dotplot"
```

---

## Chunk 4: Analysis Plots (Task 6)

### Task 6: analysis.py (ripley_plot, colocalization_heatmap, neighborhood_bar, envelope_plot)

**Files:**
- Create: `src/s_spatioloji/visualization/analysis.py`
- Create: `tests/unit/test_visualization_analysis.py`

- [ ] **Step 1: Write analysis tests**

Create `tests/unit/test_visualization_analysis.py`:

```python
"""Unit tests for s_spatioloji.visualization.analysis."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from s_spatioloji.spatial.point.patterns import colocalization
from s_spatioloji.spatial.point.neighborhoods import neighborhood_composition
from s_spatioloji.spatial.point.ripley import ripley_k
from s_spatioloji.spatial.point.statistics import dclf_envelope
from s_spatioloji.visualization.analysis import (
    colocalization_heatmap,
    envelope_plot,
    neighborhood_bar,
    ripley_plot,
)


class TestRipleyPlot:
    def test_returns_axes(self, sj_with_pt_clusters):
        ripley_k(sj_with_pt_clusters, n_radii=10)
        ax = ripley_plot(sj_with_pt_clusters, function="K", save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj_with_pt_clusters):
        ripley_k(sj_with_pt_clusters, n_radii=10)
        ripley_plot(sj_with_pt_clusters, function="K")
        figures_dir = sj_with_pt_clusters.config.root / "figures"
        assert (figures_dir / "ripley_k.png").exists()
        plt.close("all")

    def test_has_lines(self, sj_with_pt_clusters):
        ripley_k(sj_with_pt_clusters, n_radii=10)
        ax = ripley_plot(sj_with_pt_clusters, function="K", save=False)
        assert len(ax.get_lines()) >= 2  # observed + theoretical
        plt.close("all")


class TestColocalizationHeatmap:
    def test_returns_axes(self, sj_with_pt_clusters):
        colocalization(sj_with_pt_clusters)
        ax = colocalization_heatmap(sj_with_pt_clusters, save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj_with_pt_clusters):
        colocalization(sj_with_pt_clusters)
        colocalization_heatmap(sj_with_pt_clusters)
        figures_dir = sj_with_pt_clusters.config.root / "figures"
        assert (figures_dir / "colocalization_log2_ratio.png").exists()
        plt.close("all")


class TestNeighborhoodBar:
    def test_returns_axes(self, sj_with_pt_clusters):
        neighborhood_composition(sj_with_pt_clusters)
        ax = neighborhood_bar(sj_with_pt_clusters, save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj_with_pt_clusters):
        neighborhood_composition(sj_with_pt_clusters)
        neighborhood_bar(sj_with_pt_clusters)
        figures_dir = sj_with_pt_clusters.config.root / "figures"
        assert (figures_dir / "neighborhood_bar.png").exists()
        plt.close("all")


class TestEnvelopePlot:
    def test_returns_axes(self, sj):
        dclf_envelope(sj, n_simulations=9, n_radii=5)
        ax = envelope_plot(sj, save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj):
        dclf_envelope(sj, n_simulations=9, n_radii=5)
        envelope_plot(sj)
        figures_dir = sj.config.root / "figures"
        assert (figures_dir / "envelope_pt_dclf_envelope.png").exists()
        plt.close("all")

    def test_has_fill_between(self, sj):
        dclf_envelope(sj, n_simulations=9, n_radii=5)
        ax = envelope_plot(sj, save=False)
        # Check for fill_between (PolyCollection)
        assert len(ax.collections) >= 1
        plt.close("all")
```

- [ ] **Step 2: Implement analysis.py**

```python
"""Spatial analysis result plots (Ripley's, colocalization, neighborhoods, envelopes).

All plotting functions accept an ``s_spatioloji`` object and return
``matplotlib.axes.Axes``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from s_spatioloji.visualization._common import (
    _categorical_palette,
    _save_figure,
    _setup_ax,
)

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def ripley_plot(
    sj: s_spatioloji,
    function: str = "K",
    maps_key: str | None = None,
    figsize: tuple[float, float] = (8, 6),
    ax: Axes | None = None,
    save: bool = True,
    save_path: Path | str | None = None,
    dpi: int = 200,
    fmt: str = "png",
) -> Axes:
    """Plot Ripley's K, L, G, or F function.

    Args:
        sj: Dataset instance.
        function: One of ``"K"``, ``"L"``, ``"G"``, ``"F"``.
        maps_key: Maps key for the Ripley result. Auto-resolves if None.
        figsize: Figure size.
        ax: Existing axes to plot on.
        save: If True, auto-save to figures/.
        save_path: Override save location.
        dpi: Output resolution.
        fmt: Output format.

    Returns:
        The matplotlib Axes.

    Example:
        >>> ripley_plot(sj, function="K")
    """
    if maps_key is None:
        maps_key = f"pt_ripley_{function.lower()}"

    fig, ax = _setup_ax(ax, figsize)
    df = sj.maps[maps_key].compute()

    clusters = sorted(df["cluster"].unique(), key=str)
    colors = _categorical_palette(len(clusters))

    func_upper = function.upper()
    for i, cluster in enumerate(clusters):
        cdf = df[df["cluster"] == cluster]
        r = cdf["r"].values
        label = str(cluster) if cluster != "all" else "Observed"

        if func_upper == "K":
            ax.plot(r, cdf["K"].values, color=colors[i], label=label)
            if i == 0:
                ax.plot(r, cdf["K_theo"].values, "--", color="gray", label="CSR (theoretical)")
        elif func_upper == "L":
            ax.plot(r, cdf["L"].values, color=colors[i], label=label)
            if i == 0:
                ax.axhline(0, linestyle="--", color="gray", label="CSR (L=0)")
        elif func_upper == "G":
            ax.plot(r, cdf["G"].values, color=colors[i], label=label)
            if i == 0:
                ax.plot(r, cdf["G_theo"].values, "--", color="gray", label="CSR (theoretical)")
        elif func_upper == "F":
            ax.plot(r, cdf["F"].values, color=colors[i], label=label)
            if i == 0:
                ax.plot(r, cdf["F_theo"].values, "--", color="gray", label="CSR (theoretical)")

    ax.set_xlabel("r")
    ax.set_ylabel(func_upper)
    ax.legend(frameon=False)

    _save_figure(fig, sj, f"ripley_{function.lower()}", save=save, save_path=save_path, dpi=dpi, fmt=fmt)
    return ax


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
    dpi: int = 200,
    fmt: str = "png",
) -> Axes:
    """Heatmap of colocalization ratios between cluster pairs.

    Args:
        sj: Dataset instance.
        maps_key: Maps key for colocalization results.
        metric: Column to plot (``"log2_ratio"``, ``"ratio"``, etc.).
        cmap: Colormap (diverging recommended for log2_ratio).
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.
        figsize: Figure size (auto-sized if None).
        ax: Existing axes to plot on.
        save: If True, auto-save to figures/.
        save_path: Override save location.
        dpi: Output resolution.
        fmt: Output format.

    Returns:
        The matplotlib Axes.

    Example:
        >>> colocalization_heatmap(sj, metric="log2_ratio")
    """
    df = sj.maps[maps_key].compute()
    clusters = sorted(set(df["cluster_a"]).union(set(df["cluster_b"])), key=str)
    n = len(clusters)
    cluster_to_idx = {str(c): i for i, c in enumerate(clusters)}

    matrix = np.full((n, n), np.nan)
    for _, row in df.iterrows():
        i = cluster_to_idx[str(row["cluster_a"])]
        j = cluster_to_idx[str(row["cluster_b"])]
        val = row[metric]
        matrix[i, j] = val
        matrix[j, i] = val

    if figsize is None:
        figsize = (max(4, n * 0.8 + 2), max(4, n * 0.8 + 1))

    fig, ax = _setup_ax(ax, figsize)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    plt.colorbar(im, ax=ax, shrink=0.6, label=metric)

    ax.set_xticks(range(n))
    ax.set_xticklabels([str(c) for c in clusters], rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(c) for c in clusters])

    _save_figure(fig, sj, f"colocalization_{metric}", save=save, save_path=save_path, dpi=dpi, fmt=fmt)
    return ax


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
    dpi: int = 200,
    fmt: str = "png",
) -> Axes:
    """Stacked bar chart of neighborhood composition per cluster.

    Args:
        sj: Dataset instance.
        maps_key: Maps key for neighborhood composition results.
        cluster_key: Cluster labels key for grouping.
        normalize: Show proportions (True) or raw counts (False).
        palette: Override categorical palette.
        figsize: Figure size (auto-sized if None).
        ax: Existing axes to plot on.
        save: If True, auto-save to figures/.
        save_path: Override save location.
        dpi: Output resolution.
        fmt: Output format.

    Returns:
        The matplotlib Axes.

    Example:
        >>> neighborhood_bar(sj)
    """
    comp_df = sj.maps[maps_key].compute()
    cluster_df = sj.maps[cluster_key].compute()

    # Merge cluster labels
    merged = comp_df.merge(cluster_df[["cell_id", cluster_key]], on="cell_id")
    neighbor_cols = [c for c in comp_df.columns if c != "cell_id"]
    clusters = sorted(merged[cluster_key].unique(), key=str)

    # Mean composition per source cluster
    mean_comp = np.zeros((len(clusters), len(neighbor_cols)))
    for ci, cluster in enumerate(clusters):
        mask = merged[cluster_key] == cluster
        mean_comp[ci] = merged.loc[mask, neighbor_cols].values.mean(axis=0)

    if normalize:
        row_sums = mean_comp.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        mean_comp = mean_comp / row_sums

    if figsize is None:
        figsize = (8, max(3, len(clusters) * 0.5 + 1))

    fig, ax = _setup_ax(ax, figsize)
    colors = palette or _categorical_palette(len(neighbor_cols))

    left = np.zeros(len(clusters))
    for j, col in enumerate(neighbor_cols):
        ax.barh(range(len(clusters)), mean_comp[:, j], left=left,
                color=colors[j % len(colors)], label=str(col), edgecolor="white", linewidth=0.3)
        left += mean_comp[:, j]

    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels([str(c) for c in clusters])
    ax.set_xlabel("Proportion" if normalize else "Count")
    ax.legend(title="Neighbor cluster", bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)

    _save_figure(fig, sj, "neighborhood_bar", save=save, save_path=save_path, dpi=dpi, fmt=fmt)
    return ax


def envelope_plot(
    sj: s_spatioloji,
    maps_key: str = "pt_dclf_envelope",
    figsize: tuple[float, float] = (8, 6),
    ax: Axes | None = None,
    save: bool = True,
    save_path: Path | str | None = None,
    dpi: int = 200,
    fmt: str = "png",
) -> Axes:
    """DCLF envelope plot with simulation band.

    Args:
        sj: Dataset instance.
        maps_key: Maps key for envelope results.
        figsize: Figure size.
        ax: Existing axes to plot on.
        save: If True, auto-save to figures/.
        save_path: Override save location.
        dpi: Output resolution.
        fmt: Output format.

    Returns:
        The matplotlib Axes.

    Example:
        >>> envelope_plot(sj)
    """
    fig, ax = _setup_ax(ax, figsize)
    df = sj.maps[maps_key].compute()

    clusters = sorted(df["cluster"].unique(), key=str)
    colors = _categorical_palette(len(clusters))

    for i, cluster in enumerate(clusters):
        cdf = df[df["cluster"] == cluster]
        r = cdf["r"].values
        p_value = cdf["p_value"].iloc[0]

        label = f"{cluster} (p={p_value:.3f})" if cluster != "all" else f"Observed (p={p_value:.3f})"
        ax.plot(r, cdf["observed"].values, color=colors[i], label=label)
        ax.fill_between(r, cdf["lo"].values, cdf["hi"].values, color=colors[i], alpha=0.15)
        if i == 0:
            ax.plot(r, cdf["theo"].values, "--", color="gray", label="CSR (theoretical)")

    ax.set_xlabel("r")
    ax.set_ylabel("Function value")
    ax.legend(frameon=False)

    _save_figure(fig, sj, f"envelope_{maps_key}", save=save, save_path=save_path, dpi=dpi, fmt=fmt)
    return ax
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/unit/test_visualization_analysis.py -v`
Expected: ALL PASS

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/unit/ -v`
Expected: ALL PASS

- [ ] **Step 5: Lint and commit**

```bash
ruff check src/s_spatioloji/visualization/analysis.py tests/unit/test_visualization_analysis.py --fix
ruff format src/s_spatioloji/visualization/analysis.py tests/unit/test_visualization_analysis.py
git add src/s_spatioloji/visualization/analysis.py tests/unit/test_visualization_analysis.py
git commit -m "feat(visualization): add Ripley plot, colocalization heatmap, neighborhood bar, envelope plot"
```
