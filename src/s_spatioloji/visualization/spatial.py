"""Spatial coordinate plots: scatter, polygon, and gene expression overlays.

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
    _filter_bbox,
    _get_feature_values,
    _save_figure,
    _setup_ax,
    _subsample,
)

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


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
    image_alpha: float = 0.3,
    figsize: tuple[float, float] = (8, 8),
    point_size: float = 5,
    alpha: float = 1.0,
    ax: Axes | None = None,
    save: bool = True,
    save_path: Path | str | None = None,
    dpi: int = 200,
    fmt: str = "png",
) -> Axes:
    """Scatter plot of cells at their spatial (x, y) coordinates.

    Reads cell centroids from ``sj.cells.df``, applies optional bounding-box
    filtering and subsampling, then renders a scatter plot colored by a
    categorical or continuous feature.

    Args:
        sj: Dataset instance.
        color_by: Feature to color by (maps key, gene name, or cells column).
        cmap: Colormap for continuous features.
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.
        palette: Override categorical palette (list of hex colors).
        xlim: (x_min, x_max) bounding box filter on x-coordinates.
        ylim: (y_min, y_max) bounding box filter on y-coordinates.
        max_cells: Subsample threshold.
        image_path: Path to a TIF image to overlay behind cells.
        image_alpha: Transparency of the background image.
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
        >>> spatial_scatter(sj, color_by="leiden")
    """
    fig, ax = _setup_ax(ax, figsize)

    # Load cell coordinates
    cells_df = sj.cells.df.compute()

    # Filter by bounding box
    cells_df = _filter_bbox(cells_df, xlim=xlim, ylim=ylim)

    # Subsample
    cells_df = _subsample(cells_df, max_cells)

    cell_ids = cells_df["cell_id"].values
    x = cells_df["x"].values
    y = cells_df["y"].values

    # Overlay background image if provided
    if image_path is not None:
        try:
            import tifffile
        except ImportError:
            raise ImportError("Install tifffile to overlay images: pip install tifffile")
        img = tifffile.imread(str(image_path))
        ax.imshow(img, alpha=image_alpha, extent=[x.min(), x.max(), y.max(), y.min()], aspect="auto")

    # Resolve feature
    values, is_cat = _get_feature_values(sj, color_by, cell_ids)

    if is_cat:
        categories = sorted(set(values), key=str)
        colors = palette or _categorical_palette(len(categories))
        for i, cat in enumerate(categories):
            mask = values == cat
            ax.scatter(
                x[mask], y[mask], c=colors[i % len(colors)], s=point_size,
                alpha=alpha, label=str(cat), edgecolors="none", rasterized=True,
            )
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False, markerscale=2)
    else:
        vals = values.astype(float)
        sc = ax.scatter(
            x, y, c=vals, cmap=cmap, s=point_size, alpha=alpha,
            vmin=vmin, vmax=vmax, edgecolors="none", rasterized=True,
        )
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
    image_alpha: float = 0.3,
    figsize: tuple[float, float] = (8, 8),
    alpha: float = 0.7,
    ax: Axes | None = None,
    save: bool = True,
    save_path: Path | str | None = None,
    dpi: int = 200,
    fmt: str = "png",
) -> Axes:
    """Polygon map of cell boundaries colored by a feature.

    Loads polygon boundaries from ``sj.boundaries``, optionally filters by
    bounding box, subsamples, and plots using geopandas.

    Args:
        sj: Dataset instance.
        color_by: Feature to color by (maps key, gene name, or cells column).
        cmap: Colormap for continuous features.
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.
        palette: Override categorical palette (list of hex colors).
        xlim: (x_min, x_max) bounding box filter.
        ylim: (y_min, y_max) bounding box filter.
        max_cells: Subsample threshold.
        image_path: Path to a TIF image to overlay behind polygons.
        image_alpha: Transparency of the background image.
        figsize: Figure size.
        alpha: Polygon fill transparency.
        ax: Existing axes to plot on.
        save: If True, auto-save to figures/.
        save_path: Override save location.
        dpi: Output resolution.
        fmt: Output format.

    Returns:
        The matplotlib Axes with the polygon plot.

    Raises:
        FileNotFoundError: If boundaries are not available.

    Example:
        >>> spatial_polygons(sj, color_by="leiden")
    """
    fig, ax = _setup_ax(ax, figsize)

    # Load boundaries
    if xlim is not None and ylim is not None:
        gdf = sj.boundaries.query_bbox(xlim[0], ylim[0], xlim[1], ylim[1])
    else:
        gdf = sj.boundaries.load()

    # Subsample
    gdf = _subsample(gdf, max_cells)

    cell_ids = gdf["cell_id"].values

    # Overlay background image if provided
    if image_path is not None:
        try:
            import tifffile
        except ImportError:
            raise ImportError("Install tifffile to overlay images: pip install tifffile")
        img = tifffile.imread(str(image_path))
        bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
        ax.imshow(img, alpha=image_alpha, extent=[bounds[0], bounds[2], bounds[3], bounds[1]], aspect="auto")

    # Resolve feature
    values, is_cat = _get_feature_values(sj, color_by, cell_ids)

    if is_cat:
        from matplotlib.patches import Patch

        categories = sorted(set(values), key=str)
        colors = palette or _categorical_palette(len(categories))
        cat_to_color = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}
        gdf = gdf.copy()
        gdf["_color"] = [cat_to_color[v] for v in values]
        for cat in categories:
            subset = gdf[gdf["_color"] == cat_to_color[cat]]
            subset.plot(ax=ax, color=cat_to_color[cat], alpha=alpha, edgecolor="black", linewidth=0.3)
        handles = [Patch(facecolor=cat_to_color[cat], label=str(cat)) for cat in categories]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
    else:
        vals = values.astype(float)
        gdf = gdf.copy()
        gdf["_value"] = vals
        gdf.plot(
            ax=ax, column="_value", cmap=cmap, alpha=alpha,
            edgecolor="black", linewidth=0.3, vmin=vmin, vmax=vmax,
            legend=True, legend_kwds={"shrink": 0.6, "label": color_by},
        )

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
    image_alpha: float = 0.3,
    figsize: tuple[float, float] = (8, 8),
    point_size: float = 5,
    alpha: float = 1.0,
    ax: Axes | None = None,
    save: bool = True,
    save_path: Path | str | None = None,
    dpi: int = 200,
    fmt: str = "png",
) -> Axes:
    """Plot spatial expression of a single gene.

    Convenience wrapper that validates the gene name and delegates to
    :func:`spatial_scatter` (``mode="scatter"``) or
    :func:`spatial_polygons` (``mode="polygon"``).

    Args:
        sj: Dataset instance.
        gene: Gene name to plot (must exist in ``sj.expression.gene_names``).
        mode: ``"scatter"`` for centroid dots, ``"polygon"`` for filled polygons.
        cmap: Colormap for expression values.
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.
        xlim: (x_min, x_max) bounding box filter.
        ylim: (y_min, y_max) bounding box filter.
        max_cells: Subsample threshold.
        image_path: Path to a TIF image to overlay behind cells.
        image_alpha: Transparency of the background image.
        figsize: Figure size.
        point_size: Marker size (scatter mode only).
        alpha: Transparency.
        ax: Existing axes to plot on.
        save: If True, auto-save to figures/.
        save_path: Override save location.
        dpi: Output resolution.
        fmt: Output format.

    Returns:
        The matplotlib Axes with the expression plot.

    Raises:
        ValueError: If ``gene`` is not found in the expression matrix.

    Example:
        >>> spatial_expression(sj, gene="gene_0")
    """
    gene_names = list(sj.expression.gene_names)
    if gene not in gene_names:
        raise ValueError(
            f"Gene {gene!r} not found in expression matrix. "
            f"Available genes: {gene_names[:5]}... ({len(gene_names)} total)"
        )

    if mode == "scatter":
        return spatial_scatter(
            sj, color_by=gene, cmap=cmap, vmin=vmin, vmax=vmax,
            xlim=xlim, ylim=ylim, max_cells=max_cells,
            image_path=image_path, image_alpha=image_alpha,
            figsize=figsize, point_size=point_size, alpha=alpha,
            ax=ax, save=save, save_path=save_path, dpi=dpi, fmt=fmt,
        )
    elif mode == "polygon":
        return spatial_polygons(
            sj, color_by=gene, cmap=cmap, vmin=vmin, vmax=vmax,
            xlim=xlim, ylim=ylim, max_cells=max_cells,
            image_path=image_path, image_alpha=image_alpha,
            figsize=figsize, alpha=alpha,
            ax=ax, save=save, save_path=save_path, dpi=dpi, fmt=fmt,
        )
    else:
        raise ValueError(f"mode must be 'scatter' or 'polygon', got {mode!r}")
