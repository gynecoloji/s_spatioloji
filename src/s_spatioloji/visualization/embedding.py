"""Embedding scatter plots (UMAP, PCA, tSNE).

All plotting functions accept an ``s_spatioloji`` object and return
``matplotlib.axes.Axes``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from s_spatioloji.visualization._common import (
    _categorical_palette,
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

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    _save_figure(fig, sj, f"scatter_{basis}_{color_by}", save=save, save_path=save_path, dpi=dpi, fmt=fmt)
    return ax
