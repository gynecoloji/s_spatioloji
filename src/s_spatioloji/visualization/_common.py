"""Shared helpers for visualization functions.

Private module -- not exported from ``s_spatioloji.visualization``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from matplotlib.colors import Colormap

    from s_spatioloji.data.core import s_spatioloji

# Publication-friendly categorical palette (colorblind-accessible)
_PALETTE = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
    "#AA3377", "#BBBBBB", "#EE8866", "#44BB99", "#FFAABB",
    "#332288", "#882255", "#117733", "#999933", "#CC6677",
    "#661100", "#6699CC", "#AA4499", "#DDCC77", "#88CCEE",
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
                np.issubdtype(values.dtype, np.integer)
                and len(set(values[~pd.isna(values)])) < 20
            )
            return values, is_cat

    # 2. Try gene name
    gene_names = list(sj.expression.gene_names)
    if color_by in gene_names:
        gene_idx = gene_names.index(color_by)
        expr_values = sj.expression.select_genes([gene_idx]).compute().ravel()
        expr_cell_ids = sj.expression.cell_ids
        if expr_cell_ids is not None:
            id_to_val = dict(zip(list(expr_cell_ids), expr_values, strict=True))
        else:
            # Fall back to cells store ordering (row-aligned)
            cells_df = sj.cells.df.compute()
            id_to_val = dict(zip(cells_df["cell_id"], expr_values, strict=True))
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
