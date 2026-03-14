"""Expression summary plots: heatmap, violin, dotplot.

Visualise per-cluster gene expression statistics using grouped
summaries of the underlying expression matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from s_spatioloji.visualization._common import (
    _categorical_palette,
    _save_figure,
    _setup_ax,
)

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _build_cluster_gene_df(
    sj: s_spatioloji,
    genes: list[str],
    cluster_key: str,
) -> pd.DataFrame:
    """Build a long-form DataFrame with columns: cell_id, cluster, gene, expression.

    Args:
        sj: Dataset instance.
        genes: Gene names to include.
        cluster_key: Maps key for cluster labels.

    Returns:
        Long-form DataFrame.

    Raises:
        ValueError: If a gene name is not found in the expression store.
    """
    # Load cluster labels
    cluster_df = sj.maps[cluster_key].compute()
    value_cols = [c for c in cluster_df.columns if c != "cell_id"]
    cluster_col = value_cols[0]
    cluster_df = cluster_df.rename(columns={cluster_col: "cluster"})

    # Load gene expression
    gene_names = list(sj.expression.gene_names)
    gene_indices = []
    for g in genes:
        if g not in gene_names:
            raise ValueError(f"Gene {g!r} not found in expression store. Available: {gene_names[:5]}...")
        gene_indices.append(gene_names.index(g))

    expr_matrix = sj.expression.select_genes(gene_indices).compute()  # (n_cells, len(genes))

    # Build cell_id alignment
    cell_ids = sj.expression.cell_ids
    if cell_ids is None:
        cells_df = sj.cells.df.compute()
        cell_ids = cells_df["cell_id"].values

    # Wide expression DataFrame
    expr_df = pd.DataFrame(expr_matrix, columns=genes)
    expr_df["cell_id"] = cell_ids

    # Merge with clusters
    merged = expr_df.merge(cluster_df[["cell_id", "cluster"]], on="cell_id", how="inner")

    # Melt to long form
    long = merged.melt(
        id_vars=["cell_id", "cluster"],
        value_vars=genes,
        var_name="gene",
        value_name="expression",
    )
    return long


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


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
    """Heatmap of mean expression per cluster for selected genes.

    Args:
        sj: Dataset instance.
        genes: List of gene names to display.
        cluster_key: Maps key for cluster labels (default ``"leiden"``).
        cmap: Matplotlib colormap name.
        vmin: Minimum value for colour scale.
        vmax: Maximum value for colour scale.
        standardize: If True, z-score each gene across clusters and
            auto-switch cmap to ``"RdBu_r"`` when cmap is ``"magma"``.
        figsize: Figure size; auto-computed if None.
        ax: Existing axes to draw on, or None.
        save: Whether to save the figure.
        save_path: Override save path.
        dpi: Resolution for saved figure.
        fmt: File format for saved figure.

    Returns:
        The matplotlib Axes with the heatmap.

    Raises:
        ValueError: If a gene is not found in the expression store.

    Example:
        >>> ax = heatmap(sj, ["gene_0", "gene_1", "gene_2"])
    """
    long = _build_cluster_gene_df(sj, genes, cluster_key)

    # Pivot to matrix: rows = clusters, cols = genes
    pivot = long.pivot_table(index="cluster", columns="gene", values="expression", aggfunc="mean")
    # Reorder columns to match input gene order
    pivot = pivot[genes]

    matrix = pivot.values.astype(float)

    if standardize:
        # Z-score each gene (column) across clusters
        col_means = matrix.mean(axis=0, keepdims=True)
        col_stds = matrix.std(axis=0, keepdims=True)
        col_stds[col_stds == 0] = 1.0
        matrix = (matrix - col_means) / col_stds
        if cmap == "magma":
            cmap = "RdBu_r"

    if figsize is None:
        figsize = (max(4, len(genes) * 0.6 + 2), max(3, len(pivot) * 0.5 + 1.5))

    fig, ax = _setup_ax(ax, figsize=figsize)

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=90, ha="center", fontsize=8)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Gene")
    ax.set_ylabel("Cluster")

    fig.tight_layout()
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
    """Violin plots of gene expression by cluster.

    Creates a multi-panel grid with one subplot per gene.

    Args:
        sj: Dataset instance.
        genes: List of gene names to display.
        cluster_key: Maps key for cluster labels (default ``"leiden"``).
        palette: Colour palette for clusters. Auto-generated if None.
        ncols: Number of columns in the subplot grid.
        figsize: Figure size; auto-computed if None.
        save: Whether to save the figure.
        save_path: Override save path.
        dpi: Resolution for saved figure.
        fmt: File format for saved figure.

    Returns:
        The matplotlib Figure containing the violin subplots.

    Raises:
        ImportError: If seaborn is not installed.
        ValueError: If a gene is not found in the expression store.

    Example:
        >>> fig = violin(sj, ["gene_0", "gene_1", "gene_2"])
    """
    try:
        import seaborn as sns
    except ImportError:
        raise ImportError("Install seaborn for violin plots: pip install seaborn")

    long = _build_cluster_gene_df(sj, genes, cluster_key)

    n_genes = len(genes)
    nrows = max(1, (n_genes + ncols - 1) // ncols)

    if figsize is None:
        figsize = (ncols * 3.5, nrows * 3)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    clusters = sorted(long["cluster"].unique())
    if palette is None:
        palette = _categorical_palette(len(clusters))

    for i, gene in enumerate(genes):
        gene_data = long[long["gene"] == gene]
        sns.violinplot(
            data=gene_data,
            x="cluster",
            y="expression",
            hue="cluster",
            order=clusters,
            hue_order=clusters,
            palette=palette[: len(clusters)],
            ax=axes_flat[i],
            inner="box",
            linewidth=0.8,
            legend=False,
        )
        axes_flat[i].set_title(gene, fontsize=10)
        axes_flat[i].set_xlabel("")
        axes_flat[i].set_ylabel("Expression" if i % ncols == 0 else "")

    # Hide unused axes
    for j in range(n_genes, len(axes_flat)):
        axes_flat[j].set_visible(False)

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
    """Dot plot of gene expression by cluster.

    Dot size encodes the fraction of cells expressing each gene (expression > 0),
    and dot colour encodes mean expression.

    Args:
        sj: Dataset instance.
        genes: List of gene names to display (x-axis).
        cluster_key: Maps key for cluster labels (default ``"leiden"``).
        cmap: Matplotlib colormap for mean expression.
        vmin: Minimum value for colour scale.
        vmax: Maximum value for colour scale.
        max_dot_size: Maximum dot size in points squared.
        figsize: Figure size; auto-computed if None.
        ax: Existing axes to draw on, or None.
        save: Whether to save the figure.
        save_path: Override save path.
        dpi: Resolution for saved figure.
        fmt: File format for saved figure.

    Returns:
        The matplotlib Axes with the dot plot.

    Raises:
        ValueError: If a gene is not found in the expression store.

    Example:
        >>> ax = dotplot(sj, ["gene_0", "gene_1", "gene_2"])
    """
    long = _build_cluster_gene_df(sj, genes, cluster_key)

    # Compute summary statistics per cluster x gene
    summary = long.groupby(["cluster", "gene"]).agg(
        mean_expr=("expression", "mean"),
        frac_expressing=("expression", lambda x: (x > 0).mean()),
    ).reset_index()

    clusters = sorted(summary["cluster"].unique())

    if figsize is None:
        figsize = (max(4, len(genes) * 0.8 + 2), max(3, len(clusters) * 0.5 + 2))

    fig, ax = _setup_ax(ax, figsize=figsize)

    # Map genes/clusters to numeric positions
    gene_to_x = {g: i for i, g in enumerate(genes)}
    cluster_to_y = {c: i for i, c in enumerate(clusters)}

    xs = summary["gene"].map(gene_to_x).values
    ys = summary["cluster"].map(cluster_to_y).values
    sizes = summary["frac_expressing"].values * max_dot_size
    colors = summary["mean_expr"].values

    scatter = ax.scatter(
        xs, ys,
        s=sizes,
        c=colors,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
    )

    # Colorbar for mean expression
    fig.colorbar(scatter, ax=ax, label="Mean expression", shrink=0.8)

    # Size legend
    legend_fracs = [0.25, 0.5, 0.75, 1.0]
    legend_handles = []
    for frac in legend_fracs:
        h = ax.scatter([], [], s=frac * max_dot_size, c="grey", edgecolors="none")
        legend_handles.append(h)
    ax.legend(
        legend_handles,
        [f"{int(f * 100)}%" for f in legend_fracs],
        title="% expressing",
        loc="center left",
        bbox_to_anchor=(1.25, 0.5),
        frameon=False,
        scatterpoints=1,
    )

    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=90, ha="center", fontsize=8)
    ax.set_yticks(range(len(clusters)))
    ax.set_yticklabels(clusters, fontsize=8)
    ax.set_xlabel("Gene")
    ax.set_ylabel("Cluster")

    fig.tight_layout()
    _save_figure(fig, sj, f"dotplot_{cluster_key}", save=save, save_path=save_path, dpi=dpi, fmt=fmt)
    return ax
