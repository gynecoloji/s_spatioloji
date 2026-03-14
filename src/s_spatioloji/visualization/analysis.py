"""Analysis result plots: Ripley, colocalization, neighborhood, envelope.

All functions accept an ``s_spatioloji`` instance, return
``matplotlib.axes.Axes``, and auto-save to ``{dataset_root}/figures/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from s_spatioloji.visualization._common import _categorical_palette, _save_figure, _setup_ax

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def ripley_plot(
    sj: s_spatioloji,
    function: str = "K",
    input_key: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8, 6),
    save: bool = True,
    save_path: Path | str | None = None,
    dpi: int = 200,
) -> Axes:
    """Plot Ripley's K, L, G, or F function results.

    Draws one line per cluster plus a theoretical reference (dashed).
    For K: ``K_theo`` column. For L: horizontal line at 0.
    For G/F: ``G_theo`` / ``F_theo`` column.

    Args:
        sj: Dataset instance with precomputed Ripley results in maps/.
        function: One of ``"K"``, ``"L"``, ``"G"``, ``"F"``.
        input_key: Maps key to load. Defaults to ``"pt_ripley_{function.lower()}"``.
        ax: Existing axes to reuse, or ``None`` to create new.
        figsize: Figure size if creating new axes.
        save: If ``True``, save figure to ``figures/``.
        save_path: Override save path.
        dpi: Resolution in dots per inch.

    Returns:
        The matplotlib Axes with the plot.

    Raises:
        ValueError: If *function* is not one of K, L, G, F.
        FileNotFoundError: If the Ripley result is not found in maps/.

    Example:
        >>> from s_spatioloji.spatial.point.ripley import ripley_k
        >>> ripley_k(sj, n_radii=50)
        >>> ax = ripley_plot(sj, function="K")
    """
    function = function.upper()
    if function not in ("K", "L", "G", "F"):
        raise ValueError(f"Unknown function: {function!r}. Must be one of 'K', 'L', 'G', 'F'.")

    if input_key is None:
        input_key = f"pt_ripley_{function.lower()}"

    df = sj.maps[input_key].compute()
    fig, ax = _setup_ax(ax, figsize)

    clusters = sorted(df["cluster"].unique(), key=str)
    colors = _categorical_palette(len(clusters))

    for i, cluster in enumerate(clusters):
        sub = df[df["cluster"] == cluster].sort_values("r")
        ax.plot(sub["r"], sub[function], color=colors[i], label=f"{cluster}")

    # Theoretical reference
    if function == "L":
        ax.axhline(0, color="black", linestyle="--", linewidth=1, label="CSR (L=0)")
    else:
        theo_col = f"{function}_theo"
        if theo_col in df.columns:
            # Use the first cluster's theoretical values (they are identical)
            sub0 = df[df["cluster"] == clusters[0]].sort_values("r")
            ax.plot(sub0["r"], sub0[theo_col], color="black", linestyle="--", linewidth=1, label="CSR")

    ax.set_xlabel("r")
    ax.set_ylabel(f"{function}(r)")
    ax.set_title(f"Ripley's {function}")
    ax.legend(fontsize="small")

    _save_figure(fig, sj, f"ripley_{function.lower()}", save=save, save_path=save_path, dpi=dpi)
    return ax


def colocalization_heatmap(
    sj: s_spatioloji,
    metric: str = "log2_ratio",
    input_key: str = "pt_colocalization",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8, 7),
    cmap: str = "RdBu_r",
    save: bool = True,
    save_path: Path | str | None = None,
    dpi: int = 200,
) -> Axes:
    """Heatmap of pairwise cluster colocalization.

    Pivots ``cluster_a x cluster_b`` into a symmetric matrix using
    the chosen *metric* column. Plots with ``imshow`` and a colorbar.

    Args:
        sj: Dataset instance with precomputed colocalization in maps/.
        metric: Column to use as values (``"log2_ratio"``, ``"ratio"``, etc.).
        input_key: Maps key to load.
        ax: Existing axes to reuse, or ``None`` to create new.
        figsize: Figure size if creating new axes.
        cmap: Matplotlib colormap name.
        save: If ``True``, save figure to ``figures/``.
        save_path: Override save path.
        dpi: Resolution in dots per inch.

    Returns:
        The matplotlib Axes with the heatmap.

    Example:
        >>> from s_spatioloji.spatial.point.patterns import colocalization
        >>> colocalization(sj)
        >>> ax = colocalization_heatmap(sj)
    """
    df = sj.maps[input_key].compute()

    # Build symmetric matrix
    labels = sorted(set(df["cluster_a"].tolist() + df["cluster_b"].tolist()), key=str)
    n = len(labels)
    label_to_idx = {str(lab): i for i, lab in enumerate(labels)}

    matrix = np.full((n, n), np.nan)
    for _, row in df.iterrows():
        i = label_to_idx[str(row["cluster_a"])]
        j = label_to_idx[str(row["cluster_b"])]
        val = row[metric]
        matrix[i, j] = val
        matrix[j, i] = val

    fig, ax = _setup_ax(ax, figsize)

    # Determine symmetric color limits
    finite_vals = matrix[np.isfinite(matrix)]
    if len(finite_vals) > 0:
        vabs = max(abs(finite_vals.min()), abs(finite_vals.max()))
    else:
        vabs = 1.0

    im = ax.imshow(matrix, cmap=cmap, vmin=-vabs, vmax=vabs, aspect="equal")
    fig.colorbar(im, ax=ax, label=metric)

    tick_labels = [str(lab) for lab in labels]
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cluster")
    ax.set_title(f"Colocalization ({metric})")

    _save_figure(fig, sj, f"colocalization_{metric}", save=save, save_path=save_path, dpi=dpi)
    return ax


def neighborhood_bar(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    input_key: str = "pt_nhood_composition",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8, 6),
    save: bool = True,
    save_path: Path | str | None = None,
    dpi: int = 200,
) -> Axes:
    """Stacked horizontal bar chart of mean neighborhood composition per cluster.

    Loads per-cell composition from *input_key*, merges with cluster labels
    from *cluster_key*, computes the mean proportion for each cluster, and
    draws a stacked horizontal bar chart.

    Args:
        sj: Dataset instance with precomputed neighborhood composition.
        cluster_key: Maps key for cluster labels.
        input_key: Maps key for neighborhood composition table.
        ax: Existing axes to reuse, or ``None`` to create new.
        figsize: Figure size if creating new axes.
        save: If ``True``, save figure to ``figures/``.
        save_path: Override save path.
        dpi: Resolution in dots per inch.

    Returns:
        The matplotlib Axes with the bar chart.

    Example:
        >>> from s_spatioloji.spatial.point.neighborhoods import neighborhood_composition
        >>> neighborhood_composition(sj)
        >>> ax = neighborhood_bar(sj)
    """
    comp_df = sj.maps[input_key].compute()
    cluster_df = sj.maps[cluster_key].compute()

    # Merge to get each cell's own cluster label
    merged = comp_df.merge(cluster_df[["cell_id", cluster_key]], on="cell_id")

    # Neighbor-type columns are everything except cell_id and cluster_key
    neighbor_cols = [c for c in comp_df.columns if c != "cell_id"]

    # Group by the cell's cluster and compute mean counts
    grouped = merged.groupby(cluster_key)[neighbor_cols].mean()

    # Normalize rows to proportions
    row_sums = grouped.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    proportions = grouped.div(row_sums, axis=0)

    fig, ax = _setup_ax(ax, figsize)
    colors = _categorical_palette(len(neighbor_cols))

    cluster_labels = [str(c) for c in proportions.index]
    y_pos = np.arange(len(cluster_labels))
    lefts = np.zeros(len(cluster_labels))

    for j, col in enumerate(neighbor_cols):
        widths = proportions[col].values
        ax.barh(y_pos, widths, left=lefts, color=colors[j], label=str(col), edgecolor="white", linewidth=0.5)
        lefts += widths

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cluster_labels)
    ax.set_xlabel("Proportion")
    ax.set_ylabel("Cluster")
    ax.set_title("Neighborhood composition")
    ax.legend(title="Neighbor type", fontsize="small", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()

    _save_figure(fig, sj, "neighborhood_bar", save=save, save_path=save_path, dpi=dpi)
    return ax


def envelope_plot(
    sj: s_spatioloji,
    input_key: str = "pt_dclf_envelope",
    ax: Axes | None = None,
    figsize: tuple[float, float] = (8, 6),
    save: bool = True,
    save_path: Path | str | None = None,
    dpi: int = 200,
) -> Axes:
    """Envelope plot for DCLF global envelope test results.

    Draws the observed line, a fill_between band (lo--hi), and a
    theoretical dashed reference line. Shows the p-value in the label.

    Args:
        sj: Dataset instance with precomputed envelope results.
        input_key: Maps key for envelope results.
        ax: Existing axes to reuse, or ``None`` to create new.
        figsize: Figure size if creating new axes.
        save: If ``True``, save figure to ``figures/``.
        save_path: Override save path.
        dpi: Resolution in dots per inch.

    Returns:
        The matplotlib Axes with the envelope plot.

    Example:
        >>> from s_spatioloji.spatial.point.statistics import dclf_envelope
        >>> dclf_envelope(sj, n_simulations=199)
        >>> ax = envelope_plot(sj)
    """
    df = sj.maps[input_key].compute()
    fig, ax = _setup_ax(ax, figsize)

    clusters = sorted(df["cluster"].unique(), key=str)
    colors = _categorical_palette(len(clusters))

    for i, cluster in enumerate(clusters):
        sub = df[df["cluster"] == cluster].sort_values("r")
        r = sub["r"].values
        observed = sub["observed"].values
        lo = sub["lo"].values
        hi = sub["hi"].values
        theo = sub["theo"].values
        p_val = sub["p_value"].iloc[0]

        label_obs = f"{cluster} observed (p={p_val:.3f})"
        ax.plot(r, observed, color=colors[i], linewidth=1.5, label=label_obs)
        ax.fill_between(r, lo, hi, color=colors[i], alpha=0.2, label=f"{cluster} envelope")
        ax.plot(r, theo, color=colors[i], linestyle="--", linewidth=1, label=f"{cluster} CSR")

    ax.set_xlabel("r")
    ax.set_ylabel("Function value")
    ax.set_title("DCLF envelope test")
    ax.legend(fontsize="small")

    _save_figure(fig, sj, f"envelope_{input_key}", save=save, save_path=save_path, dpi=dpi)
    return ax
