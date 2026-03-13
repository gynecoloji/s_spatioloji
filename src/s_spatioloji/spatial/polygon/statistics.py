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
    cell_to_cluster = dict(zip(cell_ids, labels, strict=True))

    unique_labels = sorted(set(labels), key=str)
    pairs = list(combinations_with_replacement(unique_labels, 2))
    pair_keys = [tuple(sorted([str(a), str(b)])) for a, b in pairs]

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

    observed = _count_edges(cell_to_cluster)

    rng = np.random.default_rng(random_state)
    perm_counts = {pk: [] for pk in pair_keys}

    for _ in range(n_permutations):
        shuffled_labels = rng.permutation(labels)
        perm_c2c = dict(zip(cell_ids, shuffled_labels, strict=True))
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

        p = (np.sum(null_dist >= obs) + 1) / (n_permutations + 1)

        records.append(
            {
                "cluster_a": pk[0],
                "cluster_b": pk[1],
                "observed_ratio": float(obs),
                "p_value": float(p),
                "z_score": float(z),
            }
        )

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

    merged = cells_df[["cell_id", "x", "y"]].merge(cluster_df[["cell_id", cluster_key]], on="cell_id")

    x = merged["x"].values
    y = merged["y"].values
    labels = merged[cluster_key].values

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

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

        counts = np.zeros((n_bins, n_bins))
        for xi, yi in zip(x_b, y_b, strict=True):
            counts[xi, yi] += 1

        flat = counts.flatten()
        expected = np.full_like(flat, flat.mean())

        if expected[0] > 0:
            chi2, p = stats.chisquare(flat, f_exp=expected)
        else:
            chi2, p = 0.0, 1.0

        records.append(
            {
                "cluster": label,
                "chi2": float(chi2),
                "p_value": float(p),
                "density_mean": float(flat.mean()),
                "density_std": float(flat.std()),
            }
        )

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
