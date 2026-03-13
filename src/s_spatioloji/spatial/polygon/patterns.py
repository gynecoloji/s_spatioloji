"""Spatial pattern analysis functions for polygon-based spatial data.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from s_spatioloji.compute import _atomic_write_parquet
from s_spatioloji.spatial.polygon.graph import _load_contact_graph

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def colocalization(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "contact_graph",
    output_key: str = "colocalization",
    force: bool = True,
) -> str:
    """Observed vs expected contact frequency for cluster pairs.

    Args:
        sj: Dataset instance.
        cluster_key: Key for cluster labels in maps/.
        graph_key: Key for the contact graph edge list.
        output_key: Key to write colocalization table under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> colocalization(sj, cluster_key="leiden")
        'colocalization'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    G = _load_contact_graph(sj, graph_key)

    cluster_df = sj.maps[cluster_key].compute()
    cell_to_cluster = dict(zip(cluster_df["cell_id"], cluster_df[cluster_key], strict=True))
    cluster_labels = sorted(set(cell_to_cluster.values()), key=str)

    cluster_sizes: dict = defaultdict(int)
    for label in cell_to_cluster.values():
        cluster_sizes[label] += 1
    n_total = sum(cluster_sizes.values())
    total_edges = G.number_of_edges()

    observed: dict = defaultdict(int)
    for u, v in G.edges():
        if u in cell_to_cluster and v in cell_to_cluster:
            a, b = cell_to_cluster[u], cell_to_cluster[v]
            pair = tuple(sorted([str(a), str(b)]))
            observed[pair] += 1

    records = []
    for a, b in combinations_with_replacement(cluster_labels, 2):
        pair = tuple(sorted([str(a), str(b)]))
        obs = observed.get(pair, 0)
        n_a = cluster_sizes[a]
        n_b = cluster_sizes[b]

        if a == b:
            exp = n_a * (n_a - 1) * total_edges / max(n_total * (n_total - 1), 1)
        else:
            exp = 2 * n_a * n_b * total_edges / max(n_total * (n_total - 1), 1)

        ratio = obs / exp if exp > 0 else 0.0
        log2_ratio = np.log2(ratio) if ratio > 0 else float("nan")

        records.append(
            {
                "cluster_a": pair[0],
                "cluster_b": pair[1],
                "observed": obs,
                "expected": exp,
                "ratio": ratio,
                "log2_ratio": log2_ratio,
            }
        )

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def _compute_morans_i(values: np.ndarray, G: nx.Graph, nodes: list[str]) -> dict:
    """Compute Moran's I for a single numeric vector.

    Args:
        values: 1-D array of numeric values aligned with ``nodes``.
        G: Contact graph.
        nodes: Cell IDs corresponding to ``values``.

    Returns:
        Dict with keys: I, expected_I, z_score, p_value.
    """
    N = len(values)
    if N < 3:
        return {"I": 0.0, "expected_I": 0.0, "z_score": 0.0, "p_value": 1.0}

    x_bar = values.mean()
    dev = values - x_bar
    denom = np.sum(dev**2)
    if denom == 0:
        return {"I": 0.0, "expected_I": 0.0, "z_score": 0.0, "p_value": 1.0}

    node_idx = {n: i for i, n in enumerate(nodes)}
    W = 0.0
    numerator = 0.0
    S1 = 0.0
    degree_sums = np.zeros(N)

    for u, v in G.edges():
        if u in node_idx and v in node_idx:
            i, j = node_idx[u], node_idx[v]
            W += 2  # binary symmetric: w_ij = w_ji = 1
            numerator += dev[i] * dev[j]
            S1 += 2  # 0.5 * (w_ij + w_ji)^2 = 2 per edge
            degree_sums[i] += 1
            degree_sums[j] += 1

    if W == 0:
        return {"I": 0.0, "expected_I": 0.0, "z_score": 0.0, "p_value": 1.0}

    numerator *= 2  # count both directions
    moran_I = (N / W) * numerator / denom  # noqa: E741
    expected_I = -1.0 / (N - 1)

    # Randomization assumption variance
    S1_total = S1  # accumulated from the loop
    S2 = np.sum((2 * degree_sums) ** 2)
    k = N * np.sum(dev**4) / denom**2  # kurtosis

    num1 = N * (S1_total * (N**2 - 3 * N + 3) - N * S2 + 3 * W**2)
    num2 = k * (S1_total * (N**2 - N) - 2 * N * S2 + 6 * W**2)
    var_I = (num1 - num2) / ((N - 1) * (N - 2) * (N - 3) * W**2) - expected_I**2
    var_I = max(var_I, 1e-15)

    z = (moran_I - expected_I) / np.sqrt(var_I)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    return {"I": float(moran_I), "expected_I": float(expected_I), "z_score": float(z), "p_value": float(p)}


def morans_i(
    sj: s_spatioloji,
    feature_key: str = "leiden",
    graph_key: str = "contact_graph",
    output_key: str = "morans_i",
    force: bool = True,
) -> str:
    """Moran's I spatial autocorrelation.

    For categorical features (dtype object/category), computes one-hot
    indicators and returns one row per category.  Integer labels are
    treated as numeric and produce a single row.

    Args:
        sj: Dataset instance.
        feature_key: Key for the feature in maps/.
        graph_key: Key for the contact graph edge list.
        output_key: Key to write Moran's I results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> morans_i(sj, feature_key="leiden")
        'morans_i'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    G = _load_contact_graph(sj, graph_key)

    feat_df = sj.maps[feature_key].compute()
    cell_ids = list(feat_df["cell_id"])
    value_cols = [c for c in feat_df.columns if c != "cell_id"]

    if len(value_cols) != 1:
        raise ValueError(f"Expected exactly one feature column besides cell_id, got {len(value_cols)}")

    col = value_cols[0]
    values = feat_df[col]

    records = []
    if values.dtype == object or hasattr(values, "cat"):
        # Categorical: one-hot encoding
        categories = sorted(values.unique(), key=str)
        for cat in categories:
            indicator = (values == cat).astype(float).values
            result = _compute_morans_i(indicator, G, cell_ids)
            result["feature"] = str(cat)
            records.append(result)
    else:
        result = _compute_morans_i(values.values.astype(float), G, cell_ids)
        result["feature"] = col
        records.append(result)

    df = pd.DataFrame(records)[["feature", "I", "expected_I", "z_score", "p_value"]]
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def _compute_gearys_c(values: np.ndarray, G: nx.Graph, nodes: list[str]) -> dict:
    """Compute Geary's C for a single numeric vector.

    Args:
        values: 1-D array of numeric values aligned with ``nodes``.
        G: Contact graph.
        nodes: Cell IDs corresponding to ``values``.

    Returns:
        Dict with keys: C, expected_C, z_score, p_value.
    """
    N = len(values)
    if N < 3:
        return {"C": 1.0, "expected_C": 1.0, "z_score": 0.0, "p_value": 1.0}

    x_bar = values.mean()
    dev = values - x_bar
    denom = np.sum(dev**2)
    if denom == 0:
        return {"C": 1.0, "expected_C": 1.0, "z_score": 0.0, "p_value": 1.0}

    node_idx = {n: i for i, n in enumerate(nodes)}
    W = 0.0
    numerator = 0.0

    for u, v in G.edges():
        if u in node_idx and v in node_idx:
            i, j = node_idx[u], node_idx[v]
            W += 2  # binary symmetric
            numerator += (values[i] - values[j]) ** 2

    if W == 0:
        return {"C": 1.0, "expected_C": 1.0, "z_score": 0.0, "p_value": 1.0}

    C = ((N - 1) / (2 * W)) * numerator / denom
    expected_C = 1.0

    # Randomization assumption variance
    degree_sums = np.zeros(N)
    edge_count = 0
    for u, v in G.edges():
        if u in node_idx and v in node_idx:
            degree_sums[node_idx[u]] += 1
            degree_sums[node_idx[v]] += 1
            edge_count += 1

    S1 = 2.0 * edge_count
    S2 = np.sum((2 * degree_sums) ** 2)
    k = N * np.sum(dev**4) / denom**2

    num1 = (N - 1) * S1 * (N**2 - 3 * N + 3 - (N - 1) * k)
    num2 = (1 / 4) * (N - 1) * S2 * (N**2 + 3 * N - 6 - (N**2 - N + 2) * k)
    num3 = W**2 * (N**2 - 3 - (N - 1) ** 2 * k)
    var_C = (num1 - num2 + num3) / ((N - 1) * (N - 2) * (N - 3) * W**2)
    var_C = max(var_C, 1e-15)

    z = (C - expected_C) / np.sqrt(var_C)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    return {"C": float(C), "expected_C": expected_C, "z_score": float(z), "p_value": float(p)}


def gearys_c(
    sj: s_spatioloji,
    feature_key: str = "leiden",
    graph_key: str = "contact_graph",
    output_key: str = "gearys_c",
    force: bool = True,
) -> str:
    """Geary's C spatial autocorrelation.

    Same interface and categorical handling as :func:`morans_i`.

    Args:
        sj: Dataset instance.
        feature_key: Key for the feature in maps/.
        graph_key: Key for the contact graph edge list.
        output_key: Key to write Geary's C results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> gearys_c(sj, feature_key="leiden")
        'gearys_c'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    G = _load_contact_graph(sj, graph_key)

    feat_df = sj.maps[feature_key].compute()
    cell_ids = list(feat_df["cell_id"])
    value_cols = [c for c in feat_df.columns if c != "cell_id"]

    if len(value_cols) != 1:
        raise ValueError(f"Expected exactly one feature column besides cell_id, got {len(value_cols)}")

    col = value_cols[0]
    values = feat_df[col]

    records = []
    if values.dtype == object or hasattr(values, "cat"):
        categories = sorted(values.unique(), key=str)
        for cat in categories:
            indicator = (values == cat).astype(float).values
            result = _compute_gearys_c(indicator, G, cell_ids)
            result["feature"] = str(cat)
            records.append(result)
    else:
        result = _compute_gearys_c(values.values.astype(float), G, cell_ids)
        result["feature"] = col
        records.append(result)

    df = pd.DataFrame(records)[["feature", "C", "expected_C", "z_score", "p_value"]]
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def clustering_coefficient(
    sj: s_spatioloji,
    graph_key: str = "contact_graph",
    output_key: str = "clustering_coeff",
    force: bool = True,
) -> str:
    """Per-cell local clustering coefficient.

    Args:
        sj: Dataset instance.
        graph_key: Key for the contact graph edge list.
        output_key: Key to write clustering coefficients under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> clustering_coefficient(sj)
        'clustering_coeff'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    G = _load_contact_graph(sj, graph_key)

    cells_df = sj.cells.df.compute()
    all_cell_ids = list(cells_df["cell_id"])

    cc = nx.clustering(G)
    records = [{"cell_id": cid, "clustering_coeff": cc.get(cid, 0.0)} for cid in all_cell_ids]

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def border_enrichment(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "contact_graph",
    output_key: str = "border_enrichment",
    force: bool = True,
) -> str:
    """Identify border cells and compute enrichment per cluster.

    A cell is a border cell if at least one neighbor belongs to a
    different cluster.

    Args:
        sj: Dataset instance.
        cluster_key: Key for cluster labels in maps/.
        graph_key: Key for the contact graph edge list.
        output_key: Key to write border enrichment under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> border_enrichment(sj, cluster_key="leiden")
        'border_enrichment'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    G = _load_contact_graph(sj, graph_key)

    cluster_df = sj.maps[cluster_key].compute()
    cell_to_cluster = dict(zip(cluster_df["cell_id"], cluster_df[cluster_key], strict=True))

    cluster_cells: dict = defaultdict(list)
    for cid, label in cell_to_cluster.items():
        cluster_cells[label].append(cid)

    n_total = len(cell_to_cluster)
    k_avg = 2.0 * G.number_of_edges() / max(n_total, 1)

    records = []
    for label in sorted(cluster_cells.keys(), key=str):
        cells = cluster_cells[label]
        n_cells = len(cells)
        n_border = 0
        for cid in cells:
            if G.has_node(cid):
                for neighbor in G.neighbors(cid):
                    if neighbor in cell_to_cluster and cell_to_cluster[neighbor] != label:
                        n_border += 1
                        break

        border_fraction = n_border / n_cells if n_cells > 0 else 0.0
        expected_fraction = 1.0 - (n_cells / max(n_total, 1)) ** k_avg if k_avg > 0 else 0.0
        enrichment = border_fraction / expected_fraction if expected_fraction > 0 else 0.0

        records.append(
            {
                "cluster": label,
                "n_cells": n_cells,
                "n_border": n_border,
                "border_fraction": border_fraction,
                "enrichment": enrichment,
            }
        )

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
