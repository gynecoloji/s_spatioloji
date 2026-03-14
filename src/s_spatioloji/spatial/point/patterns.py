"""Spatial pattern analysis functions for point-based spatial data.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.

Uses sparse matrix operations for scalability to 50-100M cells.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations_with_replacement
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse
from scipy import stats

from s_spatioloji.compute import _atomic_write_parquet
from s_spatioloji.spatial.point.graph import _load_point_graph, _load_point_graph_sparse

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def colocalization(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "knn_graph",
    output_key: str = "pt_colocalization",
    force: bool = True,
) -> str:
    """Observed vs expected contact frequency for cluster pairs.

    Args:
        sj: Dataset instance.
        cluster_key: Key for cluster labels in maps/.
        graph_key: Key for the graph edge list.
        output_key: Key to write colocalization table under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> colocalization(sj, cluster_key="leiden")
        'pt_colocalization'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)

    cluster_df = sj.maps[cluster_key].compute()
    cell_to_cluster = dict(zip(cluster_df["cell_id"], cluster_df[cluster_key], strict=True))
    cluster_labels = sorted(set(cell_to_cluster.values()), key=str)

    cluster_sizes: dict = defaultdict(int)
    for label in cell_to_cluster.values():
        cluster_sizes[label] += 1
    n_total = sum(cluster_sizes.values())

    edge_df = sj.maps[graph_key].compute()
    total_edges = len(edge_df)

    observed: dict = defaultdict(int)
    for _, row in edge_df.iterrows():
        a_id, b_id = row["cell_id_a"], row["cell_id_b"]
        if a_id in cell_to_cluster and b_id in cell_to_cluster:
            la, lb = cell_to_cluster[a_id], cell_to_cluster[b_id]
            pair = tuple(sorted([str(la), str(lb)]))
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


def _compute_morans_i_sparse(values: np.ndarray, adj: scipy.sparse.csr_matrix, n: int) -> dict:
    """Compute Moran's I using sparse adjacency matrix.

    Args:
        values: 1-D numeric array (length n).
        adj: Sparse binary adjacency matrix (n x n).
        n: Number of cells.

    Returns:
        Dict with I, expected_I, z_score, p_value.
    """
    if n < 3:
        return {"I": 0.0, "expected_I": 0.0, "z_score": 0.0, "p_value": 1.0}

    x_bar = values.mean()
    z = values - x_bar
    denom = np.sum(z**2)
    if denom == 0:
        return {"I": 0.0, "expected_I": 0.0, "z_score": 0.0, "p_value": 1.0}

    W = float(adj.nnz)
    if W == 0:
        return {"I": 0.0, "expected_I": 0.0, "z_score": 0.0, "p_value": 1.0}

    Wz = adj @ z
    numerator = float(z @ Wz)
    moran_I = (n / W) * numerator / denom

    expected_I = -1.0 / (n - 1)

    ones = np.ones(n)
    degree = np.asarray(adj @ ones).ravel()
    S1 = float(adj.nnz)
    S2 = float(np.sum((2 * degree) ** 2))
    k = n * np.sum(z**4) / denom**2

    num1 = n * (S1 * (n**2 - 3 * n + 3) - n * S2 + 3 * W**2)
    num2 = k * (S1 * (n**2 - n) - 2 * n * S2 + 6 * W**2)
    var_I = (num1 - num2) / ((n - 1) * (n - 2) * (n - 3) * W**2) - expected_I**2
    var_I = max(var_I, 1e-15)

    z_score = (moran_I - expected_I) / np.sqrt(var_I)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

    return {
        "I": float(moran_I),
        "expected_I": float(expected_I),
        "z_score": float(z_score),
        "p_value": float(p_value),
    }


def morans_i(
    sj: s_spatioloji,
    feature_key: str = "leiden",
    graph_key: str = "knn_graph",
    output_key: str = "pt_morans_i",
    force: bool = True,
) -> str:
    """Moran's I spatial autocorrelation using sparse matrix operations.

    For categorical features (dtype object/category), computes one-hot
    indicators and returns one row per category.

    Args:
        sj: Dataset instance.
        feature_key: Key for the feature in maps/.
        graph_key: Key for the graph edge list.
        output_key: Key to write Moran's I results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> morans_i(sj, feature_key="leiden")
        'pt_morans_i'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    adj, graph_cell_ids = _load_point_graph_sparse(sj, graph_key, weighted=False)

    feat_df = sj.maps[feature_key].compute()
    value_cols = [c for c in feat_df.columns if c != "cell_id"]
    if len(value_cols) != 1:
        raise ValueError(f"Expected exactly one feature column besides cell_id, got {len(value_cols)}")

    col = value_cols[0]
    values_series = feat_df.set_index("cell_id")[col]
    aligned = values_series.reindex(graph_cell_ids)
    n = len(graph_cell_ids)

    records = []
    if aligned.dtype == object or hasattr(aligned, "cat"):
        categories = sorted(aligned.dropna().unique(), key=str)
        for cat in categories:
            indicator = (aligned == cat).astype(float).values
            result = _compute_morans_i_sparse(indicator, adj, n)
            result["feature"] = str(cat)
            records.append(result)
    else:
        vals = aligned.values.astype(float)
        result = _compute_morans_i_sparse(vals, adj, n)
        result["feature"] = col
        records.append(result)

    df = pd.DataFrame(records)[["feature", "I", "expected_I", "z_score", "p_value"]]
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def _compute_gearys_c_sparse(
    values: np.ndarray, adj: scipy.sparse.csr_matrix, edge_df: pd.DataFrame, id_to_idx: dict, n: int
) -> dict:
    """Compute Geary's C using edge list for pairwise differences.

    Args:
        values: 1-D numeric array (length n).
        adj: Sparse binary adjacency matrix.
        edge_df: Edge list DataFrame with cell_id_a, cell_id_b.
        id_to_idx: Mapping from cell_id to matrix index.
        n: Number of cells.

    Returns:
        Dict with C, expected_C, z_score, p_value.
    """
    if n < 3:
        return {"C": 1.0, "expected_C": 1.0, "z_score": 0.0, "p_value": 1.0}

    x_bar = values.mean()
    dev = values - x_bar
    denom = np.sum(dev**2)
    if denom == 0:
        return {"C": 1.0, "expected_C": 1.0, "z_score": 0.0, "p_value": 1.0}

    W = float(adj.nnz)
    if W == 0:
        return {"C": 1.0, "expected_C": 1.0, "z_score": 0.0, "p_value": 1.0}

    rows_a = edge_df["cell_id_a"].map(id_to_idx).values
    rows_b = edge_df["cell_id_b"].map(id_to_idx).values
    valid = ~(np.isnan(rows_a.astype(float)) | np.isnan(rows_b.astype(float)))
    rows_a = rows_a[valid].astype(int)
    rows_b = rows_b[valid].astype(int)

    sq_diffs = (values[rows_a] - values[rows_b]) ** 2
    numerator = float(sq_diffs.sum()) * 2  # both directions

    C = ((n - 1) / (2 * W)) * numerator / denom
    expected_C = 1.0

    ones = np.ones(n)
    degree = np.asarray(adj @ ones).ravel()
    S1 = float(adj.nnz)
    S2 = float(np.sum((2 * degree) ** 2))
    k = n * np.sum(dev**4) / denom**2

    num1 = (n - 1) * S1 * (n**2 - 3 * n + 3 - (n - 1) * k)
    num2 = (1 / 4) * (n - 1) * S2 * (n**2 + 3 * n - 6 - (n**2 - n + 2) * k)
    num3 = W**2 * (n**2 - 3 - (n - 1) ** 2 * k)
    var_C = (num1 - num2 + num3) / ((n - 1) * (n - 2) * (n - 3) * W**2)
    var_C = max(var_C, 1e-15)

    z_score = (C - expected_C) / np.sqrt(var_C)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

    return {
        "C": float(C),
        "expected_C": expected_C,
        "z_score": float(z_score),
        "p_value": float(p_value),
    }


def gearys_c(
    sj: s_spatioloji,
    feature_key: str = "leiden",
    graph_key: str = "knn_graph",
    output_key: str = "pt_gearys_c",
    force: bool = True,
) -> str:
    """Geary's C spatial autocorrelation.

    Same interface and categorical handling as :func:`morans_i`.

    Args:
        sj: Dataset instance.
        feature_key: Key for the feature in maps/.
        graph_key: Key for the graph edge list.
        output_key: Key to write Geary's C results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> gearys_c(sj, feature_key="leiden")
        'pt_gearys_c'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    adj, graph_cell_ids = _load_point_graph_sparse(sj, graph_key, weighted=False)
    id_to_idx = {cid: i for i, cid in enumerate(graph_cell_ids)}
    n = len(graph_cell_ids)

    edge_df = sj.maps[graph_key].compute()

    feat_df = sj.maps[feature_key].compute()
    value_cols = [c for c in feat_df.columns if c != "cell_id"]
    if len(value_cols) != 1:
        raise ValueError(f"Expected exactly one feature column besides cell_id, got {len(value_cols)}")

    col = value_cols[0]
    values_series = feat_df.set_index("cell_id")[col]
    aligned = values_series.reindex(graph_cell_ids)

    records = []
    if aligned.dtype == object or hasattr(aligned, "cat"):
        categories = sorted(aligned.dropna().unique(), key=str)
        for cat in categories:
            indicator = (aligned == cat).astype(float).values
            result = _compute_gearys_c_sparse(indicator, adj, edge_df, id_to_idx, n)
            result["feature"] = str(cat)
            records.append(result)
    else:
        vals = aligned.values.astype(float)
        result = _compute_gearys_c_sparse(vals, adj, edge_df, id_to_idx, n)
        result["feature"] = col
        records.append(result)

    df = pd.DataFrame(records)[["feature", "C", "expected_C", "z_score", "p_value"]]
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def clustering_coefficient(
    sj: s_spatioloji,
    graph_key: str = "knn_graph",
    output_key: str = "pt_clustering_coeff",
    force: bool = True,
) -> str:
    """Per-cell local clustering coefficient.

    Requires networkx. Not scalable beyond ~5M cells.

    Args:
        sj: Dataset instance.
        graph_key: Key for the graph edge list.
        output_key: Key to write clustering coefficients under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Raises:
        ValueError: If dataset has more than 5 million cells.

    Example:
        >>> clustering_coefficient(sj)
        'pt_clustering_coeff'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    n_cells = sj.cells.n_cells
    if n_cells > 5_000_000:
        raise ValueError(
            "clustering_coefficient requires networkx and is not scalable beyond ~5M cells. "
            "Consider using morans_i or getis_ord_gi for large datasets."
        )

    maps_dir.mkdir(exist_ok=True)
    G = _load_point_graph(sj, graph_key)

    cells_df = sj.cells.df.compute()
    all_cell_ids = list(cells_df["cell_id"])

    cc = nx.clustering(G)
    records = [{"cell_id": cid, "clustering_coeff": cc.get(cid, 0.0)} for cid in all_cell_ids]

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def getis_ord_gi(
    sj: s_spatioloji,
    feature_key: str,
    graph_key: str = "knn_graph",
    star: bool = True,
    output_key: str = "pt_getis_ord",
    force: bool = True,
) -> str:
    """Getis-Ord Gi* (or Gi) statistic per cell.

    Args:
        sj: Dataset instance.
        feature_key: Key for a numeric feature in maps/ (required, no default).
        graph_key: Key for the graph edge list.
        star: If ``True``, compute Gi* (includes self). If ``False``, Gi.
        output_key: Key to write results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Raises:
        ValueError: If feature is categorical.

    Example:
        >>> getis_ord_gi(sj, feature_key="leiden")
        'pt_getis_ord'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    adj, graph_cell_ids = _load_point_graph_sparse(sj, graph_key, weighted=False)

    feat_df = sj.maps[feature_key].compute()
    value_cols = [c for c in feat_df.columns if c != "cell_id"]
    if len(value_cols) != 1:
        raise ValueError(f"Expected exactly one feature column besides cell_id, got {len(value_cols)}")

    col = value_cols[0]
    values_series = feat_df.set_index("cell_id")[col]
    aligned = values_series.reindex(graph_cell_ids)

    if aligned.dtype == object or hasattr(aligned, "cat"):
        raise ValueError("Getis-Ord Gi* requires a numeric feature.")

    x = aligned.values.astype(np.float64)
    n = len(x)

    if star:
        adj_work = adj + scipy.sparse.eye(n, format="csr")
    else:
        adj_work = adj.copy()

    x_bar = x.mean()
    S = x.std()

    if S == 0:
        df = pd.DataFrame(
            {
                "cell_id": graph_cell_ids,
                "gi_stat": np.zeros(n),
                "p_value": np.ones(n),
            }
        )
        _atomic_write_parquet(df, maps_dir, output_key)
        return output_key

    ones = np.ones(n)
    Wi = np.asarray(adj_work @ ones).ravel()
    Wx = np.asarray(adj_work @ x).ravel()

    Wi_sq = Wi  # binary weights: sum w_ij^2 = sum w_ij

    numerator = Wx - x_bar * Wi
    denominator = S * np.sqrt((n * Wi_sq - Wi**2) / (n - 1))
    denominator[denominator == 0] = 1e-15

    gi_stat = numerator / denominator
    p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(gi_stat)))

    df = pd.DataFrame(
        {
            "cell_id": graph_cell_ids,
            "gi_stat": gi_stat,
            "p_value": p_value,
        }
    )
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
