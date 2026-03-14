"""Statistical testing functions for point-based spatial data.

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
from scipy.spatial import cKDTree

from s_spatioloji.compute import _atomic_write_parquet

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def permutation_test(
    sj: s_spatioloji,
    cluster_key: str = "leiden",
    graph_key: str = "knn_graph",
    n_permutations: int = 1000,
    random_state: int = 42,
    output_key: str = "pt_permutation_test",
    force: bool = True,
) -> str:
    """Permutation test for spatial colocalization significance.

    Permutes cluster labels, recomputes colocalization observed/expected
    ratios, and compares against the null distribution.

    Args:
        sj: Dataset instance.
        cluster_key: Key for cluster labels in maps/.
        graph_key: Key for the graph edge list.
        n_permutations: Number of label permutations.
        random_state: Seed for reproducibility.
        output_key: Key to write test results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> permutation_test(sj, n_permutations=1000)
        'pt_permutation_test'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)

    edge_df = sj.maps[graph_key].compute()
    cluster_df = sj.maps[cluster_key].compute()
    cell_ids = list(cluster_df["cell_id"])
    labels = list(cluster_df[cluster_key])
    cell_to_cluster = dict(zip(cell_ids, labels, strict=True))

    unique_labels = sorted(set(labels), key=str)
    pairs = list(combinations_with_replacement(unique_labels, 2))
    pair_keys = [tuple(sorted([str(a), str(b)])) for a, b in pairs]

    edge_pairs = []
    for _, row in edge_df.iterrows():
        a, b = row["cell_id_a"], row["cell_id_b"]
        if a in cell_to_cluster and b in cell_to_cluster:
            edge_pairs.append((a, b))

    def _compute_ratios(c2c):
        counts = defaultdict(int)
        cluster_sizes = defaultdict(int)
        for lab in c2c.values():
            cluster_sizes[str(lab)] += 1
        n_total = sum(cluster_sizes.values())
        total_edges = len(edge_pairs)

        for u, v in edge_pairs:
            la, lb = c2c[u], c2c[v]
            pair = tuple(sorted([str(la), str(lb)]))
            counts[pair] += 1

        ratios = {}
        for pk in pair_keys:
            obs = counts.get(pk, 0)
            a_lab, b_lab = pk
            n_a = cluster_sizes.get(a_lab, 0)
            n_b = cluster_sizes.get(b_lab, 0)
            if a_lab == b_lab:
                exp = n_a * (n_a - 1) * total_edges / max(n_total * (n_total - 1), 1)
            else:
                exp = 2 * n_a * n_b * total_edges / max(n_total * (n_total - 1), 1)
            ratios[pk] = obs / exp if exp > 0 else 0.0
        return ratios

    observed_ratios = _compute_ratios(cell_to_cluster)

    rng = np.random.default_rng(random_state)
    perm_ratios = {pk: [] for pk in pair_keys}

    for _ in range(n_permutations):
        shuffled = rng.permutation(labels)
        perm_c2c = dict(zip(cell_ids, shuffled, strict=True))
        ratios = _compute_ratios(perm_c2c)
        for pk in pair_keys:
            perm_ratios[pk].append(ratios.get(pk, 0.0))

    records = []
    for pk in pair_keys:
        obs_ratio = observed_ratios.get(pk, 0.0)
        null_dist = np.array(perm_ratios[pk])
        null_mean = null_dist.mean()
        null_std = null_dist.std()

        z = (obs_ratio - null_mean) / null_std if null_std > 0 else 0.0
        p = (np.sum(null_dist >= obs_ratio) + 1) / (n_permutations + 1)

        records.append(
            {
                "cluster_a": pk[0],
                "cluster_b": pk[1],
                "observed_ratio": float(obs_ratio),
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
    output_key: str = "pt_quadrat_density",
    force: bool = True,
) -> str:
    """Quadrat-based density analysis with chi-squared test.

    Divides spatial extent into ``n_bins x n_bins`` grid quadrats.
    Does not depend on graph.

    Args:
        sj: Dataset instance.
        cluster_key: Key for cluster labels in maps/.
        n_bins: Number of bins per dimension.
        output_key: Key to write density results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> quadrat_density(sj, n_bins=10)
        'pt_quadrat_density'
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

    x_edges = np.linspace(x.min() - 1e-6, x.max() + 1e-6, n_bins + 1)
    y_edges = np.linspace(y.min() - 1e-6, y.max() + 1e-6, n_bins + 1)

    x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, n_bins - 1)
    y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, n_bins - 1)

    unique_labels = sorted(set(labels), key=str)
    records = []

    for label in unique_labels:
        mask = labels == label
        counts = np.zeros((n_bins, n_bins))
        for xi, yi in zip(x_bin[mask], y_bin[mask], strict=True):
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


def clark_evans(
    sj: s_spatioloji,
    cluster_key: str | None = None,
    output_key: str = "pt_clark_evans",
    force: bool = True,
) -> str:
    """Clark-Evans nearest-neighbor index.

    ``R = r_obs / r_exp`` where ``R < 1`` = clustered, ``R = 1`` = random,
    ``R > 1`` = dispersed.

    Args:
        sj: Dataset instance.
        cluster_key: If set, compute per cluster. If ``None``, global.
        output_key: Key to write results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> clark_evans(sj)
        'pt_clark_evans'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    cells_df = sj.cells.df.compute()

    if cluster_key is None:
        groups = [("all", cells_df)]
    else:
        cluster_df = sj.maps[cluster_key].compute()
        merged = cells_df[["cell_id", "x", "y"]].merge(cluster_df, on="cell_id")
        col = [c for c in cluster_df.columns if c != "cell_id"][0]
        groups = [(str(label), merged[merged[col] == label]) for label in sorted(merged[col].unique(), key=str)]

    records = []
    for group_name, group_df in groups:
        coords = group_df[["x", "y"]].values.astype(np.float64)
        N = len(coords)
        if N < 2:
            records.append(
                {
                    "cluster": group_name,
                    "R": 0.0,
                    "r_observed": 0.0,
                    "r_expected": 0.0,
                    "z_score": 0.0,
                    "p_value": 1.0,
                }
            )
            continue

        x_range = coords[:, 0].max() - coords[:, 0].min()
        y_range = coords[:, 1].max() - coords[:, 1].min()
        A = max(x_range * y_range, 1e-10)
        lam = N / A

        tree = cKDTree(coords)
        nn_dists, _ = tree.query(coords, k=2)
        r_obs = float(nn_dists[:, 1].mean())

        r_exp = 1.0 / (2.0 * np.sqrt(lam))
        sigma_r = 0.26136 / np.sqrt(N * lam)

        R = r_obs / r_exp if r_exp > 0 else 0.0
        z = (r_obs - r_exp) / sigma_r if sigma_r > 0 else 0.0
        p = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

        records.append(
            {
                "cluster": group_name,
                "R": float(R),
                "r_observed": r_obs,
                "r_expected": float(r_exp),
                "z_score": float(z),
                "p_value": float(p),
            }
        )

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def dclf_envelope(
    sj: s_spatioloji,
    function: str = "K",
    cluster_key: str | None = None,
    n_simulations: int = 199,
    radii: np.ndarray | None = None,
    n_radii: int = 50,
    max_radius: float | None = None,
    max_cells: int = 100_000,
    random_state: int = 42,
    output_key: str = "pt_dclf_envelope",
    force: bool = True,
) -> str:
    """DCLF global envelope test against CSR.

    Generates CSR point patterns and builds pointwise min/max envelope.

    Args:
        sj: Dataset instance.
        function: One of ``"K"``, ``"L"``, ``"G"``, ``"F"``.
        cluster_key: If set, compute per cluster.
        n_simulations: Number of CSR simulations.
        radii: Explicit evaluation distances.
        n_radii: Number of radii to auto-generate.
        max_radius: Upper bound for radii.
        max_cells: Subsampling threshold.
        random_state: Seed for reproducibility.
        output_key: Key to write results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> dclf_envelope(sj, function="K", n_simulations=199)
        'pt_dclf_envelope'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(random_state)

    from s_spatioloji.spatial.point.ripley import _auto_radii, _get_coords_and_groups, _study_area

    groups = _get_coords_and_groups(sj, cluster_key)

    records = []
    for group_name, coords in groups:
        N = len(coords)
        if N < 2:
            continue

        # Subsample if needed
        if N > max_cells:
            idx = rng.choice(N, max_cells, replace=False)
            coords = coords[idx]
            N = max_cells

        r_vals = radii if radii is not None else _auto_radii(coords, n_radii, max_radius)
        A = _study_area(coords)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        def _compute_function(pts, r_vals, rng_local):
            """Compute the chosen Ripley function for a point pattern."""
            tree = cKDTree(pts)
            n = len(pts)
            area = max(
                (pts[:, 0].max() - pts[:, 0].min()) * (pts[:, 1].max() - pts[:, 1].min()),
                1e-10,
            )

            result = np.zeros(len(r_vals))
            if function == "K":
                for i, r in enumerate(r_vals):
                    count = tree.count_neighbors(tree, r) - n
                    result[i] = area / (n * (n - 1)) * count
            elif function == "L":
                for i, r in enumerate(r_vals):
                    count = tree.count_neighbors(tree, r) - n
                    K = area / (n * (n - 1)) * count
                    result[i] = np.sqrt(max(K, 0) / np.pi) - r
            elif function == "G":
                nn_dists, _ = tree.query(pts, k=2)
                nn_dists = nn_dists[:, 1]
                for i, r in enumerate(r_vals):
                    result[i] = np.mean(nn_dists <= r)
            elif function == "F":
                random_pts = np.column_stack(
                    [
                        rng_local.uniform(pts[:, 0].min(), pts[:, 0].max(), min(1000, n)),
                        rng_local.uniform(pts[:, 1].min(), pts[:, 1].max(), min(1000, n)),
                    ]
                )
                nn_dists, _ = tree.query(random_pts, k=1)
                for i, r in enumerate(r_vals):
                    result[i] = np.mean(nn_dists <= r)
            else:
                raise ValueError(f"Unknown function: {function!r}. Must be one of 'K', 'L', 'G', 'F'.")
            return result

        observed = _compute_function(coords, r_vals, rng)

        lam = N / A
        if function == "K":
            theo = np.pi * r_vals**2
        elif function == "L":
            theo = np.zeros(len(r_vals))
        elif function in ("G", "F"):
            theo = 1.0 - np.exp(-lam * np.pi * r_vals**2)
        else:
            raise ValueError(f"Unknown function: {function!r}. Must be one of 'K', 'L', 'G', 'F'.")

        sim_results = np.zeros((n_simulations, len(r_vals)))
        for s in range(n_simulations):
            sim_pts = np.column_stack(
                [
                    rng.uniform(x_min, x_max, N),
                    rng.uniform(y_min, y_max, N),
                ]
            )
            sim_results[s] = _compute_function(sim_pts, r_vals, rng)

        lo = sim_results.min(axis=0)
        hi = sim_results.max(axis=0)

        obs_max_dev = np.max(np.abs(observed - theo))
        sim_max_devs = np.max(np.abs(sim_results - theo[np.newaxis, :]), axis=1)
        p_value = float((np.sum(sim_max_devs >= obs_max_dev) + 1) / (n_simulations + 1))

        for i, r in enumerate(r_vals):
            records.append(
                {
                    "cluster": group_name,
                    "r": float(r),
                    "observed": float(observed[i]),
                    "lo": float(lo[i]),
                    "hi": float(hi[i]),
                    "theo": float(theo[i]),
                    "p_value": p_value,
                }
            )

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
