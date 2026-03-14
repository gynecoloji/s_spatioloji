"""Ripley's spatial statistics (K, L, G, F) from cell centroids.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.

Uses ``scipy.spatial.cKDTree`` for O(N log N) distance queries.
No graph dependency -- purely coordinate-based.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from s_spatioloji.compute import _atomic_write_parquet

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def _get_coords_and_groups(
    sj: s_spatioloji,
    cluster_key: str | None,
) -> list[tuple[str, np.ndarray]]:
    """Extract coordinates, optionally grouped by cluster.

    Args:
        sj: Dataset instance.
        cluster_key: If set, group by this maps key. If ``None``, return all.

    Returns:
        List of ``(group_name, coords_array)`` tuples.
    """
    cells_df = sj.cells.df.compute()
    coords = cells_df[["x", "y"]].values.astype(np.float64)

    if cluster_key is None:
        return [("all", coords)]

    cluster_df = sj.maps[cluster_key].compute()
    merged = cells_df[["cell_id", "x", "y"]].merge(cluster_df, on="cell_id")
    col = [c for c in cluster_df.columns if c != "cell_id"][0]
    groups = []
    for label in sorted(merged[col].unique(), key=str):
        mask = merged[col] == label
        group_coords = merged.loc[mask, ["x", "y"]].values.astype(np.float64)
        groups.append((str(label), group_coords))
    return groups


def _auto_radii(coords: np.ndarray, n_radii: int, max_radius: float | None) -> np.ndarray:
    """Generate evenly spaced radii up to *max_radius*.

    Args:
        coords: Array of shape ``(N, 2)``.
        n_radii: Number of radii to generate.
        max_radius: Upper bound. Defaults to 1/4 of shorter bounding box side.

    Returns:
        1-D array of radii values.
    """
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    if max_radius is None:
        max_radius = min(x_range, y_range) / 4.0
    return np.linspace(max_radius / n_radii, max_radius, n_radii)


def _study_area(coords: np.ndarray) -> float:
    """Bounding box area.

    Args:
        coords: Array of shape ``(N, 2)``.

    Returns:
        Bounding box area (clamped to a small positive value).
    """
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    return float(max(x_range * y_range, 1e-10))


def ripley_k(
    sj: s_spatioloji,
    radii: np.ndarray | None = None,
    n_radii: int = 50,
    max_radius: float | None = None,
    cluster_key: str | None = None,
    output_key: str = "pt_ripley_k",
    force: bool = True,
) -> str:
    """Ripley's K function.

    ``K(r) = (A / (N*(N-1))) * sum_i sum_{j!=i} 1(d_ij <= r)``

    Args:
        sj: Dataset instance.
        radii: Explicit evaluation distances. If ``None``, auto-generates.
        n_radii: Number of radii to auto-generate.
        max_radius: Upper bound. Defaults to 1/4 of shorter bounding box side.
        cluster_key: If set, compute per cluster. If ``None``, global.
        output_key: Key to write results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> ripley_k(sj, n_radii=50)
        'pt_ripley_k'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    groups = _get_coords_and_groups(sj, cluster_key)

    records = []
    for group_name, coords in groups:
        n = len(coords)
        if n < 2:
            continue

        r_vals = radii if radii is not None else _auto_radii(coords, n_radii, max_radius)
        area = _study_area(coords)
        tree = cKDTree(coords)

        for r in r_vals:
            count = tree.count_neighbors(tree, r) - n  # subtract self-pairs
            k_val = area / (n * (n - 1)) * count
            k_theo = np.pi * r**2
            records.append(
                {
                    "cluster": group_name,
                    "r": float(r),
                    "K": float(k_val),
                    "K_theo": float(k_theo),
                }
            )

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def ripley_l(
    sj: s_spatioloji,
    radii: np.ndarray | None = None,
    n_radii: int = 50,
    max_radius: float | None = None,
    cluster_key: str | None = None,
    output_key: str = "pt_ripley_l",
    force: bool = True,
) -> str:
    """Ripley's L function (variance-stabilized K).

    ``L(r) = sqrt(K(r) / pi) - r``. Under CSR, ``L(r) = 0``.

    Args:
        sj: Dataset instance.
        radii: Explicit evaluation distances.
        n_radii: Number of radii to auto-generate.
        max_radius: Upper bound.
        cluster_key: If set, compute per cluster.
        output_key: Key to write results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> ripley_l(sj)
        'pt_ripley_l'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    groups = _get_coords_and_groups(sj, cluster_key)

    records = []
    for group_name, coords in groups:
        n = len(coords)
        if n < 2:
            continue

        r_vals = radii if radii is not None else _auto_radii(coords, n_radii, max_radius)
        area = _study_area(coords)
        tree = cKDTree(coords)

        for r in r_vals:
            count = tree.count_neighbors(tree, r) - n
            k_val = area / (n * (n - 1)) * count
            l_val = np.sqrt(max(k_val, 0) / np.pi) - r
            records.append({"cluster": group_name, "r": float(r), "L": float(l_val)})

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def ripley_g(
    sj: s_spatioloji,
    radii: np.ndarray | None = None,
    n_radii: int = 50,
    max_radius: float | None = None,
    cluster_key: str | None = None,
    output_key: str = "pt_ripley_g",
    force: bool = True,
) -> str:
    """Ripley's G function (nearest-neighbor distance distribution).

    ``G(r) = fraction of points whose NN distance <= r``.

    Args:
        sj: Dataset instance.
        radii: Explicit evaluation distances.
        n_radii: Number of radii to auto-generate.
        max_radius: Upper bound.
        cluster_key: If set, compute per cluster.
        output_key: Key to write results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> ripley_g(sj)
        'pt_ripley_g'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    groups = _get_coords_and_groups(sj, cluster_key)

    records = []
    for group_name, coords in groups:
        n = len(coords)
        if n < 2:
            continue

        r_vals = radii if radii is not None else _auto_radii(coords, n_radii, max_radius)
        area = _study_area(coords)
        lam = n / area

        tree = cKDTree(coords)
        nn_dists, _ = tree.query(coords, k=2)
        nn_dists = nn_dists[:, 1]

        for r in r_vals:
            g_val = float(np.mean(nn_dists <= r))
            g_theo = 1.0 - np.exp(-lam * np.pi * r**2)
            records.append(
                {
                    "cluster": group_name,
                    "r": float(r),
                    "G": g_val,
                    "G_theo": float(g_theo),
                }
            )

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def ripley_f(
    sj: s_spatioloji,
    radii: np.ndarray | None = None,
    n_radii: int = 50,
    max_radius: float | None = None,
    cluster_key: str | None = None,
    n_random: int = 1000,
    output_key: str = "pt_ripley_f",
    force: bool = True,
) -> str:
    """Ripley's F function (empty-space distance distribution).

    ``F(r) = fraction of random reference points whose nearest data point <= r``.

    Args:
        sj: Dataset instance.
        radii: Explicit evaluation distances.
        n_radii: Number of radii to auto-generate.
        max_radius: Upper bound.
        cluster_key: If set, compute per cluster.
        n_random: Number of random reference points.
        output_key: Key to write results under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> ripley_f(sj)
        'pt_ripley_f'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    groups = _get_coords_and_groups(sj, cluster_key)

    rng = np.random.default_rng(42)
    records = []

    for group_name, coords in groups:
        n = len(coords)
        if n < 2:
            continue

        r_vals = radii if radii is not None else _auto_radii(coords, n_radii, max_radius)
        area = _study_area(coords)
        lam = n / area

        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)

        random_pts = np.column_stack(
            [
                rng.uniform(x_min, x_max, n_random),
                rng.uniform(y_min, y_max, n_random),
            ]
        )

        tree = cKDTree(coords)
        nn_dists, _ = tree.query(random_pts, k=1)

        for r in r_vals:
            f_val = float(np.mean(nn_dists <= r))
            f_theo = 1.0 - np.exp(-lam * np.pi * r**2)
            records.append(
                {
                    "cluster": group_name,
                    "r": float(r),
                    "F": f_val,
                    "F_theo": float(f_theo),
                }
            )

    df = pd.DataFrame(records)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
