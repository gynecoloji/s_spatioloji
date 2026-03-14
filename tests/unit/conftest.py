"""Shared fixtures for compute layer unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from s_spatioloji.data.cells import CellStore
from s_spatioloji.data.config import SpatiolojiConfig
from s_spatioloji.data.core import s_spatioloji
from s_spatioloji.data.expression import ExpressionStore

N_CELLS = 200
N_GENES = 30


@pytest.fixture()
def sj(tmp_path: Path) -> s_spatioloji:
    """Minimal dataset: 200 cells x 30 genes, uint16 raw counts.

    Provides:
        - expression.zarr with random integer counts [0, 100)
        - cells.parquet with cell_id, x, y, fov_id (3 FOVs)
        - gene names: gene_0 .. gene_29
    """
    root = tmp_path / "dataset"
    root.mkdir()
    cfg = SpatiolojiConfig(root=root)

    # Expression store with raw counts
    rng = np.random.default_rng(42)
    data = rng.integers(0, 100, (N_CELLS, N_GENES), dtype=np.uint16)
    store = ExpressionStore.create(
        cfg.paths.expression, N_CELLS, N_GENES, cfg.chunks, dtype="uint16"
    )
    store.write_chunk(0, data)
    store.gene_names = [f"gene_{i}" for i in range(N_GENES)]

    # Cell metadata
    records = [
        {
            "cell_id": f"cell_{i}",
            "x": float(i % 20) * 50.0,
            "y": float(i // 20) * 50.0,
            "fov_id": i % 3,
        }
        for i in range(N_CELLS)
    ]
    CellStore.create(cfg.paths.cells, pd.DataFrame(records))

    return s_spatioloji.open(root)


@pytest.fixture()
def sj_with_boundaries(sj):
    """sj with synthetic Voronoi polygon boundaries."""
    import geopandas as gpd
    from scipy.spatial import Voronoi
    from shapely.geometry import Polygon, box
    from shapely.ops import clip_by_rect

    from s_spatioloji.data.boundaries import BoundaryStore

    cells_df = sj.cells.df.compute()
    coords = cells_df[["x", "y"]].values

    x_min, y_min = coords.min(axis=0) - 100
    x_max, y_max = coords.max(axis=0) + 100
    mirror_pts = np.concatenate([
        coords,
        np.column_stack([2 * x_min - coords[:, 0], coords[:, 1]]),
        np.column_stack([2 * x_max - coords[:, 0], coords[:, 1]]),
        np.column_stack([coords[:, 0], 2 * y_min - coords[:, 1]]),
        np.column_stack([coords[:, 0], 2 * y_max - coords[:, 1]]),
    ])

    vor = Voronoi(mirror_pts)
    clip_box = (x_min, y_min, x_max, y_max)

    geometries = []
    for i in range(len(coords)):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            cx, cy = coords[i]
            geometries.append(box(cx - 5, cy - 5, cx + 5, cy + 5))
        else:
            verts = [vor.vertices[v] for v in region]
            poly = Polygon(verts)
            poly = clip_by_rect(poly, *clip_box)
            if poly.is_empty or not poly.is_valid:
                cx, cy = coords[i]
                geometries.append(box(cx - 5, cy - 5, cx + 5, cy + 5))
            else:
                geometries.append(poly)

    gdf = gpd.GeoDataFrame({
        "cell_id": cells_df["cell_id"].values,
        "geometry": geometries,
    })
    BoundaryStore.create(sj.config.paths.boundaries, gdf)
    return sj


@pytest.fixture()
def sj_with_graph(sj_with_boundaries):
    """sj_with_boundaries + pre-built contact graph."""
    from s_spatioloji.spatial.polygon.graph import build_contact_graph

    build_contact_graph(sj_with_boundaries)
    return sj_with_boundaries


@pytest.fixture()
def sj_with_clusters(sj_with_graph):
    """sj_with_graph + synthetic cluster labels in maps/leiden.parquet."""
    cells_df = sj_with_graph.cells.df.compute()
    x = cells_df["x"].values
    y = cells_df["y"].values
    median_x = np.median(x)
    median_y = np.median(y)
    labels = np.where(
        x < median_x,
        np.where(y < median_y, 0, 1),
        np.where(y < median_y, 2, 3),
    )
    df = pd.DataFrame({"cell_id": cells_df["cell_id"].values, "leiden": labels})
    from s_spatioloji.compute import _atomic_write_parquet

    maps_dir = sj_with_graph.config.root / "maps"
    maps_dir.mkdir(exist_ok=True)
    _atomic_write_parquet(df, maps_dir, "leiden")
    return sj_with_graph


@pytest.fixture()
def sj_with_knn_graph(sj):
    """sj + pre-built KNN graph (k=6)."""
    from s_spatioloji.spatial.point.graph import build_knn_graph

    build_knn_graph(sj, k=6)
    return sj


@pytest.fixture()
def sj_with_pt_clusters(sj_with_knn_graph):
    """sj_with_knn_graph + synthetic cluster labels in maps/leiden.parquet."""
    cells_df = sj_with_knn_graph.cells.df.compute()
    x = cells_df["x"].values
    y = cells_df["y"].values
    median_x = np.median(x)
    median_y = np.median(y)
    labels = np.where(
        x < median_x,
        np.where(y < median_y, 0, 1),
        np.where(y < median_y, 2, 3),
    )
    df = pd.DataFrame({"cell_id": cells_df["cell_id"].values, "leiden": labels})
    from s_spatioloji.compute import _atomic_write_parquet

    maps_dir = sj_with_knn_graph.config.root / "maps"
    maps_dir.mkdir(exist_ok=True)
    _atomic_write_parquet(df, maps_dir, "leiden")
    return sj_with_knn_graph
