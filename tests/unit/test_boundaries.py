"""Unit tests for s_spatioloji.data.boundaries.BoundaryStore."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon, box

from s_spatioloji.data.boundaries import BoundaryStore, _add_bbox_columns

N = 25  # 5x5 grid of cells


def _make_grid_gdf(n_side: int = 5, cell_size: float = 100.0) -> gpd.GeoDataFrame:
    """Build a regular grid of square polygons.

    Cell (i, j) occupies [i*cell_size, (i+1)*cell_size] x [j*cell_size, (j+1)*cell_size].
    """
    records = []
    for i in range(n_side):
        for j in range(n_side):
            x0, y0 = i * cell_size, j * cell_size
            records.append({
                "cell_id": f"cell_{i}_{j}",
                "geometry": box(x0, y0, x0 + cell_size, y0 + cell_size),
            })
    return gpd.GeoDataFrame(records, geometry="geometry")


@pytest.fixture()
def gdf() -> gpd.GeoDataFrame:
    return _make_grid_gdf()


@pytest.fixture()
def store(tmp_path: Path, gdf: gpd.GeoDataFrame) -> BoundaryStore:
    return BoundaryStore.create(tmp_path / "boundaries.parquet", gdf)


class TestCreate:
    def test_creates_file(self, tmp_path: Path, gdf: gpd.GeoDataFrame) -> None:
        p = tmp_path / "boundaries.parquet"
        BoundaryStore.create(p, gdf)
        assert p.exists()

    def test_bbox_columns_written(self, store: BoundaryStore) -> None:
        loaded = store.load()
        for col in ("x_min", "y_min", "x_max", "y_max"):
            assert col in loaded.columns

    def test_n_cells(self, store: BoundaryStore) -> None:
        assert store.n_cells == N

    def test_raises_if_exists(self, tmp_path: Path, gdf: gpd.GeoDataFrame) -> None:
        p = tmp_path / "boundaries.parquet"
        BoundaryStore.create(p, gdf)
        with pytest.raises(FileExistsError):
            BoundaryStore.create(p, gdf)

    def test_raises_missing_cell_id(self, tmp_path: Path, gdf: gpd.GeoDataFrame) -> None:
        bad = gdf.drop(columns=["cell_id"])
        with pytest.raises(ValueError, match="missing required columns"):
            BoundaryStore.create(tmp_path / "b.parquet", bad)

    def test_raises_duplicate_cell_id(self, tmp_path: Path, gdf: gpd.GeoDataFrame) -> None:
        bad = gdf.copy()
        bad.loc[1, "cell_id"] = bad.loc[0, "cell_id"]
        with pytest.raises(ValueError, match="duplicate"):
            BoundaryStore.create(tmp_path / "b.parquet", bad)

    def test_raises_not_geodataframe(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="GeoDataFrame"):
            BoundaryStore.create(tmp_path / "b.parquet", pd.DataFrame({"cell_id": ["a"]}))  # type: ignore[arg-type]


class TestOpen:
    def test_open_existing(self, store: BoundaryStore, tmp_path: Path) -> None:
        reopened = BoundaryStore.open(tmp_path / "boundaries.parquet")
        assert reopened.n_cells == N

    def test_open_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            BoundaryStore.open(tmp_path / "missing.parquet")


class TestLoad:
    def test_returns_geodataframe(self, store: BoundaryStore) -> None:
        assert isinstance(store.load(), gpd.GeoDataFrame)

    def test_cached(self, store: BoundaryStore) -> None:
        gdf1 = store.load()
        gdf2 = store.load()
        assert gdf1 is gdf2  # same object — cached


class TestMetadata:
    def test_columns(self, store: BoundaryStore) -> None:
        assert "cell_id" in store.columns
        assert "geometry" in store.columns

    def test_crs_none_when_not_set(self, store: BoundaryStore) -> None:
        assert store.crs is None


class TestQueryBbox:
    def test_full_extent(self, store: BoundaryStore) -> None:
        # grid spans 0-500 in both axes
        result = store.query_bbox(0, 0, 500, 500)
        assert len(result) == N

    def test_single_cell(self, store: BoundaryStore) -> None:
        # cell (0,0) occupies [0,100]x[0,100]
        result = store.query_bbox(0, 0, 100, 100)
        assert len(result) >= 1

    def test_empty_region(self, store: BoundaryStore) -> None:
        result = store.query_bbox(9000, 9000, 9500, 9500)
        assert len(result) == 0

    def test_returns_geodataframe(self, store: BoundaryStore) -> None:
        result = store.query_bbox(0, 0, 200, 200)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_invalid_x(self, store: BoundaryStore) -> None:
        with pytest.raises(ValueError, match="x_min"):
            store.query_bbox(200, 0, 100, 100)

    def test_invalid_y(self, store: BoundaryStore) -> None:
        with pytest.raises(ValueError, match="y_min"):
            store.query_bbox(0, 200, 100, 100)


class TestQueryPolygon:
    def test_intersects_all(self, store: BoundaryStore) -> None:
        big_poly = box(0, 0, 500, 500)
        result = store.query_polygon(big_poly, predicate="intersects")
        assert len(result) == N

    def test_intersects_corner(self, store: BoundaryStore) -> None:
        # small box in top-right corner: only cells near (400,400)
        corner = box(350, 350, 500, 500)
        result = store.query_polygon(corner, predicate="intersects")
        assert len(result) >= 1
        assert len(result) < N

    def test_contains_interior_cells(self, store: BoundaryStore) -> None:
        # predicate="contains": query polygon contains the cell boundary
        # interior 3x3 cells (i,j in 1..3) span [100,400]x[100,400]
        # container [50,450]x[50,450] fully contains exactly those 9 cells
        container = box(50, 50, 450, 450)
        result = store.query_polygon(container, predicate="contains")
        assert len(result) == 9

    def test_within_reverses_direction(self, store: BoundaryStore) -> None:
        # predicate="within": query polygon is within each cell boundary
        # no cell (100x100) is large enough to contain the full container
        container = box(50, 50, 450, 450)
        result = store.query_polygon(container, predicate="within")
        assert len(result) == 0  # expected: container is not within any single cell

    def test_invalid_predicate(self, store: BoundaryStore) -> None:
        with pytest.raises(ValueError, match="predicate must be one of"):
            store.query_polygon(box(0, 0, 100, 100), predicate="touches")


class TestQueryRadius:
    def test_centre_cell(self, store: BoundaryStore) -> None:
        # centre of cell (2,2) = (250, 250), radius large enough to hit it
        result = store.query_radius(250, 250, radius=1.0)
        assert len(result) >= 1

    def test_large_radius_returns_all(self, store: BoundaryStore) -> None:
        result = store.query_radius(250, 250, radius=5000.0)
        assert len(result) == N

    def test_invalid_radius(self, store: BoundaryStore) -> None:
        with pytest.raises(ValueError, match="radius must be positive"):
            store.query_radius(250, 250, radius=0)


class TestCellById:
    def test_found(self, store: BoundaryStore) -> None:
        result = store.cell_by_id("cell_0_0")
        assert len(result) == 1
        assert result.iloc[0]["cell_id"] == "cell_0_0"

    def test_not_found(self, store: BoundaryStore) -> None:
        with pytest.raises(KeyError, match="not found"):
            store.cell_by_id("nonexistent_cell")


class TestAddBboxColumns:
    def test_adds_all_bbox_cols(self, gdf: gpd.GeoDataFrame) -> None:
        result = _add_bbox_columns(gdf)
        for col in ("x_min", "y_min", "x_max", "y_max"):
            assert col in result.columns

    def test_bbox_values_correct(self, gdf: gpd.GeoDataFrame) -> None:
        result = _add_bbox_columns(gdf)
        # cell_0_0 occupies [0,100]x[0,100]
        row = result[result["cell_id"] == "cell_0_0"].iloc[0]
        assert row["x_min"] == pytest.approx(0.0)
        assert row["y_min"] == pytest.approx(0.0)
        assert row["x_max"] == pytest.approx(100.0)
        assert row["y_max"] == pytest.approx(100.0)


class TestRepr:
    def test_repr(self, store: BoundaryStore) -> None:
        r = repr(store)
        assert "BoundaryStore" in r
        assert str(N) in r
