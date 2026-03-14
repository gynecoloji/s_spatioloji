"""Unit tests for s_spatioloji.data.core (s_spatioloji class)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import tifffile
from shapely.geometry import box

from s_spatioloji.data.boundaries import BoundaryStore
from s_spatioloji.data.cells import CellStore
from s_spatioloji.data.config import SpatiolojiConfig
from s_spatioloji.data.core import TileView, s_spatioloji
from s_spatioloji.data.expression import ExpressionStore
from s_spatioloji.data.images import ImageCollection

# -------------------------------------------------------------------------
# Dataset dimensions
# -------------------------------------------------------------------------
N_CELLS = 100
N_GENES = 20
GRID = 10  # 10x10 cell grid, each cell 100µm apart


# -------------------------------------------------------------------------
# Fixtures — build a complete minimal dataset on disk
# -------------------------------------------------------------------------


def _make_cells_df(n: int = N_CELLS) -> pd.DataFrame:
    """100 cells on a regular 10×10 grid, 100µm spacing."""
    side = int(n ** 0.5)
    records = []
    for i in range(side):
        for j in range(side):
            records.append({
                "cell_id": f"cell_{i}_{j}",
                "x": float(i * 100),
                "y": float(j * 100),
                "fov_id": i % 3,
            })
    return pd.DataFrame(records)


def _make_boundaries_gdf(n: int = N_CELLS) -> gpd.GeoDataFrame:
    side = int(n ** 0.5)
    records = []
    for i in range(side):
        for j in range(side):
            records.append({
                "cell_id": f"cell_{i}_{j}",
                "geometry": box(i * 100, j * 100, i * 100 + 80, j * 100 + 80),
            })
    return gpd.GeoDataFrame(records, geometry="geometry")


def _write_ome_tiff(path: Path) -> None:
    data = np.zeros((2, 256, 256), dtype=np.uint16)
    tifffile.imwrite(str(path), data, photometric="minisblack", metadata={"axes": "CYX"})


@pytest.fixture()
def dataset_path(tmp_path: Path) -> Path:
    """Build a complete minimal dataset: expression + cells + boundaries + image."""
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "_index").mkdir()

    cfg = SpatiolojiConfig(root=root)

    # expression
    ExpressionStore.create(
        cfg.paths.expression,
        n_cells=N_CELLS,
        n_genes=N_GENES,
        chunk_config=cfg.chunks,
    )

    # cells
    CellStore.create(cfg.paths.cells, _make_cells_df())

    # boundaries
    BoundaryStore.create(cfg.paths.boundaries, _make_boundaries_gdf())

    # morphology image
    _write_ome_tiff(cfg.paths.morphology)

    return root


@pytest.fixture()
def minimal_path(tmp_path: Path) -> Path:
    """Dataset with only expression + cells (no boundaries, no image)."""
    root = tmp_path / "minimal"
    root.mkdir()
    cfg = SpatiolojiConfig(root=root)
    ExpressionStore.create(cfg.paths.expression, N_CELLS, N_GENES, cfg.chunks)
    CellStore.create(cfg.paths.cells, _make_cells_df())
    return root


@pytest.fixture()
def sj(dataset_path: Path) -> s_spatioloji:
    return s_spatioloji.open(dataset_path)


@pytest.fixture()
def sj_minimal(minimal_path: Path) -> s_spatioloji:
    return s_spatioloji.open(minimal_path)


# -------------------------------------------------------------------------
# Tests — open / create
# -------------------------------------------------------------------------


class TestOpen:
    def test_open_valid(self, dataset_path: Path) -> None:
        sj = s_spatioloji.open(dataset_path)
        assert sj is not None

    def test_open_missing_root(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Dataset directory not found"):
            s_spatioloji.open(tmp_path / "nonexistent")

    def test_open_missing_cells(self, tmp_path: Path) -> None:
        root = tmp_path / "bad"
        root.mkdir()
        cfg = SpatiolojiConfig(root=root)
        ExpressionStore.create(cfg.paths.expression, 10, 5, cfg.chunks)
        # no cells.parquet
        with pytest.raises(FileNotFoundError, match="Required dataset files"):
            s_spatioloji.open(root)

    def test_open_missing_expression(self, tmp_path: Path) -> None:
        root = tmp_path / "bad"
        root.mkdir()
        cfg = SpatiolojiConfig(root=root)
        CellStore.create(cfg.paths.cells, _make_cells_df(4))
        # no expression.zarr
        with pytest.raises(FileNotFoundError, match="Required dataset files"):
            s_spatioloji.open(root)

    def test_open_accepts_str(self, dataset_path: Path) -> None:
        sj = s_spatioloji.open(str(dataset_path))
        assert sj is not None


class TestCreate:
    def test_create_new(self, tmp_path: Path) -> None:
        sj = s_spatioloji.create(tmp_path / "new_ds")
        assert (tmp_path / "new_ds").exists()
        assert (tmp_path / "new_ds" / "_index").exists()

    def test_create_existing_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "exists"
        p.mkdir()
        with pytest.raises(FileExistsError):
            s_spatioloji.create(p)


# -------------------------------------------------------------------------
# Tests — lazy backend properties
# -------------------------------------------------------------------------


class TestBackendProperties:
    def test_cells_type(self, sj: s_spatioloji) -> None:
        assert isinstance(sj.cells, CellStore)

    def test_expression_type(self, sj: s_spatioloji) -> None:
        assert isinstance(sj.expression, ExpressionStore)

    def test_boundaries_type(self, sj: s_spatioloji) -> None:
        from s_spatioloji.data.boundaries import BoundaryStore
        assert isinstance(sj.boundaries, BoundaryStore)

    def test_morphology_type(self, sj: s_spatioloji) -> None:
        from s_spatioloji.data.images import MorphologyImageStore
        assert isinstance(sj.morphology, MorphologyImageStore)

    def test_cells_cached(self, sj: s_spatioloji) -> None:
        c1 = sj.cells
        c2 = sj.cells
        assert c1 is c2

    def test_expression_cached(self, sj: s_spatioloji) -> None:
        e1 = sj.expression
        e2 = sj.expression
        assert e1 is e2

    def test_boundaries_missing_raises(self, sj_minimal: s_spatioloji) -> None:
        with pytest.raises(FileNotFoundError, match="boundaries.parquet"):
            _ = sj_minimal.boundaries

    def test_morphology_missing_raises(self, sj_minimal: s_spatioloji) -> None:
        with pytest.raises(FileNotFoundError, match="morphology.ome.tif"):
            _ = sj_minimal.morphology


# -------------------------------------------------------------------------
# Tests — availability flags
# -------------------------------------------------------------------------


class TestAvailabilityFlags:
    def test_has_boundaries_true(self, sj: s_spatioloji) -> None:
        assert sj.has_boundaries is True

    def test_has_boundaries_false(self, sj_minimal: s_spatioloji) -> None:
        assert sj_minimal.has_boundaries is False

    def test_has_morphology_true(self, sj: s_spatioloji) -> None:
        assert sj.has_morphology is True

    def test_has_morphology_false(self, sj_minimal: s_spatioloji) -> None:
        assert sj_minimal.has_morphology is False


# -------------------------------------------------------------------------
# Tests — dataset metadata
# -------------------------------------------------------------------------


class TestMetadata:
    def test_n_cells(self, sj: s_spatioloji) -> None:
        assert sj.n_cells == N_CELLS

    def test_n_genes(self, sj: s_spatioloji) -> None:
        assert sj.n_genes == N_GENES

    def test_obs_columns(self, sj: s_spatioloji) -> None:
        assert "cell_id" in sj.obs_columns
        assert "x" in sj.obs_columns
        assert "y" in sj.obs_columns


# -------------------------------------------------------------------------
# Tests — iter_tiles
# -------------------------------------------------------------------------


class TestIterTiles:
    def test_returns_tile_views(self, sj: s_spatioloji) -> None:
        tiles = list(sj.iter_tiles(tile_size=500.0))
        assert all(isinstance(t, TileView) for t in tiles)

    def test_tile_ids_sequential(self, sj: s_spatioloji) -> None:
        tiles = list(sj.iter_tiles(tile_size=500.0))
        ids = [t.tile_id for t in tiles]
        assert ids == list(range(len(tiles)))

    def test_tile_covers_all_cells(self, sj: s_spatioloji) -> None:
        # large tile_size → single tile containing all cells
        tiles = list(sj.iter_tiles(tile_size=10_000.0))
        assert len(tiles) == 1
        df = tiles[0].cells.compute()
        assert len(df) == N_CELLS

    def test_small_tiles_partition_dataset(self, sj: s_spatioloji) -> None:
        # tile_size smaller than the grid → multiple tiles
        tiles = list(sj.iter_tiles(tile_size=300.0, overlap=0.0))
        assert len(tiles) > 1

    def test_tile_boundaries_available(self, sj: s_spatioloji) -> None:
        tiles = list(sj.iter_tiles(tile_size=10_000.0))
        b = tiles[0].boundaries
        assert b is not None

    def test_tile_boundaries_none_when_missing(self, sj_minimal: s_spatioloji) -> None:
        tiles = list(sj_minimal.iter_tiles(tile_size=10_000.0))
        assert tiles[0].boundaries is None

    def test_tile_repr(self, sj: s_spatioloji) -> None:
        tile = next(sj.iter_tiles(tile_size=500.0))
        r = repr(tile)
        assert "TileView" in r


# -------------------------------------------------------------------------
# Tests — iter_cells
# -------------------------------------------------------------------------


class TestIterCells:
    def test_batches_cover_all_cells(self, sj: s_spatioloji) -> None:
        total = sum(len(batch.df.compute()) for batch in sj.iter_cells(batch_size=30))
        assert total == N_CELLS

    def test_batch_size_respected(self, sj: s_spatioloji) -> None:
        batches = list(sj.iter_cells(batch_size=30))
        # all but possibly the last batch should have ≤ 30 rows
        for b in batches[:-1]:
            assert len(b.df.compute()) <= 30

    def test_invalid_batch_size(self, sj: s_spatioloji) -> None:
        with pytest.raises(ValueError, match="batch_size must be positive"):
            list(sj.iter_cells(batch_size=0))


# -------------------------------------------------------------------------
# Tests — repr
# -------------------------------------------------------------------------


class TestRepr:
    def test_repr_contains_class_name(self, sj: s_spatioloji) -> None:
        assert "s_spatioloji" in repr(sj)

    def test_repr_contains_n_cells(self, sj: s_spatioloji) -> None:
        assert str(N_CELLS) in repr(sj)

    def test_repr_minimal(self, sj_minimal: s_spatioloji) -> None:
        r = repr(sj_minimal)
        assert "boundaries=no" in r
        assert "morphology=no" in r


# -------------------------------------------------------------------------
# Tests — images integration
# -------------------------------------------------------------------------


@pytest.fixture()
def sj_with_images(dataset_path: Path) -> s_spatioloji:
    """Dataset with images/ directory and images_meta.json."""
    import json

    img_dir = dataset_path / "images"
    img_dir.mkdir()
    data = np.zeros((1, 64, 64), dtype=np.uint16)
    tifffile.imwrite(
        str(img_dir / "morphology_focus_0000.ome.tif"),
        data,
        photometric="minisblack",
        metadata={"axes": "CYX"},
    )
    meta = {
        "pixel_size": 0.2125,
        "default_image": "morphology_focus_0000",
        "files": {"morphology_focus_0000": "morphology_focus_0000.ome.tif"},
        "xenium_version": "3.0.0.15",
    }
    (dataset_path / "images_meta.json").write_text(json.dumps(meta))
    return s_spatioloji.open(dataset_path)


class TestImages:
    def test_images_type(self, sj_with_images: s_spatioloji) -> None:
        assert isinstance(sj_with_images.images, ImageCollection)

    def test_images_keys(self, sj_with_images: s_spatioloji) -> None:
        assert "morphology_focus_0000" in sj_with_images.images.keys()

    def test_has_images_true(self, sj_with_images: s_spatioloji) -> None:
        assert sj_with_images.has_images

    def test_has_images_false(self, sj_minimal: s_spatioloji) -> None:
        assert not sj_minimal.has_images

    def test_morphology_returns_default(self, sj_with_images: s_spatioloji) -> None:
        store = sj_with_images.morphology
        assert store is not None

    def test_images_pixel_size(self, sj_with_images: s_spatioloji) -> None:
        assert sj_with_images.images.pixel_size == 0.2125
