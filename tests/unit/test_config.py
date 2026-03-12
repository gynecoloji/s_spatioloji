"""Unit tests for s_spatioloji.data.config."""

from pathlib import Path

import pytest

from s_spatioloji.data.config import ChunkConfig, SpatiolojiConfig, StorePaths, TileConfig


class TestTileConfig:
    def test_defaults(self) -> None:
        t = TileConfig()
        assert t.tile_size == 512.0
        assert t.overlap == 50.0

    def test_custom_values(self) -> None:
        t = TileConfig(tile_size=256.0, overlap=25.0)
        assert t.tile_size == 256.0
        assert t.overlap == 25.0


class TestChunkConfig:
    def test_defaults(self) -> None:
        c = ChunkConfig()
        assert c.expression_cells == 2048
        assert c.expression_genes == -1
        assert c.image_y == 512
        assert c.image_x == 512

    def test_custom_values(self) -> None:
        c = ChunkConfig(expression_cells=1024, expression_genes=500)
        assert c.expression_cells == 1024
        assert c.expression_genes == 500


class TestStorePaths:
    def test_all_paths_derived_from_root(self, tmp_path: Path) -> None:
        p = StorePaths(root=tmp_path)
        assert p.expression == tmp_path / "expression.zarr"
        assert p.cells == tmp_path / "cells.parquet"
        assert p.transcripts == tmp_path / "transcripts"
        assert p.boundaries == tmp_path / "boundaries.parquet"
        assert p.morphology == tmp_path / "morphology.ome.tif"
        assert p.index == tmp_path / "_index"
        assert p.spatial_index == tmp_path / "_index" / "spatial.rtree"
        assert p.knn_graph == tmp_path / "_index" / "knn.npz"
        assert p.ann_index == tmp_path / "_index" / "ann.index"


class TestSpatiolojiConfig:
    def test_defaults(self, tmp_path: Path) -> None:
        cfg = SpatiolojiConfig(root=tmp_path)
        assert cfg.tile.tile_size == 512.0
        assert cfg.chunks.expression_cells == 2048
        assert cfg.compression == "zstd"
        assert cfg.n_workers is None

    def test_paths_property(self, tmp_path: Path) -> None:
        cfg = SpatiolojiConfig(root=tmp_path)
        assert cfg.paths.expression == tmp_path / "expression.zarr"

    def test_invalid_tile_size(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="tile_size must be positive"):
            SpatiolojiConfig(root=tmp_path, tile=TileConfig(tile_size=0.0))

    def test_negative_overlap(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="overlap must be >= 0"):
            SpatiolojiConfig(root=tmp_path, tile=TileConfig(overlap=-1.0))

    def test_overlap_exceeds_tile_size(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="overlap.*must be less than tile_size"):
            SpatiolojiConfig(root=tmp_path, tile=TileConfig(tile_size=100.0, overlap=100.0))

    def test_invalid_expression_cells_chunk(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="expression_cells chunk must be positive"):
            SpatiolojiConfig(root=tmp_path, chunks=ChunkConfig(expression_cells=0))

    def test_invalid_compression(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="compression must be one of"):
            SpatiolojiConfig(root=tmp_path, compression="snappy")

    def test_valid_compression_options(self, tmp_path: Path) -> None:
        for comp in ("zstd", "lz4", "blosclz"):
            cfg = SpatiolojiConfig(root=tmp_path, compression=comp)
            assert cfg.compression == comp
