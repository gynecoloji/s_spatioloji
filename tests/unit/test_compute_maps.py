"""Unit tests for the Maps accessor on s_spatioloji."""

from __future__ import annotations

import dask.dataframe as dd
import pandas as pd
import pytest

from s_spatioloji.data.config import ChunkConfig
from s_spatioloji.data.expression import ExpressionStore


class TestMapsGetitem:
    def test_parquet_returns_dask_dataframe(self, sj):
        maps_dir = sj.config.root / "maps"
        maps_dir.mkdir()
        df = pd.DataFrame({"cell_id": ["c1", "c2"], "val": [1.0, 2.0]})
        df.to_parquet(str(maps_dir / "test_key.parquet"), engine="pyarrow", index=False)

        result = sj.maps["test_key"]
        assert isinstance(result, dd.DataFrame)
        computed = result.compute()
        assert list(computed.columns) == ["cell_id", "val"]
        assert len(computed) == 2

    def test_zarr_in_maps_returns_expression_store(self, sj):
        maps_dir = sj.config.root / "maps"
        maps_dir.mkdir()
        cfg = ChunkConfig()
        ExpressionStore.create(maps_dir / "test_zarr.zarr", n_cells=5, n_genes=3, chunk_config=cfg)

        result = sj.maps["test_zarr"]
        assert isinstance(result, ExpressionStore)
        assert result.shape == (5, 3)

    def test_zarr_at_root_returns_expression_store(self, sj):
        cfg = ChunkConfig()
        ExpressionStore.create(sj.config.root / "expression_scvi.zarr", n_cells=5, n_genes=3, chunk_config=cfg)

        result = sj.maps["expression_scvi"]
        assert isinstance(result, ExpressionStore)

    def test_missing_key_raises(self, sj):
        with pytest.raises(KeyError, match="nonexistent"):
            sj.maps["nonexistent"]

    def test_scvi_model_key_raises(self, sj):
        with pytest.raises(KeyError, match="internal directory"):
            sj.maps["_scvi_model"]

    def test_parquet_takes_precedence_over_root_zarr(self, sj):
        """If both maps/<key>.parquet and <root>/<key>.zarr exist, Parquet wins."""
        maps_dir = sj.config.root / "maps"
        maps_dir.mkdir()
        df = pd.DataFrame({"cell_id": ["c1"], "val": [1.0]})
        df.to_parquet(str(maps_dir / "expression_test.parquet"), engine="pyarrow", index=False)
        cfg = ChunkConfig()
        ExpressionStore.create(sj.config.root / "expression_test.zarr", n_cells=2, n_genes=2, chunk_config=cfg)

        result = sj.maps["expression_test"]
        assert isinstance(result, dd.DataFrame)


class TestMapsHas:
    def test_has_true(self, sj):
        maps_dir = sj.config.root / "maps"
        maps_dir.mkdir()
        df = pd.DataFrame({"cell_id": ["c1"], "val": [1.0]})
        df.to_parquet(str(maps_dir / "present.parquet"), engine="pyarrow", index=False)

        assert sj.maps.has("present") is True

    def test_has_false(self, sj):
        assert sj.maps.has("absent") is False


class TestMapsKeys:
    def test_keys_empty(self, sj):
        assert sj.maps.keys() == []

    def test_keys_includes_parquet(self, sj):
        maps_dir = sj.config.root / "maps"
        maps_dir.mkdir()
        pd.DataFrame({"cell_id": ["c1"]}).to_parquet(
            str(maps_dir / "X_pca.parquet"), engine="pyarrow", index=False
        )
        assert "X_pca" in sj.maps.keys()

    def test_keys_includes_root_zarr(self, sj):
        cfg = ChunkConfig()
        ExpressionStore.create(sj.config.root / "expression_magic.zarr", n_cells=2, n_genes=2, chunk_config=cfg)
        assert "expression_magic" in sj.maps.keys()

    def test_keys_deduplicates(self, sj):
        maps_dir = sj.config.root / "maps"
        maps_dir.mkdir()
        pd.DataFrame({"cell_id": ["c1"]}).to_parquet(
            str(maps_dir / "expression_scvi.parquet"), engine="pyarrow", index=False
        )
        cfg = ChunkConfig()
        ExpressionStore.create(sj.config.root / "expression_scvi.zarr", n_cells=2, n_genes=2, chunk_config=cfg)

        keys = sj.maps.keys()
        assert keys.count("expression_scvi") == 1

    def test_keys_excludes_scvi_model(self, sj):
        maps_dir = sj.config.root / "maps"
        maps_dir.mkdir()
        (maps_dir / "_scvi_model.zarr").mkdir()

        assert "_scvi_model" not in sj.maps.keys()


class TestMapsDelete:
    def test_delete_parquet(self, sj):
        maps_dir = sj.config.root / "maps"
        maps_dir.mkdir()
        path = maps_dir / "to_delete.parquet"
        pd.DataFrame({"cell_id": ["c1"]}).to_parquet(str(path), engine="pyarrow", index=False)

        sj.maps.delete("to_delete")
        assert not path.exists()

    def test_delete_root_zarr(self, sj):
        cfg = ChunkConfig()
        zarr_path = sj.config.root / "expression_scvi.zarr"
        ExpressionStore.create(zarr_path, n_cells=2, n_genes=2, chunk_config=cfg)

        sj.maps.delete("expression_scvi")
        assert not zarr_path.exists()

    def test_delete_missing_raises(self, sj):
        with pytest.raises(KeyError):
            sj.maps.delete("nonexistent")
