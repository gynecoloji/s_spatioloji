"""Unit tests for s_spatioloji.data.expression.ExpressionStore."""

from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from s_spatioloji.data.config import ChunkConfig
from s_spatioloji.data.expression import ExpressionStore

N_CELLS = 200
N_GENES = 50
CHUNK_CFG = ChunkConfig(expression_cells=64, expression_genes=-1)


@pytest.fixture()
def store(tmp_path: Path) -> ExpressionStore:
    """Fresh empty expression store."""
    return ExpressionStore.create(
        path=tmp_path / "expression.zarr",
        n_cells=N_CELLS,
        n_genes=N_GENES,
        chunk_config=CHUNK_CFG,
    )


@pytest.fixture()
def store_with_data(store: ExpressionStore) -> ExpressionStore:
    """Store pre-populated with incrementing float32 values."""
    data = np.arange(N_CELLS * N_GENES, dtype=np.float32).reshape(N_CELLS, N_GENES)
    store.write_chunk(0, data)
    return store


class TestCreate:
    def test_shape(self, store: ExpressionStore) -> None:
        assert store.shape == (N_CELLS, N_GENES)
        assert store.n_cells == N_CELLS
        assert store.n_genes == N_GENES

    def test_dtype_default(self, store: ExpressionStore) -> None:
        assert store.dtype == np.float32

    def test_uint16_dtype(self, tmp_path: Path) -> None:
        s = ExpressionStore.create(
            tmp_path / "raw.zarr", n_cells=10, n_genes=5,
            chunk_config=CHUNK_CFG, dtype="uint16",
        )
        assert s.dtype == np.uint16

    def test_raises_if_path_exists(self, tmp_path: Path) -> None:
        p = tmp_path / "expression.zarr"
        ExpressionStore.create(p, n_cells=10, n_genes=5, chunk_config=CHUNK_CFG)
        with pytest.raises(FileExistsError):
            ExpressionStore.create(p, n_cells=10, n_genes=5, chunk_config=CHUNK_CFG)

    def test_raises_nonpositive_cells(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="n_cells must be positive"):
            ExpressionStore.create(tmp_path / "x.zarr", n_cells=0, n_genes=5, chunk_config=CHUNK_CFG)

    def test_raises_nonpositive_genes(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="n_genes must be positive"):
            ExpressionStore.create(tmp_path / "x.zarr", n_cells=5, n_genes=0, chunk_config=CHUNK_CFG)


class TestOpen:
    def test_open_existing(self, store: ExpressionStore, tmp_path: Path) -> None:
        reopened = ExpressionStore.open(tmp_path / "expression.zarr", CHUNK_CFG)
        assert reopened.shape == (N_CELLS, N_GENES)

    def test_open_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ExpressionStore.open(tmp_path / "missing.zarr", CHUNK_CFG)


class TestMetadata:
    def test_gene_names_roundtrip(self, store: ExpressionStore) -> None:
        names = [f"gene_{i}" for i in range(N_GENES)]
        store.gene_names = names
        assert list(store.gene_names) == names  # type: ignore[arg-type]

    def test_gene_names_wrong_length(self, store: ExpressionStore) -> None:
        with pytest.raises(ValueError, match="gene names"):
            store.gene_names = ["a", "b"]

    def test_cell_ids_roundtrip(self, store: ExpressionStore) -> None:
        ids = [f"cell_{i}" for i in range(N_CELLS)]
        store.cell_ids = ids
        assert list(store.cell_ids) == ids  # type: ignore[arg-type]

    def test_cell_ids_wrong_length(self, store: ExpressionStore) -> None:
        with pytest.raises(ValueError, match="cell IDs"):
            store.cell_ids = ["x"]

    def test_gene_names_none_when_not_set(self, store: ExpressionStore) -> None:
        assert store.gene_names is None

    def test_cell_ids_none_when_not_set(self, store: ExpressionStore) -> None:
        assert store.cell_ids is None


class TestDaskAccess:
    def test_to_dask_type(self, store: ExpressionStore) -> None:
        arr = store.to_dask()
        assert isinstance(arr, da.Array)

    def test_to_dask_shape(self, store: ExpressionStore) -> None:
        arr = store.to_dask()
        assert arr.shape == (N_CELLS, N_GENES)

    def test_to_dask_lazy(self, store: ExpressionStore) -> None:
        # Should not raise and should not load data
        arr = store.to_dask()
        assert arr is not None

    def test_select_genes(self, store_with_data: ExpressionStore) -> None:
        idx = [0, 10, 20]
        arr = store_with_data.select_genes(idx).compute()
        assert arr.shape == (N_CELLS, 3)
        full = store_with_data.to_dask().compute()
        np.testing.assert_array_equal(arr, full[:, idx])

    def test_select_genes_out_of_bounds(self, store: ExpressionStore) -> None:
        with pytest.raises(ValueError, match="out of bounds"):
            store.select_genes([N_GENES])

    def test_select_cells(self, store_with_data: ExpressionStore) -> None:
        idx = [0, 50, 100]
        arr = store_with_data.select_cells(idx).compute()
        assert arr.shape == (3, N_GENES)
        full = store_with_data.to_dask().compute()
        np.testing.assert_array_equal(arr, full[idx, :])

    def test_select_cells_out_of_bounds(self, store: ExpressionStore) -> None:
        with pytest.raises(ValueError, match="out of bounds"):
            store.select_cells([N_CELLS])


class TestWriteChunk:
    def test_write_and_read(self, store: ExpressionStore) -> None:
        data = np.ones((50, N_GENES), dtype=np.float32)
        store.write_chunk(0, data)
        result = store.to_dask()[:50, :].compute()
        np.testing.assert_array_equal(result, data)

    def test_write_middle_chunk(self, store: ExpressionStore) -> None:
        data = np.full((30, N_GENES), 7.0, dtype=np.float32)
        store.write_chunk(100, data)
        result = store.to_dask()[100:130, :].compute()
        np.testing.assert_array_equal(result, data)

    def test_write_wrong_gene_count(self, store: ExpressionStore) -> None:
        with pytest.raises(ValueError, match="genes"):
            store.write_chunk(0, np.ones((10, N_GENES + 1), dtype=np.float32))

    def test_write_exceeds_bounds(self, store: ExpressionStore) -> None:
        with pytest.raises(ValueError, match="exceeds n_cells"):
            store.write_chunk(N_CELLS - 5, np.ones((10, N_GENES), dtype=np.float32))

    def test_write_non_2d(self, store: ExpressionStore) -> None:
        with pytest.raises(ValueError, match="2-D"):
            store.write_chunk(0, np.ones((10,), dtype=np.float32))


class TestRepr:
    def test_repr(self, store: ExpressionStore) -> None:
        r = repr(store)
        assert "ExpressionStore" in r
        assert str(N_CELLS) in r
        assert str(N_GENES) in r
