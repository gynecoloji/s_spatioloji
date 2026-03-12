"""Unit tests for s_spatioloji.data.cells.CellStore."""

from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import pytest

from s_spatioloji.data.cells import CellStore, REQUIRED_COLUMNS

N = 100  # number of test cells


def _make_df(n: int = N, extra_cols: dict | None = None) -> pd.DataFrame:
    """Build a minimal valid cell DataFrame."""
    df = pd.DataFrame({
        "cell_id": [f"cell_{i}" for i in range(n)],
        "x": [float(i) * 10 for i in range(n)],
        "y": [float(i) * 5 for i in range(n)],
        "fov_id": [i % 4 for i in range(n)],
    })
    if extra_cols:
        for col, vals in extra_cols.items():
            df[col] = vals
    return df


@pytest.fixture()
def store(tmp_path: Path) -> CellStore:
    return CellStore.create(tmp_path / "cells.parquet", _make_df())


class TestCreate:
    def test_creates_file(self, tmp_path: Path) -> None:
        p = tmp_path / "cells.parquet"
        CellStore.create(p, _make_df())
        assert p.exists()

    def test_n_cells(self, store: CellStore) -> None:
        assert store.n_cells == N

    def test_columns_present(self, store: CellStore) -> None:
        for col in REQUIRED_COLUMNS:
            assert col in store.columns

    def test_raises_if_exists(self, tmp_path: Path) -> None:
        p = tmp_path / "cells.parquet"
        CellStore.create(p, _make_df())
        with pytest.raises(FileExistsError):
            CellStore.create(p, _make_df())

    def test_raises_missing_required_column(self, tmp_path: Path) -> None:
        df = _make_df().drop(columns=["x"])
        with pytest.raises(ValueError, match="missing required columns"):
            CellStore.create(tmp_path / "cells.parquet", df)

    def test_raises_duplicate_cell_id(self, tmp_path: Path) -> None:
        df = _make_df()
        df.loc[1, "cell_id"] = df.loc[0, "cell_id"]  # introduce duplicate
        with pytest.raises(ValueError, match="duplicate"):
            CellStore.create(tmp_path / "cells.parquet", df)


class TestOpen:
    def test_open_existing(self, store: CellStore, tmp_path: Path) -> None:
        reopened = CellStore.open(tmp_path / "cells.parquet")
        assert reopened.n_cells == N

    def test_open_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            CellStore.open(tmp_path / "missing.parquet")


class TestMetadata:
    def test_columns(self, store: CellStore) -> None:
        assert "cell_id" in store.columns
        assert "x" in store.columns
        assert "y" in store.columns

    def test_dtypes_returns_series(self, store: CellStore) -> None:
        assert isinstance(store.dtypes, pd.Series)

    def test_has_column_true(self, store: CellStore) -> None:
        assert store.has_column("x") is True

    def test_has_column_false(self, store: CellStore) -> None:
        assert store.has_column("nonexistent") is False


class TestSelectColumns:
    def test_select_subset(self, store: CellStore) -> None:
        result = store.select_columns(["cell_id", "x"]).compute()
        assert list(result.columns) == ["cell_id", "x"]
        assert len(result) == N

    def test_returns_dask_dataframe(self, store: CellStore) -> None:
        result = store.select_columns(["x", "y"])
        assert isinstance(result, dd.DataFrame)

    def test_missing_column_raises(self, store: CellStore) -> None:
        with pytest.raises(ValueError, match="Columns not found"):
            store.select_columns(["x", "nonexistent"])


class TestFilter:
    def test_filter_fov(self, store: CellStore) -> None:
        result = store.filter(fov_id=0).compute()
        assert (result["fov_id"] == 0).all()
        assert len(result) == N // 4

    def test_filter_returns_dask(self, store: CellStore) -> None:
        result = store.filter(fov_id=1)
        assert isinstance(result, dd.DataFrame)

    def test_filter_missing_column_raises(self, store: CellStore) -> None:
        with pytest.raises(ValueError, match="Filter columns not found"):
            store.filter(nonexistent=42)

    def test_filter_no_matches(self, store: CellStore) -> None:
        result = store.filter(fov_id=999).compute()
        assert len(result) == 0


class TestWithinBbox:
    def test_bbox_subset(self, store: CellStore) -> None:
        # x = i*10, y = i*5 for i in 0..99
        # x in [0, 50] → i in 0..5 (6 cells)
        result = store.within_bbox(0.0, 0.0, 50.0, 25.0).compute()
        assert len(result) == 6
        assert (result["x"] <= 50.0).all()
        assert (result["y"] <= 25.0).all()

    def test_bbox_returns_dask(self, store: CellStore) -> None:
        result = store.within_bbox(0, 0, 100, 100)
        assert isinstance(result, dd.DataFrame)

    def test_bbox_invalid_x(self, store: CellStore) -> None:
        with pytest.raises(ValueError, match="x_min"):
            store.within_bbox(100.0, 0.0, 50.0, 100.0)

    def test_bbox_invalid_y(self, store: CellStore) -> None:
        with pytest.raises(ValueError, match="y_min"):
            store.within_bbox(0.0, 100.0, 100.0, 50.0)


class TestSave:
    def test_save_roundtrip(self, store: CellStore, tmp_path: Path) -> None:
        out = tmp_path / "cells_out.parquet"
        store.save(out)
        reopened = CellStore.open(out)
        assert reopened.n_cells == N

    def test_save_inplace(self, store: CellStore, tmp_path: Path) -> None:
        store.save()
        reopened = CellStore.open(tmp_path / "cells.parquet")
        assert reopened.n_cells == N


class TestRepr:
    def test_repr(self, store: CellStore) -> None:
        r = repr(store)
        assert "CellStore" in r
        assert str(N) in r
