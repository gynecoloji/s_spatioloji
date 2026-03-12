"""Unit tests for s_spatioloji.data.io (from_xenium, from_merscope)."""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import tifffile
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
from shapely.geometry import box

from s_spatioloji.data.core import s_spatioloji
from s_spatioloji.data.io import (
    _link_or_copy,
    _read_10x_mtx,
    _read_xenium_boundaries,
    _read_xenium_cells,
    _rename_columns,
    _write_matrix_chunked,
    from_merscope,
    from_xenium,
)

# ---------------------------------------------------------------------------
# Shared dimensions
# ---------------------------------------------------------------------------
N_CELLS = 20
N_GENES = 10


# ---------------------------------------------------------------------------
# Mock Xenium directory builder
# ---------------------------------------------------------------------------


def _make_xenium_dir(
    root: Path,
    n_cells: int = N_CELLS,
    n_genes: int = N_GENES,
    with_boundaries: bool = True,
    boundary_format: str = "vertex",  # "vertex" | "wkb"
    with_image: bool = False,
    cells_format: str = "parquet",  # "parquet" | "csv"
) -> Path:
    """Create a minimal synthetic Xenium output directory."""
    root.mkdir(parents=True, exist_ok=True)

    # cells
    cells_df = pd.DataFrame({
        "cell_id": [f"cell_{i}" for i in range(n_cells)],
        "x_centroid": np.linspace(0, 1000, n_cells),
        "y_centroid": np.linspace(0, 1000, n_cells),
        "transcript_counts": np.random.randint(50, 500, n_cells),
        "cell_area": np.random.uniform(100, 500, n_cells),
    })
    if cells_format == "parquet":
        cells_df.to_parquet(str(root / "cells.parquet"), index=False)
    else:
        cells_df.to_csv(str(root / "cells.csv.gz"), index=False, compression="gzip")

    # cell_feature_matrix
    mtx_dir = root / "cell_feature_matrix"
    mtx_dir.mkdir()
    _write_10x_mtx(mtx_dir, n_cells, n_genes)

    # boundaries
    if with_boundaries:
        if boundary_format == "vertex":
            _write_xenium_vertex_boundaries(root / "cell_boundaries.parquet", n_cells)
        else:
            _write_xenium_wkb_boundaries(root / "cell_boundaries.parquet", n_cells)

    # morphology image
    if with_image:
        data = np.zeros((2, 64, 64), dtype=np.uint16)
        tifffile.imwrite(
            str(root / "morphology.ome.tif"), data,
            photometric="minisblack", metadata={"axes": "CYX"},
        )

    return root


def _write_10x_mtx(mtx_dir: Path, n_cells: int, n_genes: int) -> None:
    """Write barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz."""
    barcodes = [f"AACC{i:08d}" for i in range(n_cells)]
    genes = [f"GENE{i:04d}" for i in range(n_genes)]

    with gzip.open(str(mtx_dir / "barcodes.tsv.gz"), "wt") as f:
        f.write("\n".join(barcodes) + "\n")

    with gzip.open(str(mtx_dir / "features.tsv.gz"), "wt") as f:
        for g in genes:
            f.write(f"{g}\t{g}\tGene Expression\n")

    # sparse matrix: genes × cells (MTX convention)
    counts = np.random.randint(0, 100, (n_genes, n_cells), dtype=np.int32)
    sparse = csr_matrix(counts)
    buf = io.BytesIO()
    mmwrite(buf, sparse)
    with gzip.open(str(mtx_dir / "matrix.mtx.gz"), "wb") as f:
        f.write(buf.getvalue())


def _write_xenium_vertex_boundaries(path: Path, n_cells: int) -> None:
    """Write vertex-per-row boundary Parquet."""
    records = []
    for i in range(n_cells):
        x0, y0 = i * 50.0, i * 50.0
        for vx, vy in [(x0, y0), (x0+40, y0), (x0+40, y0+40), (x0, y0+40), (x0, y0)]:
            records.append({"cell_id": f"cell_{i}", "vertex_x": vx, "vertex_y": vy})
    pd.DataFrame(records).to_parquet(str(path), index=False)


def _write_xenium_wkb_boundaries(path: Path, n_cells: int) -> None:
    """Write WKB geometry boundary Parquet."""
    records = []
    for i in range(n_cells):
        poly = box(i * 50, i * 50, i * 50 + 40, i * 50 + 40)
        records.append({"cell_id": f"cell_{i}", "geometry": poly.wkb})
    pd.DataFrame(records).to_parquet(str(path), index=False)


# ---------------------------------------------------------------------------
# Mock MERSCOPE directory builder
# ---------------------------------------------------------------------------


def _make_merscope_dir(
    root: Path,
    n_cells: int = N_CELLS,
    n_genes: int = N_GENES,
    with_boundaries: bool = True,
    with_image: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)

    # cell_metadata.csv
    meta = pd.DataFrame({
        "EntityID": [i for i in range(n_cells)],
        "center_x": np.linspace(0, 1000, n_cells),
        "center_y": np.linspace(0, 1000, n_cells),
        "fov": [i % 4 for i in range(n_cells)],
        "volume": np.random.uniform(1000, 5000, n_cells),
    })
    meta.to_csv(str(root / "cell_metadata.csv"), index=False)

    # cell_by_gene.csv (index = cell_id, columns = genes)
    genes = [f"GENE{i:04d}" for i in range(n_genes)]
    counts = np.random.randint(0, 100, (n_cells, n_genes))
    expr_df = pd.DataFrame(counts, columns=genes)
    expr_df.index.name = "cell_id"
    expr_df.to_csv(str(root / "cell_by_gene.csv"))

    # cell_boundaries/
    if with_boundaries:
        bd_dir = root / "cell_boundaries"
        bd_dir.mkdir()
        records = []
        for i in range(n_cells):
            poly = box(i * 50, i * 50, i * 50 + 40, i * 50 + 40)
            records.append({"EntityID": i, "geometry": poly.wkb})
        pd.DataFrame(records).to_parquet(str(bd_dir / "boundaries.parquet"), index=False)

    # images/
    if with_image:
        img_dir = root / "images"
        img_dir.mkdir()
        data = np.zeros((64, 64), dtype=np.uint16)
        tifffile.imwrite(str(img_dir / "mosaic_DAPI_z0.tif"), data)

    return root


# ===========================================================================
# Tests — from_xenium
# ===========================================================================


class TestFromXenium:
    def test_basic_output_files(self, tmp_path: Path) -> None:
        src = _make_xenium_dir(tmp_path / "xenium")
        dst = tmp_path / "out"
        sj = from_xenium(src, dst)
        assert (dst / "cells.parquet").exists()
        assert (dst / "expression.zarr").exists()

    def test_returns_s_spatioloji(self, tmp_path: Path) -> None:
        src = _make_xenium_dir(tmp_path / "xenium")
        sj = from_xenium(src, tmp_path / "out")
        assert isinstance(sj, s_spatioloji)

    def test_n_cells(self, tmp_path: Path) -> None:
        src = _make_xenium_dir(tmp_path / "xenium", n_cells=N_CELLS)
        sj = from_xenium(src, tmp_path / "out")
        assert sj.n_cells == N_CELLS

    def test_n_genes(self, tmp_path: Path) -> None:
        src = _make_xenium_dir(tmp_path / "xenium", n_genes=N_GENES)
        sj = from_xenium(src, tmp_path / "out")
        assert sj.n_genes == N_GENES

    def test_canonical_cell_columns(self, tmp_path: Path) -> None:
        src = _make_xenium_dir(tmp_path / "xenium")
        sj = from_xenium(src, tmp_path / "out")
        assert "x" in sj.obs_columns
        assert "y" in sj.obs_columns
        assert "cell_id" in sj.obs_columns

    def test_gene_names_stored(self, tmp_path: Path) -> None:
        src = _make_xenium_dir(tmp_path / "xenium")
        sj = from_xenium(src, tmp_path / "out")
        names = sj.expression.gene_names
        assert names is not None
        assert len(names) == N_GENES

    def test_cell_ids_stored(self, tmp_path: Path) -> None:
        src = _make_xenium_dir(tmp_path / "xenium")
        sj = from_xenium(src, tmp_path / "out")
        ids = sj.expression.cell_ids
        assert ids is not None
        assert len(ids) == N_CELLS

    def test_boundaries_written(self, tmp_path: Path) -> None:
        src = _make_xenium_dir(tmp_path / "xenium", with_boundaries=True)
        sj = from_xenium(src, tmp_path / "out")
        assert sj.has_boundaries
        assert sj.boundaries.n_cells == N_CELLS

    def test_wkb_boundaries(self, tmp_path: Path) -> None:
        src = _make_xenium_dir(tmp_path / "xenium", with_boundaries=True, boundary_format="wkb")
        sj = from_xenium(src, tmp_path / "out")
        assert sj.has_boundaries

    def test_no_boundaries_ok(self, tmp_path: Path) -> None:
        src = _make_xenium_dir(tmp_path / "xenium", with_boundaries=False)
        sj = from_xenium(src, tmp_path / "out")
        assert not sj.has_boundaries

    def test_morphology_linked(self, tmp_path: Path) -> None:
        src = _make_xenium_dir(tmp_path / "xenium", with_image=True)
        sj = from_xenium(src, tmp_path / "out")
        assert sj.has_morphology

    def test_csv_cells_format(self, tmp_path: Path) -> None:
        src = _make_xenium_dir(tmp_path / "xenium", cells_format="csv")
        sj = from_xenium(src, tmp_path / "out")
        assert sj.n_cells == N_CELLS

    def test_missing_xenium_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="xenium_dir"):
            from_xenium(tmp_path / "nonexistent", tmp_path / "out")

    def test_missing_cells_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "xenium"
        src.mkdir()
        (src / "cell_feature_matrix").mkdir()
        _write_10x_mtx(src / "cell_feature_matrix", N_CELLS, N_GENES)
        with pytest.raises(FileNotFoundError, match="cells"):
            from_xenium(src, tmp_path / "out")

    def test_missing_matrix_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "xenium"
        src.mkdir()
        cells_df = pd.DataFrame({
            "cell_id": [f"c{i}" for i in range(5)],
            "x_centroid": [0.0] * 5,
            "y_centroid": [0.0] * 5,
        })
        cells_df.to_parquet(str(src / "cells.parquet"), index=False)
        with pytest.raises(FileNotFoundError, match="cell_feature_matrix"):
            from_xenium(src, tmp_path / "out")

    def test_expression_values_preserved(self, tmp_path: Path) -> None:
        src = _make_xenium_dir(tmp_path / "xenium")
        sj = from_xenium(src, tmp_path / "out")
        arr = sj.expression.to_dask().compute()
        assert arr.shape == (N_CELLS, N_GENES)
        assert arr.dtype == np.uint16


# ===========================================================================
# Tests — from_merscope
# ===========================================================================


class TestFromMerscope:
    def test_basic_output_files(self, tmp_path: Path) -> None:
        src = _make_merscope_dir(tmp_path / "merscope")
        from_merscope(src, tmp_path / "out")
        assert (tmp_path / "out" / "cells.parquet").exists()
        assert (tmp_path / "out" / "expression.zarr").exists()

    def test_returns_s_spatioloji(self, tmp_path: Path) -> None:
        src = _make_merscope_dir(tmp_path / "merscope")
        sj = from_merscope(src, tmp_path / "out")
        assert isinstance(sj, s_spatioloji)

    def test_n_cells(self, tmp_path: Path) -> None:
        src = _make_merscope_dir(tmp_path / "merscope", n_cells=N_CELLS)
        sj = from_merscope(src, tmp_path / "out")
        assert sj.n_cells == N_CELLS

    def test_n_genes(self, tmp_path: Path) -> None:
        src = _make_merscope_dir(tmp_path / "merscope", n_genes=N_GENES)
        sj = from_merscope(src, tmp_path / "out")
        assert sj.n_genes == N_GENES

    def test_canonical_columns(self, tmp_path: Path) -> None:
        src = _make_merscope_dir(tmp_path / "merscope")
        sj = from_merscope(src, tmp_path / "out")
        assert "x" in sj.obs_columns
        assert "y" in sj.obs_columns

    def test_boundaries_written(self, tmp_path: Path) -> None:
        src = _make_merscope_dir(tmp_path / "merscope", with_boundaries=True)
        sj = from_merscope(src, tmp_path / "out")
        assert sj.has_boundaries

    def test_no_boundaries_ok(self, tmp_path: Path) -> None:
        src = _make_merscope_dir(tmp_path / "merscope", with_boundaries=False)
        sj = from_merscope(src, tmp_path / "out")
        assert not sj.has_boundaries

    def test_morphology_linked(self, tmp_path: Path) -> None:
        src = _make_merscope_dir(tmp_path / "merscope", with_image=True)
        sj = from_merscope(src, tmp_path / "out")
        assert sj.has_morphology

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="merscope_dir"):
            from_merscope(tmp_path / "nonexistent", tmp_path / "out")

    def test_missing_metadata_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "merscope"
        src.mkdir()
        pd.DataFrame({"cell_id": ["a"], "GENE0000": [1]}).to_csv(
            str(src / "cell_by_gene.csv")
        )
        with pytest.raises(FileNotFoundError, match="cell_metadata"):
            from_merscope(src, tmp_path / "out")

    def test_missing_expression_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "merscope"
        src.mkdir()
        pd.DataFrame({"EntityID": [0], "center_x": [0.0], "center_y": [0.0]}).to_csv(
            str(src / "cell_metadata.csv"), index=False
        )
        with pytest.raises(FileNotFoundError, match="cell_by_gene"):
            from_merscope(src, tmp_path / "out")


# ===========================================================================
# Tests — internal helpers
# ===========================================================================


class TestRead10xMtx:
    def test_shape(self, tmp_path: Path) -> None:
        mtx_dir = tmp_path / "mtx"
        mtx_dir.mkdir()
        _write_10x_mtx(mtx_dir, N_CELLS, N_GENES)
        matrix, genes, cells = _read_10x_mtx(mtx_dir)
        assert matrix.shape == (N_CELLS, N_GENES)

    def test_dtype_uint16(self, tmp_path: Path) -> None:
        mtx_dir = tmp_path / "mtx"
        mtx_dir.mkdir()
        _write_10x_mtx(mtx_dir, N_CELLS, N_GENES)
        matrix, _, _ = _read_10x_mtx(mtx_dir)
        assert matrix.dtype == np.uint16

    def test_gene_names_length(self, tmp_path: Path) -> None:
        mtx_dir = tmp_path / "mtx"
        mtx_dir.mkdir()
        _write_10x_mtx(mtx_dir, N_CELLS, N_GENES)
        _, genes, _ = _read_10x_mtx(mtx_dir)
        assert len(genes) == N_GENES

    def test_cell_ids_length(self, tmp_path: Path) -> None:
        mtx_dir = tmp_path / "mtx"
        mtx_dir.mkdir()
        _write_10x_mtx(mtx_dir, N_CELLS, N_GENES)
        _, _, cells = _read_10x_mtx(mtx_dir)
        assert len(cells) == N_CELLS

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _read_10x_mtx(tmp_path / "missing")


class TestRenameColumns:
    def test_renames_matching(self) -> None:
        df = pd.DataFrame({"x_centroid": [1.0], "y_centroid": [2.0]})
        result = _rename_columns(df, {"x_centroid": "x", "y_centroid": "y"})
        assert "x" in result.columns
        assert "y" in result.columns

    def test_ignores_missing(self) -> None:
        df = pd.DataFrame({"a": [1]})
        result = _rename_columns(df, {"nonexistent": "b"})
        assert "a" in result.columns


class TestLinkOrCopy:
    def test_copy(self, tmp_path: Path) -> None:
        src = tmp_path / "src.tif"
        src.write_bytes(b"hello")
        dst = tmp_path / "dst.tif"
        _link_or_copy(src, dst, copy=True)
        assert dst.exists()
        assert dst.read_bytes() == b"hello"

    def test_symlink_or_copy(self, tmp_path: Path) -> None:
        src = tmp_path / "src.tif"
        src.write_bytes(b"world")
        dst = tmp_path / "dst.tif"
        _link_or_copy(src, dst, copy=False)
        assert dst.exists()
        assert dst.read_bytes() == b"world"
