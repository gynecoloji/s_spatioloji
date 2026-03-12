"""ExpressionStore: chunked, lazy access to the cell × gene expression matrix.

Backed by a Zarr array on disk, exposed as a dask.array for lazy, parallel
computation.  Raw counts are stored as uint16; normalized values as float32.
All write operations go through this class to ensure consistent chunk layout
and compression settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import zarr

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from s_spatioloji.data.config import ChunkConfig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_COMPRESSOR_MAP: dict[str, str] = {
    "zstd": "zstd",
    "lz4": "lz4",
    "blosclz": "blosclz",
}


def _blosc_compressor(compression: str) -> object:
    """Return a Blosc compressor configured for the given algorithm.

    Args:
        compression: One of ``"zstd"``, ``"lz4"``, ``"blosclz"``.

    Returns:
        A ``numcodecs.Blosc`` compressor instance.

    Raises:
        ValueError: If ``compression`` is not a supported algorithm name.
        ImportError: If ``numcodecs`` is not installed.
    """
    try:
        import numcodecs
    except ImportError:
        raise ImportError("Install numcodecs: pip install numcodecs")

    cname = _COMPRESSOR_MAP.get(compression)
    if cname is None:
        raise ValueError(f"Unsupported compression {compression!r}. Choose from {set(_COMPRESSOR_MAP)}")
    return numcodecs.Blosc(cname=cname, clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)


# ---------------------------------------------------------------------------
# ExpressionStore
# ---------------------------------------------------------------------------


class ExpressionStore:
    """Lazy, chunked access to a cell × gene expression matrix stored in Zarr.

    The underlying Zarr array is never fully loaded into memory.  Slicing
    returns a ``dask.array`` that can be composed with other Dask operations
    and materialised with ``.compute()`` only when needed.

    Typical usage::

        store = ExpressionStore.open(path)
        chunk = store[:1000, :].compute()   # first 1000 cells, all genes

    Args:
        path: Path to the Zarr store directory (``expression.zarr``).
        chunk_config: Chunk shape settings from ``SpatiolojiConfig``.

    Attributes:
        path: Resolved path to the Zarr store.
        n_cells: Number of cells (rows) in the matrix.
        n_genes: Number of genes (columns) in the matrix.
        dtype: NumPy dtype of the stored array.
        gene_names: 1-D array of gene name strings, or ``None`` if not set.
        cell_ids: 1-D array of cell ID strings, or ``None`` if not set.
    """

    _ARRAY_KEY = "X"
    _GENE_NAMES_KEY = "gene_names"
    _CELL_IDS_KEY = "cell_ids"

    def __init__(self, path: Path, chunk_config: ChunkConfig) -> None:
        self.path = path
        self._chunk_config = chunk_config
        self._store = zarr.open(str(path), mode="r+")
        self._array: zarr.Array = self._store[self._ARRAY_KEY]

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        path: Path,
        n_cells: int,
        n_genes: int,
        chunk_config: ChunkConfig,
        compression: str = "zstd",
        dtype: str | np.dtype = "float32",  # type: ignore[type-arg]
    ) -> ExpressionStore:
        """Create a new, empty expression Zarr store on disk.

        Initialises the array with zeros.  Use :meth:`write_chunk` or direct
        Zarr slice assignment to populate it.

        Args:
            path: Directory to create the Zarr store at.  Must not exist.
            n_cells: Total number of cells (row dimension).
            n_genes: Total number of genes (column dimension).
            chunk_config: Chunk shape settings.
            compression: Blosc algorithm — one of ``"zstd"``, ``"lz4"``,
                ``"blosclz"``.
            dtype: NumPy dtype for the array.  Use ``"uint16"`` for raw counts,
                ``"float32"`` for normalised values.

        Returns:
            An :class:`ExpressionStore` opened in read-write mode.

        Raises:
            FileExistsError: If ``path`` already exists.
            ValueError: If ``n_cells`` or ``n_genes`` is not positive.
        """
        if path.exists():
            raise FileExistsError(f"Zarr store already exists at {path}. Use ExpressionStore.open() instead.")
        if n_cells <= 0:
            raise ValueError(f"n_cells must be positive, got {n_cells}")
        if n_genes <= 0:
            raise ValueError(f"n_genes must be positive, got {n_genes}")

        compressor = _blosc_compressor(compression)

        genes_chunk = n_genes if chunk_config.expression_genes == -1 else chunk_config.expression_genes
        chunks = (chunk_config.expression_cells, genes_chunk)

        store = zarr.open(str(path), mode="w")
        store.create_dataset(
            cls._ARRAY_KEY,
            shape=(n_cells, n_genes),
            chunks=chunks,
            dtype=dtype,
            compressor=compressor,
            fill_value=0,
        )
        instance = cls.__new__(cls)
        instance.path = path
        instance._chunk_config = chunk_config
        instance._store = store
        instance._array = store[cls._ARRAY_KEY]
        return instance

    @classmethod
    def open(cls, path: Path, chunk_config: ChunkConfig) -> ExpressionStore:
        """Open an existing expression Zarr store in read-write mode.

        Args:
            path: Path to an existing ``expression.zarr`` directory.
            chunk_config: Chunk shape settings (used for any future writes).

        Returns:
            An :class:`ExpressionStore` wrapping the existing array.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"No expression store found at {path}")
        return cls(path=path, chunk_config=chunk_config)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """Number of cells (rows)."""
        return int(self._array.shape[0])

    @property
    def n_genes(self) -> int:
        """Number of genes (columns)."""
        return int(self._array.shape[1])

    @property
    def dtype(self) -> np.dtype:  # type: ignore[type-arg]
        """NumPy dtype of the stored array."""
        return np.dtype(self._array.dtype)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the array as ``(n_cells, n_genes)``."""
        return (self.n_cells, self.n_genes)

    @property
    def gene_names(self) -> NDArray[np.str_] | None:
        """Gene name labels, or ``None`` if not stored."""
        key = self._GENE_NAMES_KEY
        if key in self._store:
            return np.array(self._store[key])
        return None

    @gene_names.setter
    def gene_names(self, names: list[str] | NDArray[np.str_]) -> None:
        """Store gene name labels alongside the expression array.

        Args:
            names: Sequence of gene name strings.  Length must equal
                :attr:`n_genes`.

        Raises:
            ValueError: If ``len(names) != n_genes``.
        """
        names_arr = np.asarray(names, dtype=str)
        if len(names_arr) != self.n_genes:
            raise ValueError(f"Expected {self.n_genes} gene names, got {len(names_arr)}")
        self._store[self._GENE_NAMES_KEY] = names_arr

    @property
    def cell_ids(self) -> NDArray[np.str_] | None:
        """Cell ID labels, or ``None`` if not stored."""
        key = self._CELL_IDS_KEY
        if key in self._store:
            return np.array(self._store[key])
        return None

    @cell_ids.setter
    def cell_ids(self, ids: list[str] | NDArray[np.str_]) -> None:
        """Store cell ID labels alongside the expression array.

        Args:
            ids: Sequence of cell ID strings.  Length must equal
                :attr:`n_cells`.

        Raises:
            ValueError: If ``len(ids) != n_cells``.
        """
        ids_arr = np.asarray(ids, dtype=str)
        if len(ids_arr) != self.n_cells:
            raise ValueError(f"Expected {self.n_cells} cell IDs, got {len(ids_arr)}")
        self._store[self._CELL_IDS_KEY] = ids_arr

    # ------------------------------------------------------------------
    # Dask access
    # ------------------------------------------------------------------

    def to_dask(self) -> da.Array:
        """Return the full array as a lazy ``dask.array``.

        No data is read from disk until ``.compute()`` is called.

        Returns:
            A ``dask.array`` with the same shape and dtype as the Zarr array.

        Example:
            >>> arr = store.to_dask()
            >>> subset = arr[:500, :100].compute()
        """
        return da.from_zarr(self._array)

    def select_genes(self, indices: list[int] | NDArray[np.intp]) -> da.Array:
        """Return a lazy column subset of the expression matrix.

        Args:
            indices: Integer indices of the genes (columns) to select.

        Returns:
            A ``dask.array`` of shape ``(n_cells, len(indices))``.

        Raises:
            ValueError: If any index is out of bounds.

        Example:
            >>> hvg_idx = [0, 5, 10]
            >>> subset = store.select_genes(hvg_idx).compute()
        """
        idx = np.asarray(indices, dtype=np.intp)
        if idx.size > 0:
            if idx.min() < 0 or idx.max() >= self.n_genes:
                raise ValueError(f"Gene indices out of bounds for n_genes={self.n_genes}")
        return da.from_zarr(self._array)[:, idx]

    def select_cells(self, indices: list[int] | NDArray[np.intp]) -> da.Array:
        """Return a lazy row subset of the expression matrix.

        Args:
            indices: Integer indices of the cells (rows) to select.

        Returns:
            A ``dask.array`` of shape ``(len(indices), n_genes)``.

        Raises:
            ValueError: If any index is out of bounds.
        """
        idx = np.asarray(indices, dtype=np.intp)
        if idx.size > 0:
            if idx.min() < 0 or idx.max() >= self.n_cells:
                raise ValueError(f"Cell indices out of bounds for n_cells={self.n_cells}")
        return da.from_zarr(self._array)[idx, :]

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_chunk(self, cell_start: int, data: NDArray) -> None:  # type: ignore[type-arg]
        """Write a block of cells into the expression array.

        Args:
            cell_start: Row index at which to start writing.
            data: 2-D array of shape ``(n, n_genes)`` where ``n`` is the
                number of cells in this chunk.

        Raises:
            ValueError: If ``data.shape[1] != n_genes`` or if the chunk
                would exceed the array bounds.
        """
        if data.ndim != 2:
            raise ValueError(f"data must be 2-D, got shape {data.shape}")
        if data.shape[1] != self.n_genes:
            raise ValueError(f"data has {data.shape[1]} genes, expected {self.n_genes}")
        cell_end = cell_start + data.shape[0]
        if cell_end > self.n_cells:
            raise ValueError(
                f"Chunk [{cell_start}:{cell_end}] exceeds n_cells={self.n_cells}"
            )
        self._array[cell_start:cell_end, :] = data

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ExpressionStore("
            f"n_cells={self.n_cells}, "
            f"n_genes={self.n_genes}, "
            f"dtype={self.dtype}, "
            f"path={self.path})"
        )
