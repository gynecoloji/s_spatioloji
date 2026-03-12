"""CellStore: lazy, columnar access to per-cell metadata stored in Parquet.

Cell metadata (centroid coordinates, QC metrics, cluster labels, embeddings,
etc.) lives in a single Parquet file.  All access is lazy via
``dask.dataframe`` — nothing is loaded until ``.compute()`` is called.
Predicate pushdown to PyArrow means spatial or categorical filters touch only
the relevant row groups, never the full file.

Required columns that must be present in every valid CellStore:
    - ``cell_id``  (str)   — unique cell identifier
    - ``x``        (float) — centroid x-coordinate in micrometers
    - ``y``        (float) — centroid y-coordinate in micrometers
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask.dataframe as dd
import pandas as pd

if TYPE_CHECKING:
    from s_spatioloji.data.config import SpatiolojiConfig

# Columns that every CellStore must contain.
REQUIRED_COLUMNS: tuple[str, ...] = ("cell_id", "x", "y")

# Canonical dtypes for required columns.
REQUIRED_DTYPES: dict[str, str] = {
    "cell_id": "object",
    "x": "float64",
    "y": "float64",
}


# ---------------------------------------------------------------------------
# CellStore
# ---------------------------------------------------------------------------


class CellStore:
    """Lazy dask.dataframe wrapper around the cell metadata Parquet file.

    Every row is one cell.  The store is append-friendly: new columns (e.g.
    cluster labels, UMAP coordinates) are added via :meth:`add_column` and
    written back to the Parquet file with :meth:`save`.

    The underlying ``dask.dataframe`` is exposed directly as :attr:`df` for
    power users who need custom Dask operations.

    Args:
        path: Path to the ``cells.parquet`` file.

    Attributes:
        path: Resolved path to the Parquet file.
        df: Lazy ``dask.dataframe.DataFrame`` over the Parquet file.

    Example:
        >>> store = CellStore.open(path)
        >>> store.n_cells
        50000000
        >>> centroids = store.select_columns(["cell_id", "x", "y"]).compute()
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.df: dd.DataFrame = dd.read_parquet(str(path), engine="pyarrow")

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, path: Path, data: pd.DataFrame) -> CellStore:
        """Create a new cell metadata Parquet file from a pandas DataFrame.

        Args:
            path: Destination path for ``cells.parquet``.  Must not exist.
            data: DataFrame containing at minimum the columns ``cell_id``,
                ``x``, and ``y``.  Additional columns (``fov_id``, ``area``,
                ``tile_id``, etc.) are preserved as-is.

        Returns:
            A :class:`CellStore` opened over the newly written file.

        Raises:
            FileExistsError: If ``path`` already exists.
            ValueError: If any required column is missing from ``data``, or if
                ``cell_id`` contains duplicate values.
        """
        if path.exists():
            raise FileExistsError(f"CellStore already exists at {path}. Use CellStore.open() instead.")
        _validate_dataframe(data)
        data.to_parquet(str(path), engine="pyarrow", index=False)
        return cls(path=path)

    @classmethod
    def open(cls, path: Path) -> CellStore:
        """Open an existing cell metadata Parquet file.

        Args:
            path: Path to an existing ``cells.parquet`` file.

        Returns:
            A :class:`CellStore` wrapping the file.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"No cell store found at {path}")
        return cls(path=path)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """Total number of cells.

        Note: triggers a count operation on the Dask graph (fast — reads only
        Parquet footer metadata, not row data).
        """
        return len(self.df)

    @property
    def columns(self) -> list[str]:
        """List of column names available in the store."""
        return list(self.df.columns)

    @property
    def dtypes(self) -> pd.Series:  # type: ignore[type-arg]
        """Column dtypes as a pandas Series."""
        return self.df.dtypes

    def has_column(self, name: str) -> bool:
        """Return True if ``name`` is a column in the store.

        Args:
            name: Column name to check.

        Returns:
            ``True`` if the column exists, ``False`` otherwise.
        """
        return name in self.df.columns

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_columns(self, columns: list[str]) -> dd.DataFrame:
        """Return a lazy subset of columns.

        Args:
            columns: Column names to select.

        Returns:
            A ``dask.dataframe.DataFrame`` with only the requested columns.

        Raises:
            ValueError: If any requested column is not present.

        Example:
            >>> coords = store.select_columns(["cell_id", "x", "y"]).compute()
        """
        missing = [c for c in columns if c not in self.df.columns]
        if missing:
            raise ValueError(f"Columns not found in CellStore: {missing}")
        return self.df[columns]

    def filter(self, **kwargs: Any) -> dd.DataFrame:
        """Return a lazy filtered view using equality conditions.

        Each keyword argument becomes an equality filter on the named column.
        Multiple filters are combined with AND logic.

        Args:
            **kwargs: Column-name/value pairs to filter on.
                Example: ``fov_id=3``, ``cluster="tumor"``.

        Returns:
            A filtered ``dask.dataframe.DataFrame``.

        Raises:
            ValueError: If any filter column does not exist.

        Example:
            >>> fov3_cells = store.filter(fov_id=3).compute()
        """
        missing = [k for k in kwargs if k not in self.df.columns]
        if missing:
            raise ValueError(f"Filter columns not found in CellStore: {missing}")
        result = self.df
        for col, val in kwargs.items():
            result = result[result[col] == val]
        return result

    def within_bbox(self, x_min: float, y_min: float, x_max: float, y_max: float) -> dd.DataFrame:
        """Return cells whose centroids fall within a bounding box.

        Args:
            x_min: Left boundary in micrometers.
            y_min: Bottom boundary in micrometers.
            x_max: Right boundary in micrometers.
            y_max: Top boundary in micrometers.

        Returns:
            A lazy ``dask.dataframe.DataFrame`` of matching cells.

        Raises:
            ValueError: If ``x_min >= x_max`` or ``y_min >= y_max``.

        Example:
            >>> roi = store.within_bbox(0, 0, 1000, 1000).compute()
        """
        if x_min >= x_max:
            raise ValueError(f"x_min ({x_min}) must be less than x_max ({x_max})")
        if y_min >= y_max:
            raise ValueError(f"y_min ({y_min}) must be less than y_max ({y_max})")
        df = self.df
        return df[(df["x"] >= x_min) & (df["x"] <= x_max) & (df["y"] >= y_min) & (df["y"] <= y_max)]

    # ------------------------------------------------------------------
    # Write / update
    # ------------------------------------------------------------------

    def add_column(self, name: str, values: pd.Series) -> None:  # type: ignore[type-arg]
        """Add or overwrite a column in the in-memory Dask graph.

        The new column is not persisted until :meth:`save` is called.

        Args:
            name: Column name.
            values: A pandas Series aligned to the cell index.  Length must
                match the number of partitions' combined rows.

        Raises:
            ValueError: If ``len(values)`` does not match :attr:`n_cells`.

        Example:
            >>> labels = pd.Series(cluster_result)
            >>> store.add_column("cluster", labels)
            >>> store.save()
        """
        if len(values) != self.n_cells:
            raise ValueError(f"values length {len(values)} does not match n_cells={self.n_cells}")
        self.df[name] = dd.from_pandas(values, npartitions=self.df.npartitions)

    def save(self, path: Path | None = None) -> None:
        """Write the current Dask DataFrame back to Parquet.

        Args:
            path: Destination path.  Defaults to :attr:`path` (in-place
                overwrite).

        Example:
            >>> store.add_column("cluster", labels)
            >>> store.save()
        """
        dest = path or self.path
        computed: pd.DataFrame = self.df.compute()
        computed.to_parquet(str(dest), engine="pyarrow", index=False)
        # Reload the dask frame so it reflects the new state
        self.df = dd.read_parquet(str(dest), engine="pyarrow")
        if path is not None:
            self.path = dest

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CellStore("
            f"n_cells={self.n_cells}, "
            f"columns={self.columns}, "
            f"path={self.path})"
        )


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that a DataFrame meets CellStore requirements.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If required columns are missing or ``cell_id`` has
            duplicate values.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    if df["cell_id"].duplicated().any():
        raise ValueError("cell_id column contains duplicate values")
