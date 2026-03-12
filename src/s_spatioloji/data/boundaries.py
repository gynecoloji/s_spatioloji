"""BoundaryStore: polygon boundaries stored as GeoParquet with a lazy STRtree index.

Each row represents one cell boundary as a Shapely geometry (Polygon or
MultiPolygon).  The file is a standard GeoParquet with an additional set of
precomputed bounding-box columns (``x_min``, ``y_min``, ``x_max``,
``y_max``) so that spatial range queries can be answered with a fast Parquet
predicate pushdown before the STRtree is consulted.

The STRtree index is built lazily on first polygon query and cached in memory.
It is never serialised to disk in this class — for persistent indexing use
``IndexManager`` (future module).

Required columns in every valid BoundaryStore:
    - ``cell_id`` (str)    — must match ``CellStore.cell_id``
    - ``geometry`` (WKB)   — Shapely-compatible geometry stored by geopandas
    - ``x_min``, ``y_min``, ``x_max``, ``y_max`` (float) — bbox columns
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

if TYPE_CHECKING:
    from numpy.typing import NDArray

REQUIRED_COLUMNS: tuple[str, ...] = ("cell_id", "geometry", "x_min", "y_min", "x_max", "y_max")
_BBOX_COLS: tuple[str, ...] = ("x_min", "y_min", "x_max", "y_max")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _add_bbox_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compute and attach bounding-box columns from the geometry column.

    Args:
        gdf: GeoDataFrame with a valid geometry column.

    Returns:
        The same GeoDataFrame with ``x_min``, ``y_min``, ``x_max``,
        ``y_max`` columns added (or overwritten).
    """
    bounds = gdf.geometry.bounds  # returns DataFrame with minx, miny, maxx, maxy
    gdf = gdf.copy()
    gdf["x_min"] = bounds["minx"].values
    gdf["y_min"] = bounds["miny"].values
    gdf["x_max"] = bounds["maxx"].values
    gdf["y_max"] = bounds["maxy"].values
    return gdf


def _validate_geodataframe(gdf: gpd.GeoDataFrame) -> None:
    """Validate that a GeoDataFrame meets BoundaryStore requirements.

    Args:
        gdf: GeoDataFrame to validate.

    Raises:
        ValueError: If required columns are missing or ``cell_id`` has
            duplicate values.
        TypeError: If ``gdf`` is not a GeoDataFrame.
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"Expected a GeoDataFrame, got {type(gdf).__name__}")
    missing = [c for c in ("cell_id", "geometry") if c not in gdf.columns]
    if missing:
        raise ValueError(f"GeoDataFrame is missing required columns: {missing}")
    if gdf["cell_id"].duplicated().any():
        raise ValueError("cell_id column contains duplicate values")


# ---------------------------------------------------------------------------
# BoundaryStore
# ---------------------------------------------------------------------------


class BoundaryStore:
    """Polygon boundary store backed by GeoParquet with a lazy STRtree index.

    Reads are performed via ``geopandas.read_parquet``, which uses PyArrow
    under the hood.  The bbox columns allow callers to pre-filter with a
    bounding-box check before loading full geometries.

    The STRtree is built once on first access to any polygon-query method and
    cached for the lifetime of the store object.

    Args:
        path: Path to the ``boundaries.parquet`` GeoParquet file.

    Attributes:
        path: Resolved path to the GeoParquet file.

    Example:
        >>> store = BoundaryStore.open(path)
        >>> hits = store.query_bbox(0, 0, 500, 500)
        >>> touching = store.query_polygon(my_roi_polygon)
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._gdf: gpd.GeoDataFrame | None = None  # loaded on demand

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, path: Path, gdf: gpd.GeoDataFrame) -> BoundaryStore:
        """Create a new boundary GeoParquet file from a GeoDataFrame.

        Bounding-box columns are computed automatically from the geometry.

        Args:
            path: Destination path for ``boundaries.parquet``.  Must not
                exist.
            gdf: GeoDataFrame with at least ``cell_id`` and ``geometry``
                columns.  The CRS is preserved if set.

        Returns:
            A :class:`BoundaryStore` wrapping the newly written file.

        Raises:
            FileExistsError: If ``path`` already exists.
            ValueError: If required columns are missing or ``cell_id`` has
                duplicate values.
            TypeError: If ``gdf`` is not a GeoDataFrame.
        """
        if path.exists():
            raise FileExistsError(f"BoundaryStore already exists at {path}. Use BoundaryStore.open() instead.")
        _validate_geodataframe(gdf)
        gdf = _add_bbox_columns(gdf)
        gdf.to_parquet(str(path), engine="pyarrow", index=False)
        return cls(path=path)

    @classmethod
    def open(cls, path: Path) -> BoundaryStore:
        """Open an existing boundary GeoParquet file.

        Args:
            path: Path to an existing ``boundaries.parquet`` file.

        Returns:
            A :class:`BoundaryStore` wrapping the file.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"No boundary store found at {path}")
        return cls(path=path)

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def load(self) -> gpd.GeoDataFrame:
        """Load the full GeoDataFrame into memory and cache it.

        Subsequent calls return the cached copy without re-reading disk.

        Returns:
            A ``geopandas.GeoDataFrame`` with all boundary rows.

        Note:
            For large datasets (millions of cells) prefer the query methods
            which load only relevant rows.
        """
        if self._gdf is None:
            self._gdf = gpd.read_parquet(str(self.path))
        return self._gdf

    @cached_property
    def _tree(self) -> STRtree:
        """STRtree built from all geometries.  Built once, cached forever."""
        gdf = self.load()
        return STRtree(gdf.geometry.values)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """Number of cell boundaries in the store."""
        return len(self.load())

    @property
    def columns(self) -> list[str]:
        """Column names available in the store."""
        return list(self.load().columns)

    @property
    def crs(self) -> str | None:
        """Coordinate reference system string, or ``None`` if not set."""
        crs = self.load().crs
        return str(crs) if crs is not None else None

    # ------------------------------------------------------------------
    # Spatial queries
    # ------------------------------------------------------------------

    def query_bbox(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> gpd.GeoDataFrame:
        """Return all cells whose bounding boxes intersect the given rectangle.

        Uses the precomputed bbox columns for a fast pandas filter before
        returning the result — no STRtree required for this operation.

        Args:
            x_min: Left boundary in micrometers.
            y_min: Bottom boundary in micrometers.
            x_max: Right boundary in micrometers.
            y_max: Top boundary in micrometers.

        Returns:
            A ``GeoDataFrame`` of matching boundary rows.

        Raises:
            ValueError: If ``x_min >= x_max`` or ``y_min >= y_max``.

        Example:
            >>> roi = store.query_bbox(0, 0, 500, 500)
        """
        if x_min >= x_max:
            raise ValueError(f"x_min ({x_min}) must be less than x_max ({x_max})")
        if y_min >= y_max:
            raise ValueError(f"y_min ({y_min}) must be less than y_max ({y_max})")
        gdf = self.load()
        mask = (
            (gdf["x_max"] >= x_min)
            & (gdf["x_min"] <= x_max)
            & (gdf["y_max"] >= y_min)
            & (gdf["y_min"] <= y_max)
        )
        return gdf[mask].copy()

    def query_polygon(
        self,
        polygon: BaseGeometry,
        predicate: str = "intersects",
    ) -> gpd.GeoDataFrame:
        """Return all cells whose geometries satisfy a spatial predicate
        against the given polygon.

        Uses the STRtree for candidate retrieval, then applies the exact
        predicate filter.

        Predicate semantics follow Shapely 2.x STRtree conventions where
        ``polygon`` is the *input* geometry and each cell boundary is the
        *tree* geometry:

        - ``"intersects"``  — cell boundary and polygon share any point
          (most common; picks up cells touching the polygon edge).
        - ``"contains"``    — polygon entirely contains the cell boundary
          (use this to find cells *fully inside* the query polygon).
        - ``"within"``      — cell boundary entirely contains the polygon
          (use this to find cells that *surround* the query polygon).
        - ``"overlaps"``    — partial overlap; neither fully contains the other.
        - ``"covers"``      — polygon covers the cell boundary (relaxed contains).
        - ``"covered_by"``  — cell boundary is covered by the polygon.

        Args:
            polygon: A Shapely geometry defining the query region.
            predicate: Spatial relationship to test (default ``"intersects"``).

        Returns:
            A ``GeoDataFrame`` of matching boundary rows.

        Raises:
            ValueError: If ``predicate`` is not a supported value.

        Example:
            >>> import shapely
            >>> roi = shapely.from_wkt("POLYGON((0 0, 500 0, 500 500, 0 500, 0 0))")
            >>> hits = store.query_polygon(roi, predicate="intersects")
            >>> interior = store.query_polygon(roi, predicate="contains")
        """
        valid_predicates = {"intersects", "within", "contains", "overlaps", "covers", "covered_by"}
        if predicate not in valid_predicates:
            raise ValueError(f"predicate must be one of {valid_predicates}, got {predicate!r}")

        gdf = self.load()
        candidate_idx: NDArray[np.intp] = self._tree.query(polygon, predicate=predicate)
        return gdf.iloc[candidate_idx].copy()

    def query_radius(self, x: float, y: float, radius: float) -> gpd.GeoDataFrame:
        """Return all cells whose boundaries intersect a circle.

        Approximates the circle as a buffered point, then runs an STRtree
        intersection query.

        Args:
            x: Circle centre x-coordinate in micrometers.
            y: Circle centre y-coordinate in micrometers.
            radius: Circle radius in micrometers.

        Returns:
            A ``GeoDataFrame`` of matching boundary rows.

        Raises:
            ValueError: If ``radius`` is not positive.

        Example:
            >>> neighbors = store.query_radius(x=1200.0, y=800.0, radius=50.0)
        """
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        from shapely.geometry import Point

        circle = Point(x, y).buffer(radius)
        return self.query_polygon(circle, predicate="intersects")

    def cell_by_id(self, cell_id: str) -> gpd.GeoDataFrame:
        """Return the boundary row(s) for a given cell ID.

        Args:
            cell_id: Cell identifier to look up.

        Returns:
            A ``GeoDataFrame`` with matching rows (typically one row).

        Raises:
            KeyError: If ``cell_id`` is not found.

        Example:
            >>> row = store.cell_by_id("cell_42")
        """
        gdf = self.load()
        result = gdf[gdf["cell_id"] == cell_id]
        if result.empty:
            raise KeyError(f"cell_id {cell_id!r} not found in BoundaryStore")
        return result.copy()

    # ------------------------------------------------------------------
    # Morphology helpers
    # ------------------------------------------------------------------

    def compute_bbox_columns(self) -> gpd.GeoDataFrame:
        """Recompute bounding-box columns from current geometries.

        Useful after geometry edits.  Does not write to disk.

        Returns:
            Updated GeoDataFrame with refreshed bbox columns.
        """
        gdf = self.load()
        return _add_bbox_columns(gdf)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        try:
            n = self.n_cells
        except Exception:
            n = "?"
        return f"BoundaryStore(n_cells={n}, path={self.path})"
