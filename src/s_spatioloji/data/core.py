"""s_spatioloji: the central data object for spatial transcriptomics datasets.

``s_spatioloji`` is the single entry point for every operation in the data
engine.  It wires together the four storage backends — expression, cells,
boundaries, and morphology — behind lazy properties.  Nothing is read from
disk until a property is first accessed.

Typical usage::

    sj = s_spatioloji.open("/data/experiment1")

    # Inspect without loading
    sj.cells.n_cells          # reads Parquet footer only
    sj.expression.shape       # reads Zarr metadata only

    # Spatial query → still lazy
    roi = sj.cells.within_bbox(0, 0, 1000, 1000)

    # Materialise only what you need
    coords = roi.compute()[["cell_id", "x", "y"]]

    # Chunked iteration
    for tile in sj.iter_tiles():
        process(tile)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import dask.dataframe as dd

from s_spatioloji.data.boundaries import BoundaryStore
from s_spatioloji.data.cells import CellStore
from s_spatioloji.data.config import SpatiolojiConfig
from s_spatioloji.data.expression import ExpressionStore
from s_spatioloji.data.images import MorphologyImageStore

# ---------------------------------------------------------------------------
# TileView — a spatially scoped sub-object
# ---------------------------------------------------------------------------


class TileView:
    """A spatially bounded view of an ``s_spatioloji`` dataset.

    Returned by :meth:`s_spatioloji.iter_tiles`.  Exposes the same four
    backends as the parent object but pre-filtered to the tile's spatial
    extent (plus overlap guard band).

    Args:
        parent: The parent ``s_spatioloji`` instance.
        x_min: Left boundary of the tile in µm (without overlap).
        y_min: Bottom boundary of the tile in µm (without overlap).
        x_max: Right boundary of the tile in µm (without overlap).
        y_max: Top boundary of the tile in µm (without overlap).
        tile_id: Integer index identifying this tile in the grid.

    Attributes:
        tile_id: Tile index.
        x_min, y_min, x_max, y_max: Core tile boundaries (no overlap).
    """

    def __init__(
        self,
        parent: s_spatioloji,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        tile_id: int,
    ) -> None:
        self._parent = parent
        self.tile_id = tile_id
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

        cfg = parent.config
        ovlp = cfg.tile.overlap
        self._qx_min = x_min - ovlp
        self._qx_max = x_max + ovlp
        self._qy_min = y_min - ovlp
        self._qy_max = y_max + ovlp

    # Lazy filtered views of each backend
    @property
    def cells(self):  # type: ignore[return]
        """Lazy dask.dataframe of cells within this tile (with overlap)."""
        return self._parent.cells.within_bbox(
            self._qx_min, self._qy_min, self._qx_max, self._qy_max
        )

    @property
    def boundaries(self) -> BoundaryStore | None:
        """Boundary rows intersecting this tile, or ``None`` if unavailable."""
        if not self._parent.has_boundaries:
            return None
        return self._parent.boundaries.query_bbox(
            self._qx_min, self._qy_min, self._qx_max, self._qy_max
        )  # type: ignore[return-value]

    @property
    def morphology(self) -> MorphologyImageStore | None:
        """Morphology image store, or ``None`` if unavailable."""
        return self._parent._morphology if self._parent.has_morphology else None

    def __repr__(self) -> str:
        return (
            f"TileView(tile_id={self.tile_id}, "
            f"x=[{self.x_min}, {self.x_max}], "
            f"y=[{self.y_min}, {self.y_max}])"
        )


# ---------------------------------------------------------------------------
# Maps accessor
# ---------------------------------------------------------------------------


class Maps:
    """Accessor for computed results stored in the ``maps/`` directory.

    Provides dict-like access to Parquet and Zarr results written by
    compute functions.  Keys are bare names (no file extension).

    Args:
        sj: The parent ``s_spatioloji`` instance.

    Example:
        >>> sj.maps["X_pca"]           # → dask.dataframe
        >>> sj.maps["expression_scvi"] # → ExpressionStore
        >>> sj.maps.keys()             # → list of available keys
        >>> sj.maps.has("X_umap")      # → bool
    """

    def __init__(self, sj: s_spatioloji) -> None:
        self._sj = sj

    def _maps_dir(self) -> Path:
        """Return the ``maps/`` directory path (may not exist yet)."""
        return self._sj.config.root / "maps"

    def __getitem__(self, key: str) -> dd.DataFrame | ExpressionStore:
        """Look up a result by bare key name.

        Args:
            key: Bare key name (e.g. ``"X_pca"``, ``"expression_scvi"``).

        Returns:
            ``dask.dataframe.DataFrame`` for Parquet results, or
            :class:`ExpressionStore` for Zarr results.

        Raises:
            KeyError: If no result exists for ``key``.
            KeyError: If ``key`` is ``"_scvi_model"`` (internal directory).
        """
        if key == "_scvi_model":
            raise KeyError("_scvi_model is an internal directory, not a user-accessible key")

        maps_dir = self._maps_dir()

        # 1. maps/<key>.parquet
        parquet_path = maps_dir / f"{key}.parquet"
        if parquet_path.exists():
            return dd.read_parquet(str(parquet_path), engine="pyarrow")

        # 2. maps/<key>.zarr/
        zarr_maps = maps_dir / f"{key}.zarr"
        if zarr_maps.exists() and zarr_maps.is_dir():
            return ExpressionStore.open(zarr_maps, self._sj.config.chunks)

        # 3. <root>/<key>.zarr/
        zarr_root = self._sj.config.root / f"{key}.zarr"
        if zarr_root.exists() and zarr_root.is_dir():
            return ExpressionStore.open(zarr_root, self._sj.config.chunks)

        raise KeyError(f"No result found for key {key!r}")

    def has(self, key: str) -> bool:
        """Return ``True`` if a result exists for ``key``.

        Args:
            key: Bare key name.

        Returns:
            ``True`` if the key resolves to an existing file.
        """
        try:
            self[key]
            return True
        except KeyError:
            return False

    def keys(self) -> list[str]:
        """Return all discoverable bare key names.

        Scans ``maps/*.parquet``, ``maps/*.zarr/``, and
        ``<root>/expression_*.zarr/`` and returns deduplicated names.

        Returns:
            Sorted list of bare key names.
        """
        result: list[str] = []
        maps_dir = self._maps_dir()

        if maps_dir.exists():
            for p in sorted(maps_dir.iterdir()):
                if p.suffix == ".parquet" and p.is_file():
                    result.append(p.stem)
                elif p.suffix == ".zarr" and p.is_dir() and p.stem != "_scvi_model":
                    result.append(p.stem)

        for p in sorted(self._sj.config.root.iterdir()):
            if p.name.startswith("expression_") and p.suffix == ".zarr" and p.is_dir():
                result.append(p.stem)

        return list(dict.fromkeys(result))

    def delete(self, key: str) -> None:
        """Delete a result from disk.

        Follows the same lookup order as ``__getitem__`` and removes only
        the first match.

        Args:
            key: Bare key name.

        Raises:
            KeyError: If no result exists for ``key``.
        """
        import shutil

        maps_dir = self._maps_dir()

        parquet_path = maps_dir / f"{key}.parquet"
        if parquet_path.exists():
            parquet_path.unlink()
            return

        zarr_maps = maps_dir / f"{key}.zarr"
        if zarr_maps.exists() and zarr_maps.is_dir():
            shutil.rmtree(zarr_maps)
            return

        zarr_root = self._sj.config.root / f"{key}.zarr"
        if zarr_root.exists() and zarr_root.is_dir():
            shutil.rmtree(zarr_root)
            return

        raise KeyError(f"No result found for key {key!r}")

    def __repr__(self) -> str:
        return f"Maps(keys={self.keys()})"


# ---------------------------------------------------------------------------
# s_spatioloji
# ---------------------------------------------------------------------------


class s_spatioloji:
    """Central data object for a spatial transcriptomics dataset.

    Wires together expression, cell metadata, boundary polygons, and
    morphology image backends.  All backends are opened lazily — file handles
    are acquired on first property access, not at construction time.

    Use :meth:`open` to open an existing dataset directory and
    :meth:`create` to initialise a new one from raw data.

    Args:
        config: Dataset configuration including all paths and chunk settings.

    Attributes:
        config: The active :class:`~s_spatioloji.data.config.SpatiolojiConfig`.

    Example:
        >>> sj = s_spatioloji.open("/data/xenium_run1")
        >>> sj.cells.n_cells
        4_200_000
        >>> arr = sj.expression.select_genes([0, 1, 2]).compute()
    """

    def __init__(self, config: SpatiolojiConfig) -> None:
        self.config = config
        # Backend instances — populated lazily
        self._expression: ExpressionStore | None = None
        self._cells: CellStore | None = None
        self._boundaries: BoundaryStore | None = None
        self._morphology: MorphologyImageStore | None = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, path: Path | str) -> s_spatioloji:
        """Open an existing ``s_spatioloji`` dataset directory.

        The directory must contain at least ``cells.parquet`` and
        ``expression.zarr``.  Boundaries and morphology are optional.

        Args:
            path: Root directory of the dataset.

        Returns:
            An :class:`s_spatioloji` instance ready for use.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            FileNotFoundError: If the required ``cells.parquet`` or
                ``expression.zarr`` are missing.

        Example:
            >>> sj = s_spatioloji.open("/data/experiment1")
        """
        root = Path(path)
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {root}")

        config = SpatiolojiConfig(root=root)
        _check_required(config)
        return cls(config=config)

    @classmethod
    def create(
        cls,
        path: Path | str,
        config: SpatiolojiConfig | None = None,
    ) -> s_spatioloji:
        """Create a new, empty dataset directory.

        Creates the root directory and the ``_index`` sub-directory.
        Does **not** create any Zarr or Parquet files — use the individual
        store ``create`` methods after calling this.

        Args:
            path: Root directory to create.  Must not already exist.
            config: Optional config override.  Defaults to
                ``SpatiolojiConfig(root=path)``.

        Returns:
            An :class:`s_spatioloji` instance pointing at the new directory.

        Raises:
            FileExistsError: If ``path`` already exists.

        Example:
            >>> sj = s_spatioloji.create("/data/new_experiment")
        """
        root = Path(path)
        if root.exists():
            raise FileExistsError(f"Dataset directory already exists: {root}")
        root.mkdir(parents=True)
        (root / "_index").mkdir()
        cfg = config or SpatiolojiConfig(root=root)
        return cls(config=cfg)

    # ------------------------------------------------------------------
    # Maps accessor
    # ------------------------------------------------------------------

    @property
    def maps(self) -> Maps:
        """Accessor for computed results in the ``maps/`` directory.

        Returns:
            A :class:`Maps` instance for looking up results by key.

        Example:
            >>> sj.maps["X_pca"]
            >>> sj.maps.keys()
        """
        return Maps(self)

    # ------------------------------------------------------------------
    # Lazy backend properties
    # ------------------------------------------------------------------

    @property
    def expression(self) -> ExpressionStore:
        """Lazy expression matrix backend (Zarr + dask.array).

        Raises:
            FileNotFoundError: If ``expression.zarr`` is not present.
        """
        if self._expression is None:
            p = self.config.paths.expression
            if not p.exists():
                raise FileNotFoundError(f"expression.zarr not found at {p}")
            self._expression = ExpressionStore.open(p, self.config.chunks)
        return self._expression

    @property
    def cells(self) -> CellStore:
        """Lazy cell metadata backend (Parquet + dask.dataframe).

        Raises:
            FileNotFoundError: If ``cells.parquet`` is not present.
        """
        if self._cells is None:
            p = self.config.paths.cells
            if not p.exists():
                raise FileNotFoundError(f"cells.parquet not found at {p}")
            self._cells = CellStore.open(p)
        return self._cells

    @property
    def boundaries(self) -> BoundaryStore:
        """Lazy polygon boundary backend (GeoParquet + STRtree).

        Raises:
            FileNotFoundError: If ``boundaries.parquet`` is not present.
                Check :attr:`has_boundaries` before accessing.
        """
        if self._boundaries is None:
            p = self.config.paths.boundaries
            if not p.exists():
                raise FileNotFoundError(
                    f"boundaries.parquet not found at {p}. "
                    "Check has_boundaries before accessing."
                )
            self._boundaries = BoundaryStore.open(p)
        return self._boundaries

    @property
    def morphology(self) -> MorphologyImageStore:
        """Lazy morphology image backend (OME-TIFF + dask.array).

        Raises:
            FileNotFoundError: If ``morphology.ome.tif`` is not present.
                Check :attr:`has_morphology` before accessing.
        """
        if self._morphology is None:
            p = self.config.paths.morphology
            if not p.exists():
                raise FileNotFoundError(
                    f"morphology.ome.tif not found at {p}. "
                    "Check has_morphology before accessing."
                )
            self._morphology = MorphologyImageStore.open(p)
        return self._morphology

    # ------------------------------------------------------------------
    # Availability flags
    # ------------------------------------------------------------------

    @property
    def has_boundaries(self) -> bool:
        """True if ``boundaries.parquet`` exists in the dataset directory."""
        return self.config.paths.boundaries.exists()

    @property
    def has_morphology(self) -> bool:
        """True if ``morphology.ome.tif`` exists in the dataset directory."""
        return self.config.paths.morphology.exists()

    # ------------------------------------------------------------------
    # Dataset metadata
    # ------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """Total number of cells (reads Parquet footer, does not load data)."""
        return self.cells.n_cells

    @property
    def n_genes(self) -> int:
        """Total number of genes (reads Zarr metadata, does not load data)."""
        return self.expression.n_genes

    @property
    def obs_columns(self) -> list[str]:
        """Column names available in the cell metadata store."""
        return self.cells.columns

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iter_tiles(
        self,
        tile_size: float | None = None,
        overlap: float | None = None,
    ) -> Iterator[TileView]:
        """Iterate over the dataset in spatial tiles.

        Tiles are non-overlapping squares of ``tile_size`` µm.  Each
        :class:`TileView` carries an additional ``overlap`` µm guard band on
        all sides so polygon queries near tile edges are not clipped.

        The tile grid is computed from the bounding box of all cell centroids.

        Args:
            tile_size: Side length of each tile in µm.  Defaults to the
                value in :attr:`config`.
            overlap: Guard-band width in µm.  Defaults to the value in
                :attr:`config`.

        Yields:
            :class:`TileView` objects, one per tile, in row-major order.

        Example:
            >>> for tile in sj.iter_tiles(tile_size=512, overlap=50):
            ...     df = tile.cells.compute()
            ...     process(df)
        """
        t_size = tile_size or self.config.tile.tile_size
        ovlp = overlap if overlap is not None else self.config.tile.overlap

        coords = self.cells.select_columns(["x", "y"]).compute()
        x_min_g = float(coords["x"].min())
        x_max_g = float(coords["x"].max())
        y_min_g = float(coords["y"].min())
        y_max_g = float(coords["y"].max())

        tile_id = 0
        y = y_min_g
        while y < y_max_g:
            x = x_min_g
            while x < x_max_g:
                yield TileView(
                    parent=self,
                    x_min=x,
                    y_min=y,
                    x_max=min(x + t_size, x_max_g),
                    y_max=min(y + t_size, y_max_g),
                    tile_id=tile_id,
                )
                x += t_size
                tile_id += 1
            y += t_size

    def iter_cells(self, batch_size: int = 50_000) -> Iterator[CellStore]:
        """Iterate over cells in sequential batches.

        Each batch is a :class:`CellStore` wrapping a row-slice of the
        underlying Parquet file.  Useful for algorithms that must visit every
        cell but cannot load the full dataset at once.

        Args:
            batch_size: Number of cells per batch.

        Yields:
            :class:`CellStore` instances, each covering ``batch_size`` cells.

        Raises:
            ValueError: If ``batch_size`` is not positive.

        Example:
            >>> for batch in sj.iter_cells(batch_size=10_000):
            ...     df = batch.df.compute()
            ...     run_model(df)
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        n = self.n_cells
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_df = self.cells.df.loc[start:end - 1]
            store = CellStore.__new__(CellStore)
            store.path = self.config.paths.cells
            store.df = batch_df
            yield store

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_cells(self, path: Path | None = None) -> None:
        """Write the current cell metadata back to Parquet.

        Args:
            path: Destination path.  Defaults to the existing
                ``cells.parquet`` (in-place overwrite).
        """
        self.cells.save(path)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"s_spatioloji(root={self.config.root}"]
        try:
            parts.append(f"n_cells={self.n_cells}")
        except FileNotFoundError:
            parts.append("n_cells=?")
        try:
            parts.append(f"n_genes={self.n_genes}")
        except FileNotFoundError:
            parts.append("n_genes=?")
        parts.append(f"boundaries={'yes' if self.has_boundaries else 'no'}")
        parts.append(f"morphology={'yes' if self.has_morphology else 'no'}")
        return ", ".join(parts) + ")"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_required(config: SpatiolojiConfig) -> None:
    """Verify required files exist in the dataset directory.

    Args:
        config: Config whose paths to check.

    Raises:
        FileNotFoundError: If ``cells.parquet`` or ``expression.zarr`` are
            missing.
    """
    missing = []
    if not config.paths.cells.exists():
        missing.append(str(config.paths.cells))
    if not config.paths.expression.exists():
        missing.append(str(config.paths.expression))
    if missing:
        raise FileNotFoundError(
            f"Required dataset files not found: {missing}. "
            "Run the appropriate loader (e.g. from_xenium()) first."
        )
