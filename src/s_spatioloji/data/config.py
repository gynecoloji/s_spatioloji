"""Configuration dataclasses for s_spatioloji dataset layout and backend settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TileConfig:
    """Spatial tile grid parameters.

    Args:
        tile_size: Side length of each square tile in micrometers.
        overlap: Overlap between adjacent tiles in micrometers. Used as a
            guard band so polygon queries near tile boundaries are not clipped.

    Example:
        >>> tile = TileConfig(tile_size=512.0, overlap=50.0)
        >>> tile.tile_size
        512.0
    """

    tile_size: float = 512.0
    overlap: float = 50.0


@dataclass
class ChunkConfig:
    """Zarr chunk shape settings for each array store.

    Args:
        expression_cells: Number of cells per chunk along the cell axis.
            Tuned for spatial locality — one chunk ≈ one tile's worth of cells.
        expression_genes: Number of genes per chunk along the gene axis.
            Set to -1 to keep all genes in a single chunk (recommended when
            n_genes < 10_000).
        image_y: Tile height in pixels for the morphology image array.
        image_x: Tile width in pixels for the morphology image array.

    Example:
        >>> chunks = ChunkConfig(expression_cells=2048, expression_genes=-1)
        >>> chunks.expression_cells
        2048
    """

    expression_cells: int = 2048
    expression_genes: int = -1
    image_y: int = 512
    image_x: int = 512


@dataclass
class StorePaths:
    """Resolved paths to every storage artifact inside a dataset directory.

    All paths are derived from a single root directory.  No files are created
    or validated here — path resolution only.

    Args:
        root: Root directory of the dataset (must already exist when opening).

    Example:
        >>> paths = StorePaths(root=Path("/data/experiment1"))
        >>> paths.expression.name
        'expression.zarr'
    """

    root: Path

    @property
    def expression(self) -> Path:
        """Path to the expression Zarr store."""
        return self.root / "expression.zarr"

    @property
    def cells(self) -> Path:
        """Path to the cell metadata Parquet file."""
        return self.root / "cells.parquet"

    @property
    def transcripts(self) -> Path:
        """Path to the partitioned transcript Parquet directory."""
        return self.root / "transcripts"

    @property
    def boundaries(self) -> Path:
        """Path to the cell boundary GeoParquet file."""
        return self.root / "boundaries.parquet"

    @property
    def morphology(self) -> Path:
        """Path to the morphology OME-TIFF image."""
        return self.root / "morphology.ome.tif"

    @property
    def index(self) -> Path:
        """Path to the precomputed index directory."""
        return self.root / "_index"

    @property
    def spatial_index(self) -> Path:
        """Path to the serialized R-tree spatial index."""
        return self.index / "spatial.rtree"

    @property
    def knn_graph(self) -> Path:
        """Path to the precomputed kNN graph (scipy sparse CSR, .npz)."""
        return self.index / "knn.npz"

    @property
    def ann_index(self) -> Path:
        """Path to the HNSWLIB/FAISS approximate nearest-neighbour index."""
        return self.index / "ann.index"


@dataclass
class SpatiolojiConfig:
    """Top-level configuration for an s_spatioloji dataset.

    Bundles all path, chunking, and tiling settings into one object that is
    passed through every layer of the data engine.  Instantiate this directly
    when creating a new dataset, or let ``s_spatioloji.open()`` build it from
    the dataset root directory.

    Args:
        root: Root directory of the dataset.
        tile: Spatial tile grid parameters.
        chunks: Zarr chunk shape settings.
        n_workers: Number of Dask workers for parallel compute. ``None`` uses
            the Dask default (typically one thread per CPU core).
        compression: Zarr compressor name passed to Blosc. One of
            ``"zstd"``, ``"lz4"``, ``"blosclz"``.  ``"zstd"`` gives the best
            compression ratio; ``"lz4"`` is faster.

    Example:
        >>> cfg = SpatiolojiConfig(root=Path("/data/experiment1"))
        >>> cfg.paths.expression.name
        'expression.zarr'
        >>> cfg.tile.tile_size
        512.0
    """

    root: Path
    tile: TileConfig = field(default_factory=TileConfig)
    chunks: ChunkConfig = field(default_factory=ChunkConfig)
    n_workers: int | None = None
    compression: str = "zstd"

    def __post_init__(self) -> None:
        """Validate config values after construction.

        Raises:
            ValueError: If tile_size or overlap are non-positive, if
                expression_cells chunk is non-positive, or if compression
                is not a supported value.
        """
        if self.tile.tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {self.tile.tile_size}")
        if self.tile.overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {self.tile.overlap}")
        if self.tile.overlap >= self.tile.tile_size:
            raise ValueError(
                f"overlap ({self.tile.overlap}) must be less than tile_size ({self.tile.tile_size})"
            )
        if self.chunks.expression_cells <= 0:
            raise ValueError(f"expression_cells chunk must be positive, got {self.chunks.expression_cells}")
        valid_compression = {"zstd", "lz4", "blosclz"}
        if self.compression not in valid_compression:
            raise ValueError(f"compression must be one of {valid_compression}, got {self.compression!r}")

    @property
    def paths(self) -> StorePaths:
        """Resolved storage paths derived from the root directory.

        Example:
            >>> cfg = SpatiolojiConfig(root=Path("/data/experiment1"))
            >>> cfg.paths.cells.name
            'cells.parquet'
        """
        return StorePaths(root=self.root)
