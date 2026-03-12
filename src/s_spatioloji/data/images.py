"""MorphologyImageStore: lazy, tiled access to morphology OME-TIFF images.

The image is memory-mapped via ``tifffile`` and exposed as a ``dask.array``
per pyramid level.  No pixel data is read until ``.compute()`` is called on a
Dask array or :meth:`read_region` is called explicitly.

Expected file: ``morphology.ome.tif``

Assumed axis order: ``(C, Y, X)`` for 2-D data or ``(Z, C, Y, X)`` for
volumetric data.  Channel and Z indices are always explicit in the API so
the caller never needs to know the internal axis layout.

Pyramid levels:
    Level 0 is full resolution.  Higher levels are downsampled by powers of
    two.  If the OME-TIFF has no sub-resolution levels (single-level file),
    :attr:`n_levels` is 1 and only ``level=0`` is valid.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import dask.array as da
import numpy as np
import tifffile
import zarr

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# MorphologyImageStore
# ---------------------------------------------------------------------------


class MorphologyImageStore:
    """Lazy dask.array access to a morphology OME-TIFF image.

    The file handle is opened once and kept open for the lifetime of the
    store.  Call :meth:`close` (or use as a context manager) to release it.

    Args:
        path: Path to the ``morphology.ome.tif`` file.

    Attributes:
        path: Resolved path to the OME-TIFF file.
        n_levels: Number of pyramid resolution levels available.
        axes: Axis string for the image series (e.g. ``"CYX"`` or ``"ZCYX"``).
        shape: Shape of the full-resolution array (level 0).
        dtype: NumPy dtype of the pixel data.

    Example:
        >>> store = MorphologyImageStore.open(path)
        >>> store.n_levels
        4
        >>> dask_arr = store.to_dask(level=0)          # full resolution, lazy
        >>> patch = store.read_region(x=500, y=300, width=256, height=256)
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._tif: tifffile.TiffFile = tifffile.TiffFile(str(path))
        self._series = self._tif.series[0]

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, path: Path) -> MorphologyImageStore:
        """Open an existing OME-TIFF morphology image.

        Args:
            path: Path to an existing ``morphology.ome.tif`` file.

        Returns:
            A :class:`MorphologyImageStore` wrapping the file.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"No morphology image found at {path}")
        return cls(path=path)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def n_levels(self) -> int:
        """Number of pyramid resolution levels (1 if no sub-resolutions)."""
        return len(self._series.levels)

    @property
    def axes(self) -> str:
        """Axis label string for the image series, e.g. ``'CYX'`` or ``'ZCYX'``."""
        return self._series.axes

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the full-resolution (level 0) array."""
        return tuple(self._series.shape)

    @property
    def dtype(self) -> np.dtype:  # type: ignore[type-arg]
        """NumPy dtype of the pixel data."""
        return np.dtype(self._series.dtype)

    def level_shape(self, level: int) -> tuple[int, ...]:
        """Shape of the array at a given pyramid level.

        Args:
            level: Pyramid level index (0 = full resolution).

        Returns:
            Shape tuple for that level.

        Raises:
            ValueError: If ``level`` is out of range.
        """
        self._check_level(level)
        return tuple(self._series.levels[level].shape)

    # ------------------------------------------------------------------
    # Dask access
    # ------------------------------------------------------------------

    def to_dask(self, level: int = 0) -> da.Array:
        """Return the full image at a given pyramid level as a lazy dask.array.

        No pixel data is read until ``.compute()`` is called.

        Args:
            level: Pyramid level (0 = full resolution, higher = downsampled).

        Returns:
            A ``dask.array`` with shape matching :meth:`level_shape`.

        Raises:
            ValueError: If ``level`` is out of range.

        Example:
            >>> arr = store.to_dask(level=0)
            >>> thumbnail = store.to_dask(level=3).compute()
        """
        self._check_level(level)
        zstore = self._tif.aszarr(level=level)
        z: zarr.Array = zarr.open(zstore, mode="r")
        return da.from_zarr(z)

    # ------------------------------------------------------------------
    # Region extraction
    # ------------------------------------------------------------------

    def read_region(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        level: int = 0,
        channel: int | None = None,
    ) -> NDArray:  # type: ignore[type-arg]
        """Read a spatial patch from the image and return it as a numpy array.

        Coordinates are in pixels at the requested pyramid level.  For
        micrometres-to-pixel conversion use :meth:`um_to_px`.

        Args:
            x: Left edge of the patch in pixels (level-space).
            y: Top edge of the patch in pixels (level-space).
            width: Patch width in pixels.
            height: Patch height in pixels.
            level: Pyramid level to read from (default 0 = full resolution).
            channel: Channel index to extract.  ``None`` returns all channels.

        Returns:
            NumPy array.  Shape is ``(height, width)`` if ``channel`` is set,
            or ``(C, height, width)`` / ``(Z, C, height, width)`` otherwise.

        Raises:
            ValueError: If ``level`` is out of range, ``width``/``height``
                are not positive, or the patch extends beyond the image bounds.

        Example:
            >>> patch = store.read_region(x=1024, y=512, width=256, height=256)
            >>> ch0 = store.read_region(x=0, y=0, width=512, height=512, channel=0)
        """
        self._check_level(level)
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be positive, got width={width}, height={height}")

        lvl_shape = self.level_shape(level)
        # Determine Y and X axis positions from the axes string
        ax = self._series.axes.upper()
        y_ax = ax.index("Y")
        x_ax = ax.index("X")
        img_height = lvl_shape[y_ax]
        img_width = lvl_shape[x_ax]

        if x < 0 or y < 0:
            raise ValueError(f"x and y must be >= 0, got x={x}, y={y}")
        if x + width > img_width or y + height > img_height:
            raise ValueError(
                f"Patch [{x}:{x+width}, {y}:{y+height}] extends beyond "
                f"image bounds [{img_width}, {img_height}] at level {level}"
            )

        arr = self.to_dask(level=level).compute()

        # Build a slice tuple matching the axis order
        slices: list[slice | int] = []
        for a in ax:
            if a == "Y":
                slices.append(slice(y, y + height))
            elif a == "X":
                slices.append(slice(x, x + width))
            elif a == "C" and channel is not None:
                slices.append(channel)
            else:
                slices.append(slice(None))

        return arr[tuple(slices)]  # type: ignore[index]

    def um_to_px(self, um: float, level: int = 0) -> int:
        """Convert micrometres to pixels at a given pyramid level.

        Requires the OME-TIFF to embed physical pixel size metadata
        (``XResolution`` / ``YResolution`` tags).  Falls back to 1 µm/px if
        metadata is absent.

        Args:
            um: Distance in micrometres.
            level: Pyramid level (pixel size scales with downsampling factor).

        Returns:
            Equivalent distance in pixels, rounded to the nearest integer.

        Raises:
            ValueError: If ``um`` is negative.
        """
        if um < 0:
            raise ValueError(f"um must be non-negative, got {um}")
        px_size = self._pixel_size_um()
        scale = 2 ** level  # each level is 2× downsampled
        return round(um / (px_size * scale))

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying TiffFile handle and release resources."""
        self._tif.close()

    def __enter__(self) -> MorphologyImageStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MorphologyImageStore("
            f"shape={self.shape}, "
            f"axes={self.axes!r}, "
            f"n_levels={self.n_levels}, "
            f"dtype={self.dtype}, "
            f"path={self.path})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_level(self, level: int) -> None:
        """Raise ValueError if ``level`` is out of range."""
        if not (0 <= level < self.n_levels):
            raise ValueError(f"level must be in [0, {self.n_levels - 1}], got {level}")

    def _pixel_size_um(self) -> float:
        """Return the physical pixel size in µm from OME metadata, or 1.0."""
        try:
            page = self._tif.pages[0]
            # XResolution tag: (numerator, denominator) in pixels-per-unit
            xres = page.tags.get("XResolution")
            if xres is not None:
                num, den = xres.value
                if num > 0:
                    # tifffile stores resolution in pixels/µm when unit is µm
                    return den / num
        except Exception:
            pass
        return 1.0
