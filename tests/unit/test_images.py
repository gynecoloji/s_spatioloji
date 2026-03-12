"""Unit tests for s_spatioloji.data.images.MorphologyImageStore."""

from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np
import pytest
import tifffile

from s_spatioloji.data.images import MorphologyImageStore

# Image dimensions used across tests
N_CHANNELS = 3
HEIGHT = 128
WIDTH = 256


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_ome_tiff(path: Path, data: np.ndarray, n_levels: int = 1) -> Path:
    """Write a minimal OME-TIFF (single or multi-level) to *path*."""
    if n_levels == 1:
        tifffile.imwrite(
            str(path),
            data,
            photometric="minisblack",
            metadata={"axes": "CYX"},
        )
    else:
        with tifffile.TiffWriter(str(path), bigtiff=False) as tw:
            tw.write(
                data,
                subifds=n_levels - 1,
                photometric="minisblack",
                metadata={"axes": "CYX"},
            )
            factor = 2
            for _ in range(n_levels - 1):
                tw.write(
                    data[:, ::factor, ::factor],
                    subfiletype=1,
                    photometric="minisblack",
                )
                factor *= 2
    return path


@pytest.fixture()
def ome_path(tmp_path: Path) -> Path:
    data = np.arange(N_CHANNELS * HEIGHT * WIDTH, dtype=np.uint16).reshape(
        N_CHANNELS, HEIGHT, WIDTH
    )
    return _write_ome_tiff(tmp_path / "morphology.ome.tif", data, n_levels=1)


@pytest.fixture()
def pyramid_path(tmp_path: Path) -> Path:
    data = np.arange(N_CHANNELS * HEIGHT * WIDTH, dtype=np.uint16).reshape(
        N_CHANNELS, HEIGHT, WIDTH
    )
    return _write_ome_tiff(tmp_path / "pyramid.ome.tif", data, n_levels=3)


@pytest.fixture()
def store(ome_path: Path) -> MorphologyImageStore:
    return MorphologyImageStore.open(ome_path)


@pytest.fixture()
def pyramid_store(pyramid_path: Path) -> MorphologyImageStore:
    return MorphologyImageStore.open(pyramid_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpen:
    def test_open_existing(self, ome_path: Path) -> None:
        s = MorphologyImageStore.open(ome_path)
        assert s is not None
        s.close()

    def test_open_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            MorphologyImageStore.open(tmp_path / "missing.ome.tif")


class TestMetadata:
    def test_shape(self, store: MorphologyImageStore) -> None:
        assert store.shape == (N_CHANNELS, HEIGHT, WIDTH)

    def test_dtype(self, store: MorphologyImageStore) -> None:
        assert store.dtype == np.uint16

    def test_axes_contains_cyx(self, store: MorphologyImageStore) -> None:
        assert "C" in store.axes
        assert "Y" in store.axes
        assert "X" in store.axes

    def test_n_levels_single(self, store: MorphologyImageStore) -> None:
        assert store.n_levels == 1

    def test_n_levels_pyramid(self, pyramid_store: MorphologyImageStore) -> None:
        assert pyramid_store.n_levels == 3

    def test_level_shape_level0(self, store: MorphologyImageStore) -> None:
        assert store.level_shape(0) == (N_CHANNELS, HEIGHT, WIDTH)

    def test_level_shape_pyramid(self, pyramid_store: MorphologyImageStore) -> None:
        assert pyramid_store.level_shape(0) == (N_CHANNELS, HEIGHT, WIDTH)
        h1, w1 = pyramid_store.level_shape(1)[1], pyramid_store.level_shape(1)[2]
        assert h1 == HEIGHT // 2
        assert w1 == WIDTH // 2

    def test_level_shape_out_of_range(self, store: MorphologyImageStore) -> None:
        with pytest.raises(ValueError, match="level must be in"):
            store.level_shape(99)


class TestToDask:
    def test_returns_dask_array(self, store: MorphologyImageStore) -> None:
        arr = store.to_dask(level=0)
        assert isinstance(arr, da.Array)

    def test_shape_matches(self, store: MorphologyImageStore) -> None:
        arr = store.to_dask(level=0)
        assert arr.shape == (N_CHANNELS, HEIGHT, WIDTH)

    def test_dtype_matches(self, store: MorphologyImageStore) -> None:
        arr = store.to_dask(level=0)
        assert arr.dtype == np.uint16

    def test_lazy_no_compute(self, store: MorphologyImageStore) -> None:
        # Should not raise and should not load data
        arr = store.to_dask(level=0)
        assert arr is not None

    def test_pyramid_level1(self, pyramid_store: MorphologyImageStore) -> None:
        arr = pyramid_store.to_dask(level=1)
        assert arr.shape == (N_CHANNELS, HEIGHT // 2, WIDTH // 2)

    def test_invalid_level(self, store: MorphologyImageStore) -> None:
        with pytest.raises(ValueError, match="level must be in"):
            store.to_dask(level=5)

    def test_compute_values(self, store: MorphologyImageStore) -> None:
        result = store.to_dask(level=0).compute()
        expected = np.arange(N_CHANNELS * HEIGHT * WIDTH, dtype=np.uint16).reshape(
            N_CHANNELS, HEIGHT, WIDTH
        )
        np.testing.assert_array_equal(result, expected)


class TestReadRegion:
    def test_full_image(self, store: MorphologyImageStore) -> None:
        patch = store.read_region(x=0, y=0, width=WIDTH, height=HEIGHT)
        assert patch.shape[-2:] == (HEIGHT, WIDTH)

    def test_patch_shape_all_channels(self, store: MorphologyImageStore) -> None:
        patch = store.read_region(x=10, y=20, width=64, height=32)
        # shape includes channel dim: (C, H, W)
        assert patch.shape[-2:] == (32, 64)

    def test_patch_single_channel(self, store: MorphologyImageStore) -> None:
        patch = store.read_region(x=0, y=0, width=64, height=32, channel=0)
        assert patch.ndim == 2
        assert patch.shape == (32, 64)

    def test_patch_values_correct(self, store: MorphologyImageStore) -> None:
        # channel 0, top-left 4×4 — verify pixel values match written data
        patch = store.read_region(x=0, y=0, width=4, height=4, channel=0)
        expected = np.arange(N_CHANNELS * HEIGHT * WIDTH, dtype=np.uint16).reshape(
            N_CHANNELS, HEIGHT, WIDTH
        )[0, :4, :4]
        np.testing.assert_array_equal(patch, expected)

    def test_out_of_bounds_raises(self, store: MorphologyImageStore) -> None:
        with pytest.raises(ValueError, match="extends beyond"):
            store.read_region(x=WIDTH - 10, y=0, width=20, height=HEIGHT)

    def test_negative_xy_raises(self, store: MorphologyImageStore) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            store.read_region(x=-1, y=0, width=64, height=32)

    def test_zero_width_raises(self, store: MorphologyImageStore) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            store.read_region(x=0, y=0, width=0, height=32)

    def test_zero_height_raises(self, store: MorphologyImageStore) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            store.read_region(x=0, y=0, width=32, height=0)

    def test_invalid_level_raises(self, store: MorphologyImageStore) -> None:
        with pytest.raises(ValueError, match="level must be in"):
            store.read_region(x=0, y=0, width=32, height=32, level=5)


class TestUmToPx:
    def test_no_metadata_defaults_to_one(self, store: MorphologyImageStore) -> None:
        # No XResolution tag written → falls back to 1 µm/px
        px = store.um_to_px(100.0, level=0)
        assert px == 100

    def test_higher_level_scales(self, pyramid_store: MorphologyImageStore) -> None:
        px0 = pyramid_store.um_to_px(100.0, level=0)
        px1 = pyramid_store.um_to_px(100.0, level=1)
        assert px1 == px0 // 2

    def test_negative_um_raises(self, store: MorphologyImageStore) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            store.um_to_px(-1.0)


class TestContextManager:
    def test_context_manager(self, ome_path: Path) -> None:
        with MorphologyImageStore.open(ome_path) as s:
            arr = s.to_dask(level=0)
            assert arr.shape == (N_CHANNELS, HEIGHT, WIDTH)


class TestRepr:
    def test_repr(self, store: MorphologyImageStore) -> None:
        r = repr(store)
        assert "MorphologyImageStore" in r
        assert str(HEIGHT) in r
        assert str(WIDTH) in r
