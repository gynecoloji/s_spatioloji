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


# ---------------------------------------------------------------------------
# New tests: pixel_size_at, px_to_um, ImageCollection
# ---------------------------------------------------------------------------

import json

import pandas as pd

from s_spatioloji.data.images import ImageCollection


class TestPixelSizeAt:
    def test_level0_default(self, store: MorphologyImageStore) -> None:
        assert store.pixel_size_at(0) == 1.0

    def test_higher_level_doubles(self, pyramid_store: MorphologyImageStore) -> None:
        base = pyramid_store.pixel_size_at(0)
        assert pyramid_store.pixel_size_at(1) == base * 2
        assert pyramid_store.pixel_size_at(2) == base * 4


class TestPxToUm:
    def test_roundtrip(self, store: MorphologyImageStore) -> None:
        um = 100.0
        px = store.um_to_px(um, level=0)
        result = store.px_to_um(float(px), level=0)
        assert abs(result - um) < 1.5

    def test_level_scaling(self, pyramid_store: MorphologyImageStore) -> None:
        px = 100.0
        um0 = pyramid_store.px_to_um(px, level=0)
        um1 = pyramid_store.px_to_um(px, level=1)
        assert um1 == um0 * 2

    def test_negative_raises(self, store: MorphologyImageStore) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            store.px_to_um(-1.0)


@pytest.fixture()
def images_dir(tmp_path: Path) -> Path:
    """Dataset root with images/ dir, images_meta.json, and two OME-TIFFs."""
    root = tmp_path / "dataset"
    root.mkdir()
    img_dir = root / "images"
    img_dir.mkdir()

    data = np.arange(N_CHANNELS * HEIGHT * WIDTH, dtype=np.uint16).reshape(
        N_CHANNELS, HEIGHT, WIDTH
    )
    _write_ome_tiff(img_dir / "morphology_focus_0000.ome.tif", data, n_levels=3)
    _write_ome_tiff(img_dir / "morphology_focus_0001.ome.tif", data, n_levels=1)

    meta = {
        "pixel_size": 0.2125,
        "default_image": "morphology_focus_0000",
        "files": {
            "morphology_focus_0000": "morphology_focus_0000.ome.tif",
            "morphology_focus_0001": "morphology_focus_0001.ome.tif",
        },
        "xenium_version": "3.0.0.15",
    }
    (root / "images_meta.json").write_text(json.dumps(meta))
    return root


@pytest.fixture()
def legacy_root(tmp_path: Path) -> Path:
    """Dataset root with only morphology.ome.tif at root (old layout)."""
    root = tmp_path / "legacy"
    root.mkdir()
    data = np.arange(N_CHANNELS * HEIGHT * WIDTH, dtype=np.uint16).reshape(
        N_CHANNELS, HEIGHT, WIDTH
    )
    _write_ome_tiff(root / "morphology.ome.tif", data, n_levels=1)
    return root


class TestImageCollection:
    def test_keys(self, images_dir: Path) -> None:
        ic = ImageCollection(images_dir)
        assert sorted(ic.keys()) == ["morphology_focus_0000", "morphology_focus_0001"]
        ic.close()

    def test_has(self, images_dir: Path) -> None:
        ic = ImageCollection(images_dir)
        assert ic.has("morphology_focus_0000")
        assert not ic.has("nonexistent")
        ic.close()

    def test_getitem_lazy(self, images_dir: Path) -> None:
        ic = ImageCollection(images_dir)
        store = ic["morphology_focus_0000"]
        assert isinstance(store, MorphologyImageStore)
        assert store.shape == (N_CHANNELS, HEIGHT, WIDTH)
        ic.close()

    def test_getitem_missing_raises(self, images_dir: Path) -> None:
        ic = ImageCollection(images_dir)
        with pytest.raises(KeyError, match="nonexistent"):
            ic["nonexistent"]
        ic.close()

    def test_pixel_size(self, images_dir: Path) -> None:
        ic = ImageCollection(images_dir)
        assert ic.pixel_size == 0.2125
        ic.close()

    def test_pixel_size_at(self, images_dir: Path) -> None:
        ic = ImageCollection(images_dir)
        assert ic.pixel_size_at(0) == 0.2125
        assert ic.pixel_size_at(3) == pytest.approx(0.2125 * 8)
        ic.close()

    def test_default_image(self, images_dir: Path) -> None:
        ic = ImageCollection(images_dir)
        assert ic.default_image == "morphology_focus_0000"
        ic.close()

    def test_scale_coordinates(self, images_dir: Path) -> None:
        ic = ImageCollection(images_dir)
        df = pd.DataFrame({"cell_id": ["a", "b"], "x": [10.0, 20.0], "y": [30.0, 40.0]})
        scaled = ic.scale_coordinates(df, level=0)
        expected_x = [10.0 / 0.2125, 20.0 / 0.2125]
        assert scaled["x"].tolist() == pytest.approx(expected_x)
        assert df["x"].tolist() == [10.0, 20.0]  # original unchanged
        ic.close()

    def test_scale_coordinates_level3(self, images_dir: Path) -> None:
        ic = ImageCollection(images_dir)
        df = pd.DataFrame({"cell_id": ["a"], "x": [10.0], "y": [20.0]})
        scaled = ic.scale_coordinates(df, level=3)
        ps3 = 0.2125 * 8
        assert scaled["x"].iloc[0] == pytest.approx(10.0 / ps3)
        ic.close()

    def test_context_manager(self, images_dir: Path) -> None:
        with ImageCollection(images_dir) as ic:
            assert ic.has("morphology_focus_0000")


class TestImageCollectionBackwardCompat:
    def test_legacy_wraps_single_file(self, legacy_root: Path) -> None:
        ic = ImageCollection(legacy_root)
        assert ic.has("morphology")
        assert ic.keys() == ["morphology"]
        store = ic["morphology"]
        assert isinstance(store, MorphologyImageStore)
        ic.close()

    def test_legacy_pixel_size_default(self, legacy_root: Path) -> None:
        ic = ImageCollection(legacy_root)
        assert ic.pixel_size == 1.0
        ic.close()

    def test_legacy_default_image(self, legacy_root: Path) -> None:
        ic = ImageCollection(legacy_root)
        assert ic.default_image == "morphology"
        ic.close()

    def test_empty_root(self, tmp_path: Path) -> None:
        root = tmp_path / "empty"
        root.mkdir()
        ic = ImageCollection(root)
        assert ic.keys() == []
        assert ic.pixel_size == 1.0
        ic.close()
