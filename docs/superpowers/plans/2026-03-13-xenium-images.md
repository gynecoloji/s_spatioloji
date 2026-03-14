# Xenium Image Handling Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend image handling to support Xenium multi-channel focus images, Z-stack morphology, `experiment.xenium` parsing, and level-aware coordinate scaling.

**Architecture:** New `ImageCollection` class wraps multiple `MorphologyImageStore` instances with lazy loading. `images_meta.json` persists pixel size and file registry. `from_xenium` parses `experiment.xenium` and ingests all image files. Coordinate scaling helpers convert µm to pixels at any pyramid level.

**Tech Stack:** tifffile (existing), json (stdlib), shapely.affinity (for polygon scaling), geopandas

**Spec:** `docs/superpowers/specs/2026-03-13-xenium-images-design.md`

---

## File Structure

```
src/s_spatioloji/data/
├── images.py        # MODIFY: add ImageCollection, extend MorphologyImageStore (pixel_size_at, px_to_um)
├── config.py        # MODIFY: add images_dir, images_meta to StorePaths
├── core.py          # MODIFY: add images property, update morphology for backward compat
└── io.py            # MODIFY: update from_xenium to parse experiment.xenium and ingest multi-image

tests/unit/
├── test_images.py   # MODIFY: add ImageCollection tests, pixel_size_at, px_to_um tests
└── test_io.py       # MODIFY: add Xenium multi-image ingestion tests
```

---

## Chunk 1: ImageCollection + MorphologyImageStore Extensions (Tasks 1–2)

### Task 1: Extend MorphologyImageStore + tests

**Files:**
- Modify: `src/s_spatioloji/data/images.py`
- Modify: `tests/unit/test_images.py`

- [ ] **Step 1: Write tests for new MorphologyImageStore methods**

Append to `tests/unit/test_images.py`:

```python
class TestPixelSizeAt:
    def test_level0_default(self, store: MorphologyImageStore) -> None:
        # No XResolution metadata → fallback 1.0 µm/px
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
        assert abs(result - um) < 1.5  # rounding tolerance

    def test_level_scaling(self, pyramid_store: MorphologyImageStore) -> None:
        px = 100.0
        um0 = pyramid_store.px_to_um(px, level=0)
        um1 = pyramid_store.px_to_um(px, level=1)
        assert um1 == um0 * 2  # higher level → larger µm per pixel

    def test_negative_raises(self, store: MorphologyImageStore) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            store.px_to_um(-1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_images.py::TestPixelSizeAt -v`
Expected: FAIL (no attribute `pixel_size_at`)

- [ ] **Step 3: Add `pixel_size_at` and `px_to_um` to MorphologyImageStore**

Add after the existing `um_to_px` method in `images.py`:

```python
def pixel_size_at(self, level: int = 0) -> float:
    """Physical pixel size in µm at a given pyramid level.

    Each level is 2x downsampled, so pixel size doubles per level.

    Args:
        level: Pyramid level (0 = full resolution).

    Returns:
        Pixel size in micrometers.

    Raises:
        ValueError: If ``level`` is out of range.
    """
    self._check_level(level)
    return self._pixel_size_um() * (2 ** level)

def px_to_um(self, px: float, level: int = 0) -> float:
    """Convert pixels to micrometers at a given pyramid level.

    Args:
        px: Distance in pixels.
        level: Pyramid level (pixel size scales with downsampling factor).

    Returns:
        Equivalent distance in micrometers.

    Raises:
        ValueError: If ``px`` is negative.
    """
    if px < 0:
        raise ValueError(f"px must be non-negative, got {px}")
    return px * self.pixel_size_at(level)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_images.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/s_spatioloji/data/images.py tests/unit/test_images.py
git commit -m "feat(images): add pixel_size_at and px_to_um to MorphologyImageStore"
```

### Task 2: ImageCollection class + tests

**Files:**
- Modify: `src/s_spatioloji/data/images.py`
- Modify: `src/s_spatioloji/data/config.py`
- Modify: `tests/unit/test_images.py`

- [ ] **Step 1: Write ImageCollection tests**

Append to `tests/unit/test_images.py`:

```python
import json

import pandas as pd

from s_spatioloji.data.images import ImageCollection


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
        # px = um / pixel_size_at(level)
        expected_x = [10.0 / 0.2125, 20.0 / 0.2125]
        assert scaled["x"].tolist() == pytest.approx(expected_x)
        # Original unchanged
        assert df["x"].tolist() == [10.0, 20.0]
        ic.close()

    def test_scale_coordinates_level3(self, images_dir: Path) -> None:
        ic = ImageCollection(images_dir)
        df = pd.DataFrame({"cell_id": ["a"], "x": [10.0], "y": [20.0]})
        scaled = ic.scale_coordinates(df, level=3)
        ps3 = 0.2125 * 8  # 1.7
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_images.py::TestImageCollection -v`
Expected: FAIL (no class ImageCollection)

- [ ] **Step 3: Add `images_dir` and `images_meta` to StorePaths**

In `config.py`, add after the `morphology` property:

```python
@property
def images_dir(self) -> Path:
    """Path to the images/ directory."""
    return self.root / "images"

@property
def images_meta(self) -> Path:
    """Path to images_meta.json."""
    return self.root / "images_meta.json"
```

- [ ] **Step 4: Implement ImageCollection in `images.py`**

Add at the end of the file:

```python
import json

import pandas as pd

if TYPE_CHECKING:
    import geopandas as gpd


class ImageCollection:
    """Lazy collection of named MorphologyImageStore instances.

    Reads ``images_meta.json`` for pixel size and file registry.
    Falls back to wrapping a single ``morphology.ome.tif`` at the
    dataset root for backward compatibility with older datasets.

    Args:
        root: Dataset root directory.

    Example:
        >>> ic = ImageCollection(root)
        >>> store = ic["morphology_focus_0000"]  # lazy-loaded
        >>> ic.pixel_size_at(level=3)
        1.7
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._stores: dict[str, MorphologyImageStore | None] = {}
        self._files: dict[str, str] = {}
        self._pixel_size: float = 1.0
        self._default_image: str = ""
        self._xenium_version: str = ""

        meta_path = root / "images_meta.json"
        images_dir = root / "images"

        if meta_path.exists():
            with open(str(meta_path)) as f:
                meta = json.load(f)
            self._pixel_size = meta.get("pixel_size", 1.0)
            self._default_image = meta.get("default_image", "")
            self._files = meta.get("files", {})
            self._xenium_version = meta.get("xenium_version", "")
            # Initialize lazy slots
            for name in self._files:
                self._stores[name] = None
        elif (root / "morphology.ome.tif").exists():
            # Backward compat: wrap single root-level file
            self._files = {"morphology": "morphology.ome.tif"}
            self._stores = {"morphology": None}
            self._default_image = "morphology"
            self._pixel_size = 1.0

    @property
    def pixel_size(self) -> float:
        """Base pixel size in µm (level 0)."""
        return self._pixel_size

    def pixel_size_at(self, level: int = 0) -> float:
        """Pixel size at a given pyramid level: ``pixel_size * 2^level``.

        Args:
            level: Pyramid level (0 = full resolution).

        Returns:
            Pixel size in micrometers.
        """
        return self._pixel_size * (2 ** level)

    @property
    def default_image(self) -> str:
        """Name of the default image (typically ``'morphology_focus_0000'``)."""
        return self._default_image

    def keys(self) -> list[str]:
        """List of available image names.

        Returns:
            Sorted list of image name strings.
        """
        return sorted(self._files.keys())

    def has(self, name: str) -> bool:
        """Check if an image is available.

        Args:
            name: Image name to check.

        Returns:
            True if the image exists in the collection.
        """
        return name in self._files

    def __getitem__(self, name: str) -> MorphologyImageStore:
        """Lazy-load and return a MorphologyImageStore by name.

        Args:
            name: Image name (e.g., ``"morphology_focus_0000"``).

        Returns:
            A :class:`MorphologyImageStore` instance.

        Raises:
            KeyError: If ``name`` is not in the collection.
        """
        if name not in self._files:
            raise KeyError(f"Image '{name}' not found. Available: {self.keys()}")

        if self._stores[name] is None:
            filename = self._files[name]
            # Check images/ dir first, then root for backward compat
            images_dir = self._root / "images"
            if (images_dir / filename).exists():
                path = images_dir / filename
            else:
                path = self._root / filename
            self._stores[name] = MorphologyImageStore.open(path)

        return self._stores[name]  # type: ignore[return-value]

    def scale_coordinates(
        self,
        df: pd.DataFrame,
        level: int = 0,
    ) -> pd.DataFrame:
        """Scale x, y columns from µm to pixels at the given pyramid level.

        Returns a copy — the original DataFrame is not modified.
        Formula: ``px = um / pixel_size_at(level)``.

        Args:
            df: DataFrame with ``x`` and ``y`` columns in micrometers.
            level: Pyramid level.

        Returns:
            DataFrame copy with x, y scaled to pixels.
        """
        ps = self.pixel_size_at(level)
        result = df.copy()
        result["x"] = result["x"] / ps
        result["y"] = result["y"] / ps
        return result

    def scale_polygons(
        self,
        gdf: gpd.GeoDataFrame,
        level: int = 0,
    ) -> gpd.GeoDataFrame:
        """Scale polygon geometries from µm to pixels at the given level.

        Returns a copy — the original GeoDataFrame is not modified.

        Args:
            gdf: GeoDataFrame with geometry column in micrometers.
            level: Pyramid level.

        Returns:
            GeoDataFrame copy with scaled geometries.
        """
        from shapely.affinity import scale as shapely_scale

        ps = self.pixel_size_at(level)
        factor = 1.0 / ps
        result = gdf.copy()
        result["geometry"] = result["geometry"].apply(
            lambda g: shapely_scale(g, xfact=factor, yfact=factor, origin=(0, 0))
        )
        return result

    def close(self) -> None:
        """Close all opened image stores."""
        for name, store in self._stores.items():
            if store is not None:
                store.close()
                self._stores[name] = None

    def __enter__(self) -> ImageCollection:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"ImageCollection("
            f"n_images={len(self._files)}, "
            f"pixel_size={self._pixel_size}, "
            f"keys={self.keys()})"
        )
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/unit/test_images.py -v`
Expected: ALL PASS

- [ ] **Step 6: Lint and commit**

```bash
ruff check src/s_spatioloji/data/images.py src/s_spatioloji/data/config.py tests/unit/test_images.py --fix
ruff format src/s_spatioloji/data/images.py src/s_spatioloji/data/config.py tests/unit/test_images.py
git add src/s_spatioloji/data/images.py src/s_spatioloji/data/config.py tests/unit/test_images.py
git commit -m "feat(images): add ImageCollection with lazy loading, pixel scaling, and backward compat"
```

---

## Chunk 2: Core Integration + from_xenium (Tasks 3–4)

### Task 3: Integrate ImageCollection into s_spatioloji

**Files:**
- Modify: `src/s_spatioloji/data/core.py`
- Modify: `tests/unit/test_core.py`

- [ ] **Step 1: Write tests for `sj.images` and updated `sj.morphology`**

Add to `tests/unit/test_core.py` (find appropriate location alongside existing morphology tests):

```python
from s_spatioloji.data.images import ImageCollection


class TestImages:
    def test_images_type(self, sj_with_images):
        assert isinstance(sj_with_images.images, ImageCollection)

    def test_images_keys(self, sj_with_images):
        assert "morphology_focus_0000" in sj_with_images.images.keys()

    def test_has_images_true(self, sj_with_images):
        assert sj_with_images.has_images

    def test_has_images_false(self, sj_minimal):
        assert not sj_minimal.has_images

    def test_morphology_returns_default(self, sj_with_images):
        store = sj_with_images.morphology
        assert store is not None

    def test_images_pixel_size(self, sj_with_images):
        assert sj_with_images.images.pixel_size == 0.2125
```

You'll need to add a `sj_with_images` fixture to the test file's conftest or inline. This fixture creates a dataset with `images/` dir and `images_meta.json`:

```python
@pytest.fixture()
def sj_with_images(dataset_path):
    """Dataset with images/ directory and images_meta.json."""
    import json
    import tifffile

    img_dir = dataset_path / "images"
    img_dir.mkdir()
    data = np.zeros((1, 64, 64), dtype=np.uint16)
    tifffile.imwrite(str(img_dir / "morphology_focus_0000.ome.tif"), data,
                     photometric="minisblack", metadata={"axes": "CYX"})
    meta = {
        "pixel_size": 0.2125,
        "default_image": "morphology_focus_0000",
        "files": {"morphology_focus_0000": "morphology_focus_0000.ome.tif"},
        "xenium_version": "3.0.0.15",
    }
    (dataset_path / "images_meta.json").write_text(json.dumps(meta))
    return s_spatioloji.open(dataset_path)
```

- [ ] **Step 2: Add `images` and `has_images` to `s_spatioloji` in `core.py`**

Add a new cached attribute `_images: ImageCollection | None = None` in `__init__`.

Add import: `from s_spatioloji.data.images import ImageCollection`

Add properties after the existing `morphology` property:

```python
@property
def images(self) -> ImageCollection:
    """Lazy-loaded image collection.

    Returns:
        An :class:`~s_spatioloji.data.images.ImageCollection` wrapping
        all available images with lazy loading.
    """
    if self._images is None:
        self._images = ImageCollection(self.config.root)
    return self._images

@property
def has_images(self) -> bool:
    """True if ``images_meta.json`` or ``images/`` directory exists."""
    return (
        self.config.paths.images_meta.exists()
        or self.config.paths.images_dir.exists()
    )
```

Update the existing `morphology` property to try `images` first:

```python
@property
def morphology(self) -> MorphologyImageStore:
    """Lazy morphology image backend (OME-TIFF + dask.array).

    If an :class:`ImageCollection` is available, returns the default image
    (typically ``morphology_focus_0000``). Otherwise falls back to
    ``morphology.ome.tif`` at the dataset root.

    Raises:
        FileNotFoundError: If no morphology image is available.
    """
    # Try ImageCollection first
    if self.has_images and self.images.default_image:
        return self.images[self.images.default_image]

    # Legacy fallback
    if self._morphology is None:
        p = self.config.paths.morphology
        if not p.exists():
            raise FileNotFoundError(
                f"morphology.ome.tif not found at {p}. "
                "Check has_morphology before accessing."
            )
        self._morphology = MorphologyImageStore.open(p)
    return self._morphology
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/unit/test_core.py -v`
Expected: ALL PASS (existing + new)

- [ ] **Step 4: Commit**

```bash
git add src/s_spatioloji/data/core.py tests/unit/test_core.py
git commit -m "feat(core): add images property with ImageCollection integration"
```

### Task 4: Update from_xenium to ingest multi-image + experiment.xenium

**Files:**
- Modify: `src/s_spatioloji/data/io.py`
- Modify: `tests/unit/test_io.py`

- [ ] **Step 1: Write tests for Xenium multi-image ingestion**

Add to `tests/unit/test_io.py`:

```python
import json

import tifffile


def _create_xenium_images(xenium_dir: Path) -> None:
    """Create mock Xenium image files and experiment.xenium."""
    data = np.zeros((1, 64, 64), dtype=np.uint16)

    # morphology.ome.tif
    tifffile.imwrite(str(xenium_dir / "morphology.ome.tif"), data,
                     photometric="minisblack", metadata={"axes": "CYX"})

    # morphology_focus/ directory with 4 channels
    focus_dir = xenium_dir / "morphology_focus"
    focus_dir.mkdir()
    for i in range(4):
        tifffile.imwrite(str(focus_dir / f"morphology_focus_{i:04d}.ome.tif"), data,
                         photometric="minisblack", metadata={"axes": "CYX"})

    # experiment.xenium
    specs = {
        "pixel_size": 0.2125,
        "analysis_sw_version": "xenium-3.0.0.15",
        "images": {
            "morphology_filepath": "morphology.ome.tif",
            "morphology_focus_filepath": "morphology_focus/morphology_focus_0000.ome.tif",
        },
    }
    (xenium_dir / "experiment.xenium").write_text(json.dumps(specs))


class TestFromXeniumImages:
    def test_images_meta_written(self, xenium_dir_with_images, tmp_path):
        """from_xenium writes images_meta.json."""
        out = tmp_path / "output"
        sj = from_xenium(xenium_dir_with_images, out)
        assert (out / "images_meta.json").exists()
        meta = json.loads((out / "images_meta.json").read_text())
        assert meta["pixel_size"] == 0.2125

    def test_focus_images_ingested(self, xenium_dir_with_images, tmp_path):
        out = tmp_path / "output"
        sj = from_xenium(xenium_dir_with_images, out)
        assert (out / "images").is_dir()
        assert (out / "images" / "morphology_focus_0000.ome.tif").exists()
        assert (out / "images" / "morphology_focus_0003.ome.tif").exists()

    def test_morphology_ingested(self, xenium_dir_with_images, tmp_path):
        out = tmp_path / "output"
        sj = from_xenium(xenium_dir_with_images, out)
        assert (out / "images" / "morphology.ome.tif").exists()

    def test_images_collection_works(self, xenium_dir_with_images, tmp_path):
        out = tmp_path / "output"
        sj = from_xenium(xenium_dir_with_images, out)
        assert sj.has_images
        assert "morphology_focus_0000" in sj.images.keys()
        assert sj.images.pixel_size == 0.2125

    def test_backward_compat_symlink(self, xenium_dir_with_images, tmp_path):
        out = tmp_path / "output"
        sj = from_xenium(xenium_dir_with_images, out)
        # morphology.ome.tif at root for backward compat
        assert (out / "morphology.ome.tif").exists()

    def test_default_image_is_focus_0000(self, xenium_dir_with_images, tmp_path):
        out = tmp_path / "output"
        sj = from_xenium(xenium_dir_with_images, out)
        assert sj.images.default_image == "morphology_focus_0000"
```

You'll need a `xenium_dir_with_images` fixture that creates a full mock Xenium directory (cells, expression, boundaries, AND images):

```python
@pytest.fixture()
def xenium_dir_with_images(xenium_dir):
    """xenium_dir fixture extended with image files and experiment.xenium."""
    _create_xenium_images(xenium_dir)
    return xenium_dir
```

(This depends on whatever existing `xenium_dir` fixture creates the base Xenium mock with cells.parquet, cell_feature_matrix/, etc.)

- [ ] **Step 2: Implement `_parse_xenium_specs` and `_ingest_xenium_images` in `io.py`**

Add helper functions:

```python
def _parse_xenium_specs(src: Path) -> dict:
    """Read experiment.xenium and return parsed JSON dict.

    Args:
        src: Xenium output directory.

    Returns:
        Parsed JSON dict, or empty dict if file not found.
    """
    specs_path = src / "experiment.xenium"
    if not specs_path.exists():
        return {}
    with open(str(specs_path)) as f:
        return json.load(f)


def _parse_xenium_version(specs: dict) -> str:
    """Extract version string from analysis_sw_version.

    Args:
        specs: Parsed experiment.xenium dict.

    Returns:
        Version string (e.g., '3.0.0.15') or empty string.
    """
    version_str = specs.get("analysis_sw_version", "")
    # Strip 'xenium-' prefix
    if version_str.lower().startswith("xenium-"):
        return version_str[7:]
    return version_str


def _ingest_xenium_images(
    src: Path,
    dst: Path,
    specs: dict,
    copy: bool,
) -> None:
    """Discover and ingest all Xenium image files.

    Creates ``images/`` directory and ``images_meta.json`` in ``dst``.

    Args:
        src: Xenium output directory.
        dst: Dataset output directory.
        specs: Parsed experiment.xenium dict.
        copy: If True, copy files; otherwise symlink.
    """
    images_dir = dst / "images"
    images_dir.mkdir(exist_ok=True)

    files: dict[str, str] = {}
    default_image = ""

    # 1. Discover focus images from morphology_focus/ directory
    focus_dir = src / "morphology_focus"
    if focus_dir.exists():
        focus_files = sorted(focus_dir.glob("*.ome.tif"))
        for f in focus_files:
            name = f.stem.replace(".ome", "")  # e.g., "morphology_focus_0000"
            files[name] = f.name
            _link_or_copy(f, images_dir / f.name, copy=copy)
        if focus_files:
            first_name = focus_files[0].stem.replace(".ome", "")
            default_image = first_name

    # 2. Morphology Z-stack
    morph_src = src / "morphology.ome.tif"
    if morph_src.exists():
        files["morphology"] = "morphology.ome.tif"
        _link_or_copy(morph_src, images_dir / "morphology.ome.tif", copy=copy)
        if not default_image:
            default_image = "morphology"

    # 3. MIP (v1 only)
    mip_src = src / "morphology_mip.ome.tif"
    if mip_src.exists():
        files["morphology_mip"] = "morphology_mip.ome.tif"
        _link_or_copy(mip_src, images_dir / "morphology_mip.ome.tif", copy=copy)

    # 4. Backward compat: symlink default focus to root morphology.ome.tif
    if default_image and default_image.startswith("morphology_focus"):
        compat_src = images_dir / files[default_image]
        compat_dst = dst / "morphology.ome.tif"
        _link_or_copy(compat_src, compat_dst, copy=copy)

    # 5. Write images_meta.json
    pixel_size = specs.get("pixel_size", 0.2125)
    meta = {
        "pixel_size": pixel_size,
        "default_image": default_image,
        "files": files,
        "xenium_version": _parse_xenium_version(specs),
    }
    (dst / "images_meta.json").write_text(json.dumps(meta, indent=2))
```

Add `import json` to the top of `io.py`.

- [ ] **Step 3: Update `from_xenium` to use new functions**

Replace the image ingestion section (step 4 in from_xenium, around lines 163-166):

```python
# ---- 4. Images + experiment.xenium -----------------------------------
specs = _parse_xenium_specs(src)
_ingest_xenium_images(src, dst, specs, copy=copy_image)

# Legacy fallback: if no focus images found, try root morphology.ome.tif
if not (dst / "images_meta.json").exists():
    morph_src = src / "morphology.ome.tif"
    if morph_src.exists():
        _link_or_copy(morph_src, dst / "morphology.ome.tif", copy=copy_image)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/unit/test_io.py -v`
Expected: ALL PASS (existing + new)

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/unit/ -v`
Expected: ALL PASS

- [ ] **Step 6: Lint and commit**

```bash
ruff check src/s_spatioloji/data/io.py tests/unit/test_io.py --fix
ruff format src/s_spatioloji/data/io.py tests/unit/test_io.py
git add src/s_spatioloji/data/io.py tests/unit/test_io.py
git commit -m "feat(io): update from_xenium to ingest multi-channel images and parse experiment.xenium"
```
