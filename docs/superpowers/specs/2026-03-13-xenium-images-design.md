# Xenium Image Handling Design Spec

**Date:** 2026-03-13
**Module:** `src/s_spatioloji/data/images.py`, `src/s_spatioloji/data/io.py`, `src/s_spatioloji/data/config.py`, `src/s_spatioloji/data/core.py`
**Status:** Approved

## Goal

Extend image handling to fully support Xenium output: multi-channel focus images (4 files), Z-stack morphology, MIP, `experiment.xenium` parsing for pixel size (0.2125 µm/px), and level-aware coordinate scaling for pyramid images (8 levels, 2x downsampling per level).

## Xenium Image Output Format

Xenium produces these image files:

- **`morphology.ome.tif`** — multi-focal-plane DAPI Z-stack (large, optional for visualization)
- **`morphology_focus/`** directory containing 1–4 OME-TIFF files:
  - `morphology_focus_0000.ome.tif` — channel 0 (DAPI, always present)
  - `morphology_focus_0001.ome.tif` — channel 1 (boundary stain, with segmentation kit)
  - `morphology_focus_0002.ome.tif` — channel 2 (interior stain)
  - `morphology_focus_0003.ome.tif` — channel 3 (nuclear expansion stain)
- **`morphology_mip.ome.tif`** — maximum intensity projection (v1 only, <2.0.0)
- **`experiment.xenium`** — JSON metadata with `pixel_size` and image file paths

Each OME-TIFF has up to 8 pyramid levels:

| Level | Pixel size (µm) | Downsampling |
|-------|-----------------|--------------|
| 0     | 0.2125          | 1x           |
| 1     | 0.4250          | 2x           |
| 2     | 0.8500          | 4x           |
| 3     | 1.7000          | 8x           |
| 4     | 3.4000          | 16x          |
| 5     | 6.8000          | 32x          |
| 6     | 13.6000         | 64x          |
| 7     | 27.2000         | 128x         |

**Coordinate system:** Cell centroids and polygon vertices are in micrometers. To convert to pixels at a given level: `px = um / (pixel_size * 2^level)`.

**`experiment.xenium` structure** (relevant fields):

```json
{
    "pixel_size": 0.2125,
    "analysis_sw_version": "xenium-3.0.0.15",
    "images": {
        "morphology_filepath": "morphology.ome.tif",
        "morphology_focus_filepath": "morphology_focus/morphology_focus_0000.ome.tif"
    }
}
```

## Architecture

### Dataset Layout Change

```
dataset_root/
├── expression.zarr/
├── cells.parquet
├── boundaries.parquet
├── images/                              # NEW: replaces root-level morphology.ome.tif
│   ├── morphology_focus_0000.ome.tif
│   ├── morphology_focus_0001.ome.tif
│   ├── morphology_focus_0002.ome.tif
│   ├── morphology_focus_0003.ome.tif
│   └── morphology.ome.tif
├── images_meta.json                     # NEW: pixel_size + file registry
├── morphology.ome.tif                   # DEPRECATED: kept for backward compat
└── _index/
```

### `images_meta.json` Format

Written during ingestion. Contains pixel size and a registry of available images:

```json
{
    "pixel_size": 0.2125,
    "default_image": "morphology_focus_0000",
    "files": {
        "morphology_focus_0000": "morphology_focus_0000.ome.tif",
        "morphology_focus_0001": "morphology_focus_0001.ome.tif",
        "morphology_focus_0002": "morphology_focus_0002.ome.tif",
        "morphology_focus_0003": "morphology_focus_0003.ome.tif",
        "morphology": "morphology.ome.tif"
    },
    "xenium_version": "3.0.0.15"
}
```

### Backward Compatibility

If an old dataset has `morphology.ome.tif` at root but no `images/` directory and no `images_meta.json`:
- `ImageCollection` wraps that single file as `"morphology"` with `pixel_size=1.0` (fallback).
- `sj.morphology` returns that store (unchanged behavior).

---

## Component Specifications

### 1. `ImageCollection` (new class in `images.py`)

```python
class ImageCollection:
    def __init__(self, root: Path) -> None:
```

- Reads `images_meta.json` from `root` if it exists.
- Falls back to checking for `morphology.ome.tif` at `root` for backward compatibility.
- Stores lazy references — `MorphologyImageStore` instances are NOT opened until accessed.
- Uses a `dict[str, MorphologyImageStore | None]` cache for lazy loading.

**Properties and methods:**

```python
@property
def pixel_size(self) -> float:
    """Base pixel size in µm (level 0). From images_meta.json or 1.0 fallback."""

def pixel_size_at(self, level: int = 0) -> float:
    """Pixel size at a given pyramid level: pixel_size * 2^level."""

@property
def default_image(self) -> str:
    """Name of the default image (typically 'morphology_focus_0000')."""

def keys(self) -> list[str]:
    """List of available image names."""

def has(self, name: str) -> bool:
    """Check if an image is available."""

def __getitem__(self, name: str) -> MorphologyImageStore:
    """Lazy-load and return a MorphologyImageStore by name.
    Raises KeyError if not found."""

def scale_coordinates(
    self,
    df: pd.DataFrame,
    level: int = 0,
) -> pd.DataFrame:
    """Scale x, y columns from µm to pixels at the given level.
    Returns a copy. Formula: px = um / pixel_size_at(level)."""

def scale_polygons(
    self,
    gdf: gpd.GeoDataFrame,
    level: int = 0,
) -> gpd.GeoDataFrame:
    """Scale polygon geometries from µm to pixels at the given level.
    Returns a copy with scaled geometries using shapely.affinity.scale."""

def close(self) -> None:
    """Close all opened image stores."""
```

**Context manager support:** `__enter__` / `__exit__` that calls `close()`.

### 2. `MorphologyImageStore` Extensions

Add to existing class:

```python
def pixel_size_at(self, level: int = 0) -> float:
    """Pixel size at a given pyramid level.
    Uses embedded TIFF metadata pixel size * 2^level."""

def px_to_um(self, px: float, level: int = 0) -> float:
    """Convert pixels to micrometers at a given pyramid level.
    Formula: um = px * pixel_size_at(level)."""
```

The existing `um_to_px` method already exists but uses `_pixel_size_um()` which reads TIFF tags. This is fine — it will work with the embedded metadata.

### 3. `StorePaths` Extensions (in `config.py`)

Add new properties:

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

The existing `morphology` property stays for backward compatibility.

### 4. `s_spatioloji` Extensions (in `core.py`)

Add new property:

```python
@property
def images(self) -> ImageCollection:
    """Lazy-loaded image collection. Returns ImageCollection even if no images exist."""

@property
def has_images(self) -> bool:
    """True if images_meta.json or images/ directory exists."""
```

Modify existing `morphology` property:
- If `images` collection exists and has a default image, return `self.images[self.images.default_image]`.
- Otherwise, fall back to old behavior (open `morphology.ome.tif` at root).

### 5. `from_xenium` Changes (in `io.py`)

The image ingestion section (step 4) is replaced:

```python
# ---- 4. Images + experiment.xenium -----------------------------------
specs = _parse_xenium_specs(src)
_ingest_xenium_images(src, dst, specs, copy_image)
```

**`_parse_xenium_specs(src: Path) -> dict`:**
- Reads `experiment.xenium` from `src`.
- Returns the parsed JSON dict.
- Returns `{}` if file not found (graceful fallback).

**`_ingest_xenium_images(src, dst, specs, copy)`:**
- Creates `dst / "images"` directory.
- Discovers image files:
  1. From `specs["images"]` dict — get `morphology_filepath` and `morphology_focus_filepath`.
  2. Glob `src / "morphology_focus" / "*.ome.tif"` for all focus channel files.
  3. Check `src / "morphology_mip.ome.tif"` (v1 only).
- Symlinks/copies each file into `dst / "images/"` with its basename.
- Writes `images_meta.json` with:
  - `pixel_size` from specs (default 0.2125 if not in specs)
  - `default_image`: `"morphology_focus_0000"` if focus files exist, else `"morphology"` if Z-stack exists
  - `files`: dict mapping name → filename
  - `xenium_version`: parsed from `analysis_sw_version`
- Also symlinks the default focus image to `dst / "morphology.ome.tif"` for backward compatibility.

### 6. `from_merscope` Changes (in `io.py`)

Similar but simpler — MERSCOPE has a single DAPI mosaic image:
- Move image into `dst / "images/"`.
- Write `images_meta.json` with `pixel_size=1.0` (MERSCOPE pixel size varies, read from metadata if available) and single file entry.
- Keep backward-compat symlink.

---

## Design Decisions

1. **`images/` directory** — all image files in one place, not scattered at root. Cleaner for multi-file datasets.
2. **`images_meta.json`** — decouples pixel size from TIFF metadata. Xenium's `experiment.xenium` is the authoritative source, not the TIFF's XResolution tag.
3. **Lazy loading** — `MorphologyImageStore` is only opened when accessed via `__getitem__`. The Z-stack (~10 GB) is never loaded unless explicitly requested.
4. **Level-aware scaling** — `scale_coordinates` and `scale_polygons` make it trivial to overlay cells on downsampled images. The pixel_size × 2^level formula matches Xenium's pyramid convention.
5. **Backward compatibility** — old datasets with just `morphology.ome.tif` at root still work. `sj.morphology` property still works. No migration needed.
6. **Default image** — `morphology_focus_0000` (DAPI) is the most commonly used for visualization. `sj.morphology` returns this.
7. **Separate files** — multi-channel focus images stay as separate files (not merged). Each is its own `MorphologyImageStore` with independent pyramid access.

---

## Testing Strategy

- **`ImageCollection` tests:** lazy loading (store not opened until accessed), `keys()`, `has()`, `pixel_size`, `pixel_size_at`, `scale_coordinates`, `scale_polygons`, `close()`, backward compat with root-level `morphology.ome.tif`.
- **`MorphologyImageStore` extensions:** `pixel_size_at`, `px_to_um` with known pyramid values.
- **`from_xenium` integration:** create mock Xenium directory with `experiment.xenium` + focus files, verify `images_meta.json` written, verify all files symlinked, verify `sj.images` works.
- **Backward compat test:** old dataset without `images/` dir — `sj.morphology` still works, `sj.images` gracefully wraps single file.
- **Coordinate scaling test:** known µm coordinates → expected pixel values at each level.
- Use `tifffile` to write small synthetic OME-TIFFs for fixtures (same pattern as existing `test_images.py`).
