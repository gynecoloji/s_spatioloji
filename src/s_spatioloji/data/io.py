"""Loaders that ingest raw platform outputs and write standardised datasets.

Each loader reads the vendor-specific directory layout, normalises column
names to the canonical schema, and writes the full ``s_spatioloji`` dataset
layout under ``output_dir``:

    output_dir/
    ├── expression.zarr/        # cells × genes, uint16 raw counts
    ├── cells.parquet           # canonical cell metadata
    ├── boundaries.parquet      # GeoParquet polygons (if available)
    ├── morphology.ome.tif      # symlinked or copied from source
    └── _index/                 # empty — populated by downstream indexing

Supported platforms
-------------------
- **10x Xenium**   — :func:`from_xenium`
- **Vizgen MERSCOPE** — :func:`from_merscope`

Canonical cell-metadata columns
---------------------------------
Required:
    cell_id (str), x (float, µm), y (float, µm)
Optional (platform-dependent):
    fov_id (int), transcript_counts (int), area (float µm²),
    nucleus_area (float µm²), z_centroid (float µm)
"""

from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import from_wkb
from shapely.geometry import Polygon

from s_spatioloji.data.boundaries import BoundaryStore
from s_spatioloji.data.cells import CellStore
from s_spatioloji.data.config import SpatiolojiConfig
from s_spatioloji.data.core import s_spatioloji
from s_spatioloji.data.expression import ExpressionStore


# ---------------------------------------------------------------------------
# Canonical column maps
# ---------------------------------------------------------------------------

#: Xenium cell CSV/Parquet → canonical name
_XENIUM_CELL_COLS: dict[str, str] = {
    "cell_id": "cell_id",
    "x_centroid": "x",
    "y_centroid": "y",
    "z_centroid": "z_centroid",
    "transcript_counts": "transcript_counts",
    "cell_area": "area",
    "nucleus_area": "nucleus_area",
    "fov_name": "fov_id",
}

#: MERSCOPE cell CSV → canonical name
_MERSCOPE_CELL_COLS: dict[str, str] = {
    "EntityID": "cell_id",
    "center_x": "x",
    "center_y": "y",
    "center_z": "z_centroid",
    "volume": "volume",
    "fov": "fov_id",
    "anisotropy": "anisotropy",
    "min_x": "x_min",
    "max_x": "x_max",
    "min_y": "y_min",
    "max_y": "y_max",
}


# ---------------------------------------------------------------------------
# Xenium experiment.xenium + image helpers
# ---------------------------------------------------------------------------


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
            name = f.stem.replace(".ome", "")
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
    meta_path = dst / "images_meta.json"
    with open(str(meta_path), "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------


def from_xenium(
    xenium_dir: Path | str,
    output_dir: Path | str,
    config: SpatiolojiConfig | None = None,
    use_nucleus_boundaries: bool = False,
    copy_image: bool = False,
) -> s_spatioloji:
    """Load a 10x Xenium output directory and write a standardised dataset.

    Reads the following files from ``xenium_dir`` (all optional except
    cells and expression):

    - ``cells.parquet`` or ``cells.csv.gz`` — cell metadata
    - ``cell_feature_matrix.h5`` (HDF5, preferred) or
      ``cell_feature_matrix/`` directory (10x MTX format with
      ``barcodes.tsv.gz``, ``features.tsv.gz``, ``matrix.mtx.gz``)
    - ``cell_boundaries.parquet`` or ``nucleus_boundaries.parquet``
    - ``morphology_focus/`` directory (multi-channel OME-TIFFs)
    - ``morphology.ome.tif`` (Z-stack)
    - ``experiment.xenium`` (metadata with pixel size)

    Args:
        xenium_dir: Root directory of the Xenium output.
        output_dir: Destination directory for the ``s_spatioloji`` dataset.
            Created if it does not exist.
        config: Optional config override for chunk sizes, compression, etc.
        use_nucleus_boundaries: If ``True``, prefer ``nucleus_boundaries``
            over ``cell_boundaries`` when both are present.
        copy_image: If ``True``, copy ``morphology.ome.tif`` to
            ``output_dir``.  If ``False`` (default), create a symlink so no
            disk space is duplicated.

    Returns:
        An :class:`~s_spatioloji.data.core.s_spatioloji` instance pointing
        at ``output_dir``.

    Raises:
        FileNotFoundError: If ``xenium_dir`` does not exist.
        FileNotFoundError: If neither ``cells.parquet`` nor ``cells.csv.gz``
            is found in ``xenium_dir``.
        FileNotFoundError: If neither ``cell_feature_matrix.h5`` nor
            ``cell_feature_matrix/`` is found.

    Example:
        >>> sj = from_xenium("/data/xenium_run1", "/data/processed/run1")
        >>> sj.n_cells
        4_200_000
    """
    src = Path(xenium_dir)
    dst = Path(output_dir)
    _ensure_dir(src, "xenium_dir")

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "_index").mkdir(exist_ok=True)

    cfg = config or SpatiolojiConfig(root=dst)

    # ---- 1. Cell metadata ------------------------------------------------
    cells_df = _read_xenium_cells(src)
    CellStore.create(cfg.paths.cells, cells_df)

    # ---- 2. Expression matrix --------------------------------------------
    expr_matrix, gene_names, cell_ids = _read_xenium_expression(src)
    n_cells, n_genes = expr_matrix.shape
    store = ExpressionStore.create(
        cfg.paths.expression,
        n_cells=n_cells,
        n_genes=n_genes,
        chunk_config=cfg.chunks,
        compression=cfg.compression,
        dtype="uint16",
    )
    _write_matrix_chunked(store, expr_matrix, cfg.chunks.expression_cells)
    store.gene_names = gene_names
    store.cell_ids = cell_ids

    # ---- 3. Boundaries (optional) ----------------------------------------
    boundary_file = _find_xenium_boundaries(src, use_nucleus_boundaries)
    if boundary_file is not None:
        gdf = _read_xenium_boundaries(boundary_file)
        BoundaryStore.create(cfg.paths.boundaries, gdf)

    # ---- 4. Images + experiment.xenium -----------------------------------
    specs = _parse_xenium_specs(src)
    _ingest_xenium_images(src, dst, specs, copy=copy_image)

    return s_spatioloji.open(dst)


def from_merscope(
    merscope_dir: Path | str,
    output_dir: Path | str,
    config: SpatiolojiConfig | None = None,
    copy_image: bool = False,
) -> s_spatioloji:
    """Load a Vizgen MERSCOPE output directory and write a standardised dataset.

    Reads the following files from ``merscope_dir``:

    - ``cell_metadata.csv`` — cell metadata (required)
    - ``cell_by_gene.csv`` — dense expression matrix, cells × genes (required)
    - ``cell_boundaries/`` — Parquet boundary files (optional)
    - ``images/mosaic_DAPI_z*.tif`` — morphology image (optional, first
      matching file is used)

    Args:
        merscope_dir: Root directory of the MERSCOPE output.
        output_dir: Destination directory for the ``s_spatioloji`` dataset.
        config: Optional config override.
        copy_image: Copy the morphology image instead of symlinking.

    Returns:
        An :class:`~s_spatioloji.data.core.s_spatioloji` instance pointing
        at ``output_dir``.

    Raises:
        FileNotFoundError: If ``merscope_dir``, ``cell_metadata.csv``, or
            ``cell_by_gene.csv`` are not found.

    Example:
        >>> sj = from_merscope("/data/merscope_run1", "/data/processed/run1")
    """
    src = Path(merscope_dir)
    dst = Path(output_dir)
    _ensure_dir(src, "merscope_dir")

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "_index").mkdir(exist_ok=True)

    cfg = config or SpatiolojiConfig(root=dst)

    # ---- 1. Cell metadata ------------------------------------------------
    cells_df = _read_merscope_cells(src)
    CellStore.create(cfg.paths.cells, cells_df)

    # ---- 2. Expression matrix --------------------------------------------
    expr_matrix, gene_names, cell_ids = _read_merscope_expression(src)
    n_cells, n_genes = expr_matrix.shape
    store = ExpressionStore.create(
        cfg.paths.expression,
        n_cells=n_cells,
        n_genes=n_genes,
        chunk_config=cfg.chunks,
        compression=cfg.compression,
        dtype="uint16",
    )
    _write_matrix_chunked(store, expr_matrix, cfg.chunks.expression_cells)
    store.gene_names = gene_names
    store.cell_ids = cell_ids

    # ---- 3. Boundaries (optional) ----------------------------------------
    boundary_dir = src / "cell_boundaries"
    if boundary_dir.exists():
        gdf = _read_merscope_boundaries(boundary_dir)
        if gdf is not None:
            BoundaryStore.create(cfg.paths.boundaries, gdf)

    # ---- 4. Morphology image (optional) ----------------------------------
    morph_src = _find_merscope_image(src)
    if morph_src is not None:
        _link_or_copy(morph_src, cfg.paths.morphology, copy=copy_image)

    return s_spatioloji.open(dst)


# ---------------------------------------------------------------------------
# Xenium helpers
# ---------------------------------------------------------------------------


def _read_xenium_cells(src: Path) -> pd.DataFrame:
    """Read and normalise Xenium cell metadata.

    Args:
        src: Xenium output directory.

    Returns:
        DataFrame with canonical column names.

    Raises:
        FileNotFoundError: If neither ``cells.parquet`` nor ``cells.csv.gz``
            is present.
    """
    parquet = src / "cells.parquet"
    csv_gz = src / "cells.csv.gz"

    if parquet.exists():
        df = pd.read_parquet(str(parquet), engine="pyarrow")
    elif csv_gz.exists():
        df = pd.read_csv(str(csv_gz), compression="gzip")
    else:
        raise FileNotFoundError(
            f"No cell metadata found in {src}. "
            "Expected 'cells.parquet' or 'cells.csv.gz'."
        )

    df = _rename_columns(df, _XENIUM_CELL_COLS)
    _ensure_cell_id_str(df)
    return df


def _find_xenium_boundaries(src: Path, use_nucleus: bool) -> Path | None:
    """Return the boundary Parquet file to use, or None if absent."""
    nucleus = src / "nucleus_boundaries.parquet"
    cell = src / "cell_boundaries.parquet"
    if use_nucleus and nucleus.exists():
        return nucleus
    if cell.exists():
        return cell
    if nucleus.exists():
        return nucleus
    return None


def _read_xenium_boundaries(path: Path) -> gpd.GeoDataFrame:
    """Read a Xenium boundary Parquet file into a GeoDataFrame.

    Xenium stores polygon vertices as repeated rows with columns
    ``cell_id``, ``vertex_x``, ``vertex_y``.  We reconstruct each polygon
    from its vertex list.

    If the file already contains a WKB ``geometry`` column (newer Xenium
    versions), it is used directly.

    Args:
        path: Path to the boundary Parquet file.

    Returns:
        A ``GeoDataFrame`` with ``cell_id`` and ``geometry`` columns.
    """
    df = pd.read_parquet(str(path), engine="pyarrow")

    # Newer Xenium: geometry stored as WKB bytes
    if "geometry" in df.columns:
        df["geometry"] = df["geometry"].apply(
            lambda g: from_wkb(g) if isinstance(g, (bytes, bytearray)) else g
        )
        return gpd.GeoDataFrame(df, geometry="geometry")

    # Older Xenium: vertex_x, vertex_y per row, grouped by cell_id
    required = {"cell_id", "vertex_x", "vertex_y"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Cannot parse boundary file {path}: "
            f"expected columns {required}, got {set(df.columns)}"
        )
    records = []
    for cell_id, group in df.groupby("cell_id", sort=False):
        coords = list(zip(group["vertex_x"], group["vertex_y"]))
        if len(coords) >= 3:
            records.append({"cell_id": str(cell_id), "geometry": Polygon(coords)})
    return gpd.GeoDataFrame(records, geometry="geometry")


# ---------------------------------------------------------------------------
# 10x MTX helpers (shared by Xenium)
# ---------------------------------------------------------------------------


def _read_10x_mtx(
    matrix_dir: Path,
) -> tuple[np.ndarray, list[str], list[str]]:  # type: ignore[type-arg]
    """Read a 10x-format sparse MTX directory into a dense uint16 array.

    Args:
        matrix_dir: Directory containing ``barcodes.tsv.gz``,
            ``features.tsv.gz``, and ``matrix.mtx.gz``.

    Returns:
        Tuple of ``(matrix, gene_names, cell_ids)`` where ``matrix`` has
        shape ``(n_cells, n_genes)`` and dtype ``uint16``.

    Raises:
        FileNotFoundError: If ``matrix_dir`` does not exist or required
            files are missing.
    """
    try:
        from scipy.io import mmread
        from scipy.sparse import csr_matrix
    except ImportError:
        raise ImportError("Install scipy: pip install scipy")

    if not matrix_dir.exists():
        raise FileNotFoundError(f"cell_feature_matrix directory not found: {matrix_dir}")

    barcodes_path = matrix_dir / "barcodes.tsv.gz"
    features_path = matrix_dir / "features.tsv.gz"
    matrix_path = matrix_dir / "matrix.mtx.gz"

    for p in (barcodes_path, features_path, matrix_path):
        if not p.exists():
            raise FileNotFoundError(f"Required MTX file not found: {p}")

    # Cell barcodes
    with gzip.open(str(barcodes_path), "rt") as f:
        cell_ids = [line.strip() for line in f]

    # Gene names (second column of features file)
    with gzip.open(str(features_path), "rt") as f:
        gene_names = [line.strip().split("\t")[1] for line in f]

    # Sparse matrix — MTX is (genes × cells), transpose to (cells × genes)
    with gzip.open(str(matrix_path), "rb") as f:
        mat = mmread(f)
    mat_csr: csr_matrix = csr_matrix(mat).T  # → (n_cells, n_genes)
    matrix = mat_csr.toarray().astype(np.uint16)

    return matrix, gene_names, cell_ids


def _read_10x_h5(
    h5_path: Path,
) -> tuple[np.ndarray, list[str], list[str]]:  # type: ignore[type-arg]
    """Read a 10x-format HDF5 expression matrix into a dense uint16 array.

    Handles both v3 (``/matrix/``) and v2 (root-level) HDF5 layouts used
    by Xenium and Cell Ranger.

    Args:
        h5_path: Path to ``cell_feature_matrix.h5``.

    Returns:
        Tuple of ``(matrix, gene_names, cell_ids)`` where ``matrix`` has
        shape ``(n_cells, n_genes)`` and dtype ``uint16``.

    Raises:
        FileNotFoundError: If ``h5_path`` does not exist.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("Install h5py: pip install h5py")

    try:
        from scipy.sparse import csc_matrix
    except ImportError:
        raise ImportError("Install scipy: pip install scipy")

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(str(h5_path), "r") as f:
        # Determine root group: v3 uses /matrix/, v2 stores at root
        if "matrix" in f:
            grp = f["matrix"]
        else:
            grp = f

        # Read sparse CSC components
        data = grp["data"][:]
        indices = grp["indices"][:]
        indptr = grp["indptr"][:]
        shape = tuple(grp["shape"][:])

        # Barcodes (cell IDs)
        barcodes_raw = grp["barcodes"][:]
        cell_ids = [b.decode("utf-8") if isinstance(b, bytes) else str(b) for b in barcodes_raw]

        # Gene names — try features/name (v3), then gene_names (v2)
        if "features" in grp:
            feat_grp = grp["features"]
            names_raw = feat_grp["name"][:]
        elif "gene_names" in grp:
            names_raw = grp["gene_names"][:]
        else:
            # Fallback: use feature IDs
            names_raw = feat_grp["id"][:] if "features" in grp else np.arange(shape[0]).astype(bytes)

        gene_names = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in names_raw]

    # Build sparse matrix — HDF5 stores as (genes × cells) CSC
    mat = csc_matrix((data, indices, indptr), shape=shape)
    # Transpose to (cells × genes)
    matrix = mat.T.toarray().astype(np.uint16)

    return matrix, gene_names, cell_ids


def _read_xenium_expression(
    src: Path,
) -> tuple[np.ndarray, list[str], list[str]]:  # type: ignore[type-arg]
    """Read Xenium expression matrix, trying H5 first then MTX directory.

    Args:
        src: Xenium output directory.

    Returns:
        Tuple of ``(matrix, gene_names, cell_ids)`` with dtype ``uint16``.

    Raises:
        FileNotFoundError: If neither ``cell_feature_matrix.h5`` nor
            ``cell_feature_matrix/`` is found.
    """
    h5_path = src / "cell_feature_matrix.h5"
    mtx_dir = src / "cell_feature_matrix"

    if h5_path.exists():
        return _read_10x_h5(h5_path)
    elif mtx_dir.exists():
        return _read_10x_mtx(mtx_dir)
    else:
        raise FileNotFoundError(
            f"No expression matrix found in {src}. "
            "Expected 'cell_feature_matrix.h5' or 'cell_feature_matrix/' directory."
        )


# ---------------------------------------------------------------------------
# MERSCOPE helpers
# ---------------------------------------------------------------------------


def _read_merscope_cells(src: Path) -> pd.DataFrame:
    """Read and normalise MERSCOPE cell metadata.

    Args:
        src: MERSCOPE output directory.

    Returns:
        DataFrame with canonical column names.

    Raises:
        FileNotFoundError: If ``cell_metadata.csv`` is not present.
    """
    path = src / "cell_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"cell_metadata.csv not found in {src}")
    df = pd.read_csv(str(path))
    df = _rename_columns(df, _MERSCOPE_CELL_COLS)
    _ensure_cell_id_str(df)
    return df


def _read_merscope_expression(
    src: Path,
) -> tuple[np.ndarray, list[str], list[str]]:  # type: ignore[type-arg]
    """Read the MERSCOPE ``cell_by_gene.csv`` dense expression matrix.

    The first column is treated as ``cell_id``; remaining columns are genes.

    Args:
        src: MERSCOPE output directory.

    Returns:
        Tuple of ``(matrix, gene_names, cell_ids)`` with dtype ``uint16``.

    Raises:
        FileNotFoundError: If ``cell_by_gene.csv`` is not present.
    """
    path = src / "cell_by_gene.csv"
    if not path.exists():
        raise FileNotFoundError(f"cell_by_gene.csv not found in {src}")
    df = pd.read_csv(str(path), index_col=0)
    cell_ids = [str(c) for c in df.index.tolist()]
    gene_names = df.columns.tolist()
    matrix = df.values.astype(np.uint16)
    return matrix, gene_names, cell_ids


def _read_merscope_boundaries(boundary_dir: Path) -> gpd.GeoDataFrame | None:
    """Read MERSCOPE boundary Parquet files from a directory.

    Concatenates all ``*.parquet`` files found in ``boundary_dir``.

    Args:
        boundary_dir: Directory containing boundary Parquet files.

    Returns:
        A ``GeoDataFrame``, or ``None`` if no Parquet files are found.
    """
    parquet_files = sorted(boundary_dir.glob("*.parquet"))
    if not parquet_files:
        return None
    frames = [pd.read_parquet(str(p), engine="pyarrow") for p in parquet_files]
    df = pd.concat(frames, ignore_index=True)

    # Determine ID column before any path
    id_col = next((c for c in ("EntityID", "cell_id", "ID") if c in df.columns), None)

    if "geometry" in df.columns:
        df["geometry"] = df["geometry"].apply(
            lambda g: from_wkb(g) if isinstance(g, (bytes, bytearray)) else g
        )
        gdf = gpd.GeoDataFrame(df, geometry="geometry")
        if id_col is not None and id_col != "cell_id":
            gdf = gdf.rename(columns={id_col: "cell_id"})
        gdf["cell_id"] = gdf["cell_id"].astype(str)
        return gdf

    # Vertex-format (EntityID, x, y columns)
    if id_col is None:
        return None
    x_col = next((c for c in ("x", "vertex_x", "boundaryX") if c in df.columns), None)
    y_col = next((c for c in ("y", "vertex_y", "boundaryY") if c in df.columns), None)
    if x_col is None or y_col is None:
        return None

    records = []
    for cell_id, group in df.groupby(id_col, sort=False):
        coords = list(zip(group[x_col], group[y_col]))
        if len(coords) >= 3:
            records.append({"cell_id": str(cell_id), "geometry": Polygon(coords)})
    return gpd.GeoDataFrame(records, geometry="geometry") if records else None


def _find_merscope_image(src: Path) -> Path | None:
    """Return the first DAPI mosaic TIFF found, or None."""
    images_dir = src / "images"
    if not images_dir.exists():
        return None
    # Prefer DAPI z0
    candidates = list(images_dir.glob("mosaic_DAPI_z0.tif"))
    if not candidates:
        candidates = list(images_dir.glob("mosaic_DAPI*.tif"))
    if not candidates:
        candidates = list(images_dir.glob("*.tif"))
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _write_matrix_chunked(
    store: ExpressionStore,
    matrix: np.ndarray,  # type: ignore[type-arg]
    chunk_size: int,
) -> None:
    """Write a (n_cells, n_genes) matrix to an ExpressionStore in row chunks.

    Args:
        store: Pre-created :class:`ExpressionStore` with matching shape.
        matrix: Dense numpy array of shape ``(n_cells, n_genes)``.
        chunk_size: Number of rows per write.
    """
    n_cells = matrix.shape[0]
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        store.write_chunk(start, matrix[start:end])


def _rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename columns present in *df* according to *mapping*; ignore missing.

    Args:
        df: Source DataFrame.
        mapping: Dict of ``{source_col: canonical_col}``.

    Returns:
        DataFrame with applicable columns renamed.
    """
    rename = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=rename)


def _ensure_cell_id_str(df: pd.DataFrame) -> None:
    """Cast the ``cell_id`` column to str in-place.

    Args:
        df: DataFrame that must contain a ``cell_id`` column.

    Raises:
        KeyError: If ``cell_id`` is not present.
    """
    if "cell_id" not in df.columns:
        raise KeyError("DataFrame has no 'cell_id' column after column renaming")
    df["cell_id"] = df["cell_id"].astype(str)


def _ensure_dir(path: Path, name: str) -> None:
    """Raise FileNotFoundError if *path* does not exist.

    Args:
        path: Path to check.
        name: Argument name for the error message.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    """Symlink or copy *src* to *dst*.

    Args:
        src: Source file.
        dst: Destination path.
        copy: If ``True``, copy the file; otherwise create a symlink.
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(str(src), str(dst))
    else:
        try:
            dst.symlink_to(src.resolve())
        except (OSError, NotImplementedError):
            # Symlinks may not be available on all Windows configurations
            shutil.copy2(str(src), str(dst))
