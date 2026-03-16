"""Normalization functions for expression data.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.

Large datasets are processed in chunks to keep RAM usage bounded (~2-4 GB
regardless of dataset size).
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from s_spatioloji.compute import _atomic_write_parquet, _load_dense, _load_hvg_genes
from s_spatioloji.data.expression import ExpressionStore

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji

# Default chunk size: number of cells per iteration
_CHUNK_CELLS = 50_000


def _iter_expression_chunks(
    sj: s_spatioloji,
    input_key: str,
    chunk_size: int = _CHUNK_CELLS,
):
    """Yield (start, end, chunk_array) tuples over expression data.

    Args:
        sj: Dataset instance.
        input_key: ``"expression"`` for Zarr, or a maps/ key for Parquet.
        chunk_size: Number of cells per chunk.

    Yields:
        Tuple of (start_idx, end_idx, numpy_array).
    """
    if input_key == "expression":
        n_cells = sj.expression.n_cells
        dask_arr = sj.expression.to_dask()
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            chunk = dask_arr[start:end].compute()
            yield start, end, chunk
    else:
        # Zarr-backed maps result
        maps_path = sj.config.root / "maps" / f"{input_key}.zarr"
        if maps_path.exists():
            store = ExpressionStore.open(maps_path, sj.config.chunks)
            n_cells = store.n_cells
            dask_arr = store.to_dask()
            for start in range(0, n_cells, chunk_size):
                end = min(start + chunk_size, n_cells)
                chunk = dask_arr[start:end].compute()
                yield start, end, chunk
        else:
            # Parquet fallback: load all at once (small matrix, e.g. HVG-subset)
            matrix, _, _ = _load_dense(sj, input_key)
            yield 0, matrix.shape[0], matrix


def _get_gene_names(sj: s_spatioloji, input_key: str) -> list[str]:
    """Get gene names for an expression key.

    Args:
        sj: Dataset instance.
        input_key: ``"expression"`` or a maps/ key.

    Returns:
        List of gene name strings.
    """
    if input_key == "expression":
        names = sj.expression.gene_names
        return list(names) if names is not None else [f"gene_{i}" for i in range(sj.n_genes)]

    maps_path = sj.config.root / "maps" / f"{input_key}.zarr"
    if maps_path.exists():
        store = ExpressionStore.open(maps_path, sj.config.chunks)
        names = store.gene_names
        return list(names) if names is not None else [f"gene_{i}" for i in range(store.n_genes)]

    # Parquet fallback
    df = sj.maps[input_key].compute()
    return [c for c in df.columns if c != "cell_id"]


def _get_cell_ids(sj: s_spatioloji, input_key: str) -> list[str]:
    """Get cell IDs for an expression key.

    Args:
        sj: Dataset instance.
        input_key: ``"expression"`` or a maps/ key.

    Returns:
        List of cell ID strings.
    """
    if input_key == "expression":
        ids = sj.expression.cell_ids
        if ids is not None:
            return list(ids)
        return list(sj.cells.df.compute()["cell_id"])

    maps_path = sj.config.root / "maps" / f"{input_key}.zarr"
    if maps_path.exists():
        store = ExpressionStore.open(maps_path, sj.config.chunks)
        ids = store.cell_ids
        if ids is not None:
            return list(ids)
        return list(sj.cells.df.compute()["cell_id"])

    df = sj.maps[input_key].compute()
    return list(df["cell_id"])


def _get_n_cells(sj: s_spatioloji, input_key: str) -> int:
    """Get number of cells for an expression key."""
    if input_key == "expression":
        return sj.expression.n_cells
    maps_path = sj.config.root / "maps" / f"{input_key}.zarr"
    if maps_path.exists():
        return ExpressionStore.open(maps_path, sj.config.chunks).n_cells
    return len(sj.maps[input_key].compute())


def _get_n_genes(sj: s_spatioloji, input_key: str) -> int:
    """Get number of genes/features for an expression key."""
    if input_key == "expression":
        return sj.expression.n_genes
    maps_path = sj.config.root / "maps" / f"{input_key}.zarr"
    if maps_path.exists():
        return ExpressionStore.open(maps_path, sj.config.chunks).n_genes
    df = sj.maps[input_key].compute()
    return len([c for c in df.columns if c != "cell_id"])


def _create_output_zarr(
    sj: s_spatioloji,
    output_key: str,
    n_cells: int,
    n_genes: int,
    gene_names: list[str],
    cell_ids: list[str],
) -> ExpressionStore:
    """Create an output Zarr store for chunked writing.

    Args:
        sj: Dataset instance.
        output_key: Output key name.
        n_cells: Number of cells.
        n_genes: Number of genes.
        gene_names: Gene name strings.
        cell_ids: Cell ID strings.

    Returns:
        A new ExpressionStore ready for write_chunk calls.
    """
    maps_dir = sj.config.root / "maps"
    maps_dir.mkdir(exist_ok=True)
    zarr_path = maps_dir / f"{output_key}.zarr"

    # Clean up if exists
    if zarr_path.exists():
        shutil.rmtree(zarr_path)

    store = ExpressionStore.create(
        zarr_path,
        n_cells=n_cells,
        n_genes=n_genes,
        chunk_config=sj.config.chunks,
        compression=sj.config.compression,
        dtype="float32",
    )
    store.gene_names = gene_names
    store.cell_ids = cell_ids
    return store


def normalize_total(
    sj: s_spatioloji,
    target_sum: float = 1e4,
    input_key: str = "expression",
    output_key: str = "X_norm",
    force: bool = True,
    chunk_size: int = _CHUNK_CELLS,
) -> str:
    """Normalize each cell to a fixed total count.

    Divides each cell's counts by its total and multiplies by *target_sum*.
    Processes in chunks to keep RAM bounded.

    Args:
        sj: Dataset instance.
        target_sum: Target total count per cell after normalization.
        input_key: Key to read input from (``"expression"`` or a maps/ key).
        output_key: Key to write the result under.
        force: If ``False``, skip if output already exists.
        chunk_size: Number of cells per processing chunk. Lower = less RAM.

    Returns:
        The *output_key* string.

    Example:
        >>> normalize_total(sj, target_sum=1e4)
        'X_norm'
        >>> normalize_total(sj, chunk_size=10_000)  # use less RAM
        'X_norm'
    """
    maps_dir = sj.config.root / "maps"
    out_zarr = maps_dir / f"{output_key}.zarr"
    out_pq = maps_dir / f"{output_key}.parquet"
    if not force and (out_zarr.exists() or out_pq.exists()):
        return output_key

    n_cells = _get_n_cells(sj, input_key)
    n_genes = _get_n_genes(sj, input_key)
    gene_names = _get_gene_names(sj, input_key)
    cell_ids = _get_cell_ids(sj, input_key)

    out_store = _create_output_zarr(sj, output_key, n_cells, n_genes, gene_names, cell_ids)

    for start, end, chunk in _iter_expression_chunks(sj, input_key, chunk_size=chunk_size):
        chunk = chunk.astype(np.float32)
        row_sums = chunk.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        normed = (chunk / row_sums) * target_sum
        out_store.write_chunk(start, normed)

    return output_key


def log1p(
    sj: s_spatioloji,
    input_key: str = "X_norm",
    output_key: str = "X_log1p",
    force: bool = True,
    chunk_size: int = _CHUNK_CELLS,
) -> str:
    """Apply natural log(1 + x) transform element-wise.

    Processes in chunks to keep RAM bounded.

    Args:
        sj: Dataset instance.
        input_key: Key to read input from.
        output_key: Key to write the result under.
        force: If ``False``, skip if output already exists.
        chunk_size: Number of cells per processing chunk. Lower = less RAM.

    Returns:
        The *output_key* string.

    Example:
        >>> log1p(sj)
        'X_log1p'
    """
    maps_dir = sj.config.root / "maps"
    out_zarr = maps_dir / f"{output_key}.zarr"
    out_pq = maps_dir / f"{output_key}.parquet"
    if not force and (out_zarr.exists() or out_pq.exists()):
        return output_key

    n_cells = _get_n_cells(sj, input_key)
    n_genes = _get_n_genes(sj, input_key)
    gene_names = _get_gene_names(sj, input_key)
    cell_ids = _get_cell_ids(sj, input_key)

    out_store = _create_output_zarr(sj, output_key, n_cells, n_genes, gene_names, cell_ids)

    for start, end, chunk in _iter_expression_chunks(sj, input_key, chunk_size=chunk_size):
        transformed = np.log1p(chunk.astype(np.float32))
        out_store.write_chunk(start, transformed)

    return output_key


def scale(
    sj: s_spatioloji,
    input_key: str = "X_log1p",
    hvg: bool = True,
    hvg_key: str = "hvg",
    max_value: float = 10.0,
    output_key: str = "X_scaled",
    force: bool = True,
    chunk_size: int = _CHUNK_CELLS,
) -> str:
    """Zero-centre and unit-variance scale, optionally subset to HVGs.

    When ``hvg=True``, only HVG columns are included in the output,
    producing a smaller ``(n_cells, n_hvg)`` matrix written as Parquet.
    When ``hvg=False``, writes a full Zarr.

    Two-pass algorithm for chunked processing:
    pass 1 computes running mean/variance, pass 2 applies scaling.

    Args:
        sj: Dataset instance.
        input_key: Key to read input from.
        hvg: Whether to subset to highly variable genes.
        hvg_key: Key for the HVG table (used when ``hvg=True``).
        max_value: Clip scaled values to ``[-max_value, max_value]``.
        output_key: Key to write the result under.
        force: If ``False``, skip if output already exists.
        chunk_size: Number of cells per processing chunk. Lower = less RAM.

    Returns:
        The *output_key* string.

    Example:
        >>> scale(sj, hvg=True)
        'X_scaled'
    """
    maps_dir = sj.config.root / "maps"
    out_zarr = maps_dir / f"{output_key}.zarr"
    out_pq = maps_dir / f"{output_key}.parquet"
    if not force and (out_zarr.exists() or out_pq.exists()):
        return output_key

    maps_dir.mkdir(exist_ok=True)

    gene_names = _get_gene_names(sj, input_key)
    cell_ids = _get_cell_ids(sj, input_key)
    n_cells = _get_n_cells(sj, input_key)

    # Determine gene subset
    gene_idx = None
    if hvg:
        hvg_genes = _load_hvg_genes(sj, hvg_key)
        gene_idx = [gene_names.index(g) for g in hvg_genes if g in gene_names]
        out_gene_names = [gene_names[i] for i in gene_idx]
    else:
        out_gene_names = gene_names

    n_out_genes = len(out_gene_names)

    # Pass 1: compute mean and variance (Welford's online algorithm)
    mean = np.zeros(n_out_genes, dtype=np.float64)
    m2 = np.zeros(n_out_genes, dtype=np.float64)
    count = 0

    for start, end, chunk in _iter_expression_chunks(sj, input_key, chunk_size=chunk_size):
        chunk = chunk.astype(np.float64)
        if gene_idx is not None:
            chunk = chunk[:, gene_idx]
        n = chunk.shape[0]
        for i in range(n):
            count += 1
            delta = chunk[i] - mean
            mean += delta / count
            delta2 = chunk[i] - mean
            m2 += delta * delta2

    stds = np.sqrt(m2 / max(count - 1, 1)).astype(np.float32)
    means = mean.astype(np.float32)
    stds[stds == 0] = 1

    # Pass 2: scale and write
    if hvg:
        # HVG subset is small enough for Parquet
        chunks_list = []
        cell_id_chunks = []
        cell_offset = 0
        for start, end, chunk in _iter_expression_chunks(sj, input_key, chunk_size=chunk_size):
            chunk = chunk.astype(np.float32)
            if gene_idx is not None:
                chunk = chunk[:, gene_idx]
            scaled = np.clip((chunk - means) / stds, -max_value, max_value)
            n = chunk.shape[0]
            chunk_cell_ids = cell_ids[cell_offset:cell_offset + n]
            cell_offset += n
            chunk_df = pd.DataFrame(scaled, columns=out_gene_names)
            chunk_df.insert(0, "cell_id", chunk_cell_ids)
            chunks_list.append(chunk_df)

        df = pd.concat(chunks_list, ignore_index=True)
        _atomic_write_parquet(df, maps_dir, output_key)
    else:
        # Full matrix: write as Zarr
        out_store = _create_output_zarr(sj, output_key, n_cells, n_out_genes, out_gene_names, cell_ids)
        for start, end, chunk in _iter_expression_chunks(sj, input_key, chunk_size=chunk_size):
            chunk = chunk.astype(np.float32)
            if gene_idx is not None:
                chunk = chunk[:, gene_idx]
            scaled = np.clip((chunk - means) / stds, -max_value, max_value)
            out_store.write_chunk(start, scaled)

    return output_key


def pearson_residuals(
    sj: s_spatioloji,
    theta: float = 100.0,
    input_key: str = "expression",
    output_key: str = "X_residuals",
    force: bool = True,
    chunk_size: int = _CHUNK_CELLS,
) -> str:
    """Compute Pearson residuals for a negative-binomial model.

    Analytically approximates the effect of variance stabilization.
    Residuals are clipped to ``[-sqrt(n_cells), sqrt(n_cells)]``.

    Two-pass chunked algorithm: pass 1 computes global sums for expected
    values, pass 2 computes residuals.

    Args:
        sj: Dataset instance.
        theta: Overdispersion parameter for the NB model.
        input_key: Key to read raw counts from.
        output_key: Key to write the result under.
        force: If ``False``, skip if output already exists.
        chunk_size: Number of cells per processing chunk. Lower = less RAM.

    Returns:
        The *output_key* string.

    Example:
        >>> pearson_residuals(sj)
        'X_residuals'
    """
    maps_dir = sj.config.root / "maps"
    out_zarr = maps_dir / f"{output_key}.zarr"
    out_pq = maps_dir / f"{output_key}.parquet"
    if not force and (out_zarr.exists() or out_pq.exists()):
        return output_key

    n_cells = _get_n_cells(sj, input_key)
    n_genes = _get_n_genes(sj, input_key)
    gene_names = _get_gene_names(sj, input_key)
    cell_ids = _get_cell_ids(sj, input_key)

    # Pass 1: compute per-cell totals (n_i) and per-gene totals (for p_j)
    gene_sums = np.zeros(n_genes, dtype=np.float64)
    cell_totals = np.zeros(n_cells, dtype=np.float64)
    total_sum = 0.0

    for start, end, chunk in _iter_expression_chunks(sj, input_key, chunk_size=chunk_size):
        chunk = chunk.astype(np.float64)
        gene_sums += chunk.sum(axis=0)
        cell_totals[start:end] = chunk.sum(axis=1)
        total_sum += chunk.sum()

    p = gene_sums / max(total_sum, 1.0)  # gene proportions
    clip_val = np.sqrt(n_cells)

    # Pass 2: compute residuals chunk by chunk
    out_store = _create_output_zarr(sj, output_key, n_cells, n_genes, gene_names, cell_ids)

    for start, end, chunk in _iter_expression_chunks(sj, input_key, chunk_size=chunk_size):
        chunk = chunk.astype(np.float64)
        n_i = cell_totals[start:end, np.newaxis]  # (chunk_size, 1)
        mu = n_i * p[np.newaxis, :]  # (chunk_size, n_genes)
        residuals = (chunk - mu) / np.sqrt(mu + mu**2 / theta)
        residuals = np.clip(residuals, -clip_val, clip_val).astype(np.float32)
        out_store.write_chunk(start, residuals)

    return output_key
