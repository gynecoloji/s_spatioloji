"""Normalization functions for expression data.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from s_spatioloji.compute import _atomic_write_parquet, _load_dense, _load_hvg_genes

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def normalize_total(
    sj: s_spatioloji,
    target_sum: float = 1e4,
    input_key: str = "expression",
    output_key: str = "X_norm",
    force: bool = True,
) -> str:
    """Normalize each cell to a fixed total count.

    Divides each cell's counts by its total and multiplies by *target_sum*.

    Args:
        sj: Dataset instance.
        target_sum: Target total count per cell after normalization.
        input_key: Key to read input from (``"expression"`` or a maps/ key).
        output_key: Key to write the result under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> normalize_total(sj, target_sum=1e4)
        'X_norm'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, gene_names = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float32)

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normed = (matrix / row_sums) * target_sum

    df = pd.DataFrame(normed, columns=gene_names)
    df.insert(0, "cell_id", cell_ids)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def log1p(
    sj: s_spatioloji,
    input_key: str = "X_norm",
    output_key: str = "X_log1p",
    force: bool = True,
) -> str:
    """Apply natural log(1 + x) transform element-wise.

    Args:
        sj: Dataset instance.
        input_key: Key to read input from.
        output_key: Key to write the result under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> log1p(sj)
        'X_log1p'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, gene_names = _load_dense(sj, input_key)
    matrix = np.log1p(matrix.astype(np.float32))

    df = pd.DataFrame(matrix, columns=gene_names)
    df.insert(0, "cell_id", cell_ids)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def scale(
    sj: s_spatioloji,
    input_key: str = "X_log1p",
    hvg: bool = True,
    hvg_key: str = "hvg",
    max_value: float = 10.0,
    output_key: str = "X_scaled",
    force: bool = True,
) -> str:
    """Zero-centre and unit-variance scale, optionally subset to HVGs.

    When ``hvg=True``, genes are subset to the HVG list **before**
    densification, producing an ``(n_cells, n_hvg)`` output.

    Args:
        sj: Dataset instance.
        input_key: Key to read input from.
        hvg: Whether to subset to highly variable genes.
        hvg_key: Key for the HVG table (used when ``hvg=True``).
        max_value: Clip scaled values to ``[-max_value, max_value]``.
        output_key: Key to write the result under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> scale(sj, hvg=True)
        'X_scaled'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, gene_names = _load_dense(sj, input_key)

    if hvg:
        hvg_genes = _load_hvg_genes(sj, hvg_key)
        gene_idx = [gene_names.index(g) for g in hvg_genes if g in gene_names]
        matrix = matrix[:, gene_idx]
        gene_names = [gene_names[i] for i in gene_idx]

    matrix = matrix.astype(np.float32)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1
    scaled = (matrix - means) / stds
    scaled = np.clip(scaled, -max_value, max_value)

    df = pd.DataFrame(scaled, columns=gene_names)
    df.insert(0, "cell_id", cell_ids)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def pearson_residuals(
    sj: s_spatioloji,
    theta: float = 100.0,
    input_key: str = "expression",
    output_key: str = "X_residuals",
    force: bool = True,
) -> str:
    """Compute Pearson residuals for a negative-binomial model.

    Analytically approximates the effect of variance stabilization.
    Residuals are clipped to ``[-sqrt(n_cells), sqrt(n_cells)]``.

    Args:
        sj: Dataset instance.
        theta: Overdispersion parameter for the NB model.
        input_key: Key to read raw counts from.
        output_key: Key to write the result under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> pearson_residuals(sj)
        'X_residuals'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, gene_names = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float64)

    n = matrix.sum(axis=1, keepdims=True)
    p = matrix.sum(axis=0, keepdims=True) / matrix.sum()
    mu = n * p
    residuals = (matrix - mu) / np.sqrt(mu + mu**2 / theta)
    clip_val = np.sqrt(matrix.shape[0])
    residuals = np.clip(residuals, -clip_val, clip_val).astype(np.float32)

    df = pd.DataFrame(residuals, columns=gene_names)
    df.insert(0, "cell_id", cell_ids)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
