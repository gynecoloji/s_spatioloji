"""Imputation functions for denoising expression data.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from s_spatioloji.compute import (
    _atomic_write_parquet,
    _atomic_write_zarr,
    _load_dense,
    _load_hvg_genes,
)

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def magic(
    sj: s_spatioloji,
    knn: int = 5,
    input_key: str = "X_log1p",
    hvg_key: str = "hvg",
    output_key: str = "X_magic",
    save_expression: bool = True,
    conda_env: str | None = None,
    timeout: int = 7200,
    force: bool = True,
) -> str:
    """MAGIC imputation.

    Runs in-process or via conda bridge.  Writes HVG-subset Parquet and
    optionally full-expression Zarr.

    Args:
        sj: Dataset instance.
        knn: Number of nearest neighbours for MAGIC.
        input_key: Key to read input from.
        hvg_key: Key for the HVG table.
        output_key: Key to write HVG-subset result under.
        save_expression: If ``True``, also write full result to
            ``expression_magic.zarr/``.
        conda_env: Conda environment name, or ``None`` for in-process.
        timeout: Subprocess timeout in seconds.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Raises:
        ImportError: If ``magic-impute`` is not installed (in-process mode).

    Example:
        >>> magic(sj, knn=5)
        'X_magic'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, gene_names = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float32)
    hvg_genes = _load_hvg_genes(sj, hvg_key)

    if conda_env is not None:
        imputed = _run_via_conda(
            "magic", matrix, cell_ids, gene_names, conda_env, timeout,
            knn=knn, hvg_genes=hvg_genes,
        )
    else:
        imputed = _magic_in_process(matrix, gene_names, knn)

    # HVG subset for Parquet
    hvg_idx = [gene_names.index(g) for g in hvg_genes if g in gene_names]
    hvg_names = [gene_names[i] for i in hvg_idx]
    hvg_subset = imputed[:, hvg_idx]

    df = pd.DataFrame(hvg_subset, columns=hvg_names)
    df.insert(0, "cell_id", cell_ids)
    _atomic_write_parquet(df, maps_dir, output_key)

    if save_expression:
        zarr_path = sj.config.root / "expression_magic.zarr"
        _atomic_write_zarr(imputed, cell_ids, gene_names, zarr_path, sj.config.chunks)

    return output_key


def _magic_in_process(matrix, gene_names, knn):
    """Run MAGIC in the current Python process."""
    try:
        import magic as magic_lib
    except ImportError:
        raise ImportError("Install with: pip install s_spatioloji[imputation]") from None

    operator = magic_lib.MAGIC(knn=knn)
    imputed = operator.fit_transform(pd.DataFrame(matrix, columns=gene_names))
    return imputed.values.astype(np.float32)


def alra(
    sj: s_spatioloji,
    input_key: str = "X_log1p",
    hvg_key: str = "hvg",
    output_key: str = "X_alra",
    save_expression: bool = True,
    conda_env: str | None = None,
    timeout: int = 7200,
    force: bool = True,
) -> str:
    """ALRA imputation (via rpy2 + R).

    ALRA is an R package; this function uses ``rpy2`` to call it.
    A conda bridge is recommended.

    Args:
        sj: Dataset instance.
        input_key: Key to read input from.
        hvg_key: Key for the HVG table.
        output_key: Key to write HVG-subset result under.
        save_expression: If ``True``, also write full result to
            ``expression_alra.zarr/``.
        conda_env: Conda environment name, or ``None`` for in-process.
        timeout: Subprocess timeout in seconds.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Raises:
        ImportError: If ``rpy2`` is not installed (in-process mode).

    Example:
        >>> alra(sj, conda_env="r_env")
        'X_alra'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, gene_names = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float32)
    hvg_genes = _load_hvg_genes(sj, hvg_key)

    if conda_env is not None:
        imputed = _run_via_conda(
            "alra", matrix, cell_ids, gene_names, conda_env, timeout,
            hvg_genes=hvg_genes,
        )
    else:
        imputed = _alra_in_process(matrix, gene_names)

    hvg_idx = [gene_names.index(g) for g in hvg_genes if g in gene_names]
    hvg_names = [gene_names[i] for i in hvg_idx]
    hvg_subset = imputed[:, hvg_idx]

    df = pd.DataFrame(hvg_subset, columns=hvg_names)
    df.insert(0, "cell_id", cell_ids)
    _atomic_write_parquet(df, maps_dir, output_key)

    if save_expression:
        zarr_path = sj.config.root / "expression_alra.zarr"
        _atomic_write_zarr(imputed, cell_ids, gene_names, zarr_path, sj.config.chunks)

    return output_key


def _alra_in_process(matrix, gene_names):
    """Run ALRA via rpy2 in the current Python process."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
    except ImportError:
        raise ImportError("Install with: pip install s_spatioloji[imputation]") from None

    numpy2ri.activate()
    ro.r("library(ALRA)")
    ro.globalenv["expr_matrix"] = matrix
    result = ro.r("alra(expr_matrix)[[1]]")
    numpy2ri.deactivate()
    return np.array(result, dtype=np.float32)


def knn_smooth(
    sj: s_spatioloji,
    k: int = 15,
    input_key: str = "X_log1p",
    hvg_key: str = "hvg",
    output_key: str = "X_knnsmooth",
    force: bool = True,
) -> str:
    """KNN-based expression smoothing (core deps only, no conda bridge).

    For each cell, replaces its expression with the mean of its *k*
    nearest neighbours.  Output is HVG-subset only (Parquet, no Zarr).

    Args:
        sj: Dataset instance.
        k: Number of nearest neighbours.
        input_key: Key to read input from.
        hvg_key: Key for the HVG table.
        output_key: Key to write smoothed HVG-subset under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> knn_smooth(sj, k=15)
        'X_knnsmooth'
    """
    from sklearn.neighbors import NearestNeighbors

    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, gene_names = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float32)

    # HVG subset
    hvg_genes = _load_hvg_genes(sj, hvg_key)
    hvg_idx = [gene_names.index(g) for g in hvg_genes if g in gene_names]
    hvg_names = [gene_names[i] for i in hvg_idx]
    hvg_matrix = matrix[:, hvg_idx]

    n_cells = hvg_matrix.shape[0]
    k_clamped = min(k, n_cells - 1)

    nn = NearestNeighbors(n_neighbors=k_clamped)
    nn.fit(hvg_matrix)
    _, indices = nn.kneighbors()

    # Mean of k neighbours for each cell
    smoothed = np.zeros_like(hvg_matrix)
    for i in range(n_cells):
        smoothed[i] = hvg_matrix[indices[i]].mean(axis=0)

    df = pd.DataFrame(smoothed, columns=hvg_names)
    df.insert(0, "cell_id", cell_ids)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def scvi_impute(
    sj: s_spatioloji,
    batch_key: str | None = None,
    input_key: str = "expression",
    n_latent: int = 30,
    n_epochs: int = 400,
    output_key: str = "X_scvi_latent",
    save_expression: bool = True,
    conda_env: str | None = None,
    timeout: int = 7200,
    force: bool = True,
) -> str:
    """scVI imputation: latent embedding + optional denoised expression.

    When ``batch_key=None``, trains an unsupervised scVI model.  Shares
    cached model with :func:`scvi_batch` when parameters match.

    Args:
        sj: Dataset instance.
        batch_key: Batch column, or ``None`` for unsupervised.
        input_key: Key to read raw counts from.
        n_latent: scVI latent dimension.
        n_epochs: Training epochs.
        output_key: Key for the latent embedding Parquet.
        save_expression: If ``True``, also write denoised expression to
            ``expression_scvi.zarr/``.
        conda_env: Conda environment name, or ``None`` for in-process.
        timeout: Subprocess timeout in seconds.
        force: If ``False``, skip if ``maps/<output_key>.parquet`` exists.

    Returns:
        The *output_key* string.

    Example:
        >>> scvi_impute(sj)
        'X_scvi_latent'
    """
    # Delegate to batch_correction.scvi_batch with the same interface
    from s_spatioloji.compute.batch_correction import scvi_batch

    return scvi_batch(
        sj,
        batch_key=batch_key,
        input_key=input_key,
        n_latent=n_latent,
        n_epochs=n_epochs,
        output_key=output_key,
        save_expression=save_expression,
        conda_env=conda_env,
        timeout=timeout,
        force=force,
    )


def _run_via_conda(fn_name, matrix, cell_ids, gene_names, conda_env, timeout, **extra_kwargs):
    """Run an imputation function via conda bridge subprocess."""
    from s_spatioloji.compute._scvi import _validate_conda_env

    _validate_conda_env(conda_env)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        np.savez(tmpdir / "input.npz", X=matrix)

        kwargs = {
            "fn": fn_name,
            "gene_names": gene_names,
            "cell_ids": cell_ids,
            **{k: v for k, v in extra_kwargs.items() if v is not None},
        }
        (tmpdir / "kwargs.json").write_text(json.dumps(kwargs))

        output_path = tmpdir / "output.npz"
        cmd = [
            "conda", "run", "-n", conda_env,
            "python", "-m", "s_spatioloji.compute._runner",
            "--fn", fn_name,
            "--input", str(tmpdir / "input.npz"),
            "--output", str(output_path),
            "--kwargs-file", str(tmpdir / "kwargs.json"),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"{fn_name} failed in conda env '{conda_env}':\n{result.stderr}"
            )

        with np.load(output_path) as data:
            return data["X"].astype(np.float32)
