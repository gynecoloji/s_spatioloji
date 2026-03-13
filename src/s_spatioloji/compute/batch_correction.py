"""Batch correction functions.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.
"""

from __future__ import annotations

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


def harmony(
    sj: s_spatioloji,
    batch_key: str = "fov_id",
    input_key: str = "X_pca",
    output_key: str = "X_pca_harmony",
    force: bool = True,
) -> str:
    """Harmony batch correction on a PCA embedding.

    Args:
        sj: Dataset instance.
        batch_key: Column in ``cells.parquet`` containing batch labels.
        input_key: Key to read PCA embedding from.
        output_key: Key to write corrected embedding under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Raises:
        ImportError: If ``harmonypy`` is not installed.

    Example:
        >>> harmony(sj, batch_key="fov_id")
        'X_pca_harmony'
    """
    try:
        import harmonypy
    except ImportError:
        raise ImportError("Install with: pip install s_spatioloji[batch]") from None

    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, gene_names = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float32)

    cells_df = sj.cells.df.compute()
    meta = pd.DataFrame({batch_key: cells_df[batch_key].values})
    ho = harmonypy.run_harmony(matrix, meta, batch_key)
    corrected = ho.Z_corr.T  # (n_components, n_cells) → (n_cells, n_components)

    df = pd.DataFrame(corrected, columns=gene_names)
    df.insert(0, "cell_id", cell_ids)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def combat(
    sj: s_spatioloji,
    batch_key: str = "fov_id",
    input_key: str = "X_log1p",
    hvg_key: str = "hvg",
    output_key: str = "X_combat",
    save_expression: bool = True,
    force: bool = True,
) -> str:
    """ComBat batch correction (in-process, no conda bridge).

    Operates on the full input matrix.  Writes an HVG-subset Parquet
    and optionally a full-expression Zarr.

    Args:
        sj: Dataset instance.
        batch_key: Column in ``cells.parquet`` containing batch labels.
        input_key: Key to read log-normalized expression from.
        hvg_key: Key for the HVG table.
        output_key: Key to write corrected HVG-subset under.
        save_expression: If ``True``, also write full corrected matrix
            to ``expression_combat.zarr/``.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Raises:
        ImportError: If ``pycombat`` is not installed.

    Example:
        >>> combat(sj, batch_key="fov_id")
        'X_combat'
    """
    try:
        from combat.pycombat import pycombat
    except ImportError:
        raise ImportError("Install with: pip install s_spatioloji[batch]") from None

    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, gene_names = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float32)

    cells_df = sj.cells.df.compute()
    batch_labels = list(cells_df[batch_key])

    # pycombat expects (n_genes, n_cells) DataFrame
    expr_df = pd.DataFrame(matrix.T, index=gene_names, columns=cell_ids)
    corrected = pycombat(expr_df, batch_labels)
    corrected_matrix = corrected.values.T  # back to (n_cells, n_genes)

    # HVG subset for Parquet
    hvg_genes = _load_hvg_genes(sj, hvg_key)
    hvg_idx = [gene_names.index(g) for g in hvg_genes if g in gene_names]
    hvg_names = [gene_names[i] for i in hvg_idx]
    hvg_subset = corrected_matrix[:, hvg_idx]

    df = pd.DataFrame(hvg_subset, columns=hvg_names)
    df.insert(0, "cell_id", cell_ids)
    _atomic_write_parquet(df, maps_dir, output_key)

    if save_expression:
        zarr_path = sj.config.root / "expression_combat.zarr"
        _atomic_write_zarr(corrected_matrix, cell_ids, gene_names, zarr_path, sj.config.chunks)

    return output_key


def regress_out(
    sj: s_spatioloji,
    keys: list[str] | None = None,
    input_key: str = "X_log1p",
    hvg_key: str = "hvg",
    output_key: str = "X_regressed",
    force: bool = True,
) -> str:
    """Regress out confounding variables from expression.

    Fits a linear model per gene with the specified cell metadata columns
    as covariates and returns the residuals.  Output is HVG-subset only.

    Args:
        sj: Dataset instance.
        keys: Cell metadata columns to regress out.  Defaults to
            ``["transcript_counts"]``.
        input_key: Key to read expression from.
        hvg_key: Key for the HVG table.
        output_key: Key to write residuals under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> regress_out(sj, keys=["transcript_counts"])
        'X_regressed'
    """
    if keys is None:
        keys = ["transcript_counts"]

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
    matrix = matrix[:, hvg_idx]

    # Build design matrix from cell metadata
    cells_df = sj.cells.df.compute()
    design = np.column_stack([cells_df[k].values.astype(np.float64) for k in keys])
    design = np.column_stack([np.ones(len(cell_ids)), design])  # intercept

    # OLS per gene: residuals = Y - X @ (X^T X)^-1 X^T Y
    XtX_inv = np.linalg.pinv(design.T @ design)
    hat = design @ XtX_inv @ design.T
    residuals = matrix - (hat @ matrix.astype(np.float64)).astype(np.float32)

    df = pd.DataFrame(residuals, columns=hvg_names)
    df.insert(0, "cell_id", cell_ids)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def scvi_batch(
    sj: s_spatioloji,
    batch_key: str = "fov_id",
    input_key: str = "expression",
    n_latent: int = 30,
    n_epochs: int = 400,
    output_key: str = "X_scvi_latent",
    save_expression: bool = True,
    conda_env: str | None = None,
    timeout: int = 7200,
    force: bool = True,
) -> str:
    """Batch-conditioned scVI: latent embedding + optional denoised expression.

    Delegates model training to :func:`_scvi_train` and shares the cached
    model with :func:`scvi_impute` when parameters match.

    Args:
        sj: Dataset instance.
        batch_key: Column in ``cells.parquet`` with batch labels.
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
        >>> scvi_batch(sj, batch_key="fov_id")
        'X_scvi_latent'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"

    if not force and out_path.exists():
        # Still write secondary output if missing
        if save_expression:
            zarr_path = sj.config.root / "expression_scvi.zarr"
            if not zarr_path.exists():
                _write_scvi_expression(sj, maps_dir, input_key, batch_key, n_latent, n_epochs, conda_env, timeout)
        return output_key

    maps_dir.mkdir(exist_ok=True)

    from s_spatioloji.compute._scvi import _scvi_train

    model_dir = _scvi_train(sj, input_key, batch_key, n_latent, n_epochs, conda_env, timeout, force)
    _write_scvi_outputs(sj, model_dir, input_key, batch_key, output_key, save_expression, maps_dir)

    return output_key


def _write_scvi_outputs(sj, model_dir, input_key, batch_key, output_key, save_expression, maps_dir):
    """Extract latent embedding and optionally denoised expression from trained model."""
    try:
        import anndata
        import scvi
    except ImportError:
        raise ImportError("Install with: pip install s_spatioloji[imputation]") from None

    from s_spatioloji.compute import _load_dense

    matrix, cell_ids, gene_names = _load_dense(sj, input_key)

    obs = pd.DataFrame({"cell_id": cell_ids})
    if batch_key is not None:
        cells_df = sj.cells.df.compute()
        obs[batch_key] = cells_df[batch_key].values

    adata = anndata.AnnData(
        X=matrix.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=gene_names),
    )

    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
    model = scvi.model.SCVI.load(str(model_dir), adata=adata)

    # Latent embedding
    latent = model.get_latent_representation()
    n_latent = latent.shape[1]
    lat_cols = [f"scvi_{i + 1}" for i in range(n_latent)]
    df = pd.DataFrame(latent, columns=lat_cols)
    df.insert(0, "cell_id", cell_ids)
    _atomic_write_parquet(df, maps_dir, output_key)

    # Denoised full expression
    if save_expression:
        denoised = model.get_normalized_expression(library_size=1e4)
        zarr_path = sj.config.root / "expression_scvi.zarr"
        _atomic_write_zarr(denoised.values, cell_ids, gene_names, zarr_path, sj.config.chunks)


def _write_scvi_expression(sj, maps_dir, input_key, batch_key, n_latent, n_epochs, conda_env, timeout):
    """Write denoised expression from existing cached model."""
    from s_spatioloji.compute._scvi import _scvi_train

    model_dir = _scvi_train(sj, input_key, batch_key, n_latent, n_epochs, conda_env, timeout, force=False)
    _write_scvi_outputs(sj, model_dir, input_key, batch_key, "X_scvi_latent", True, maps_dir)
