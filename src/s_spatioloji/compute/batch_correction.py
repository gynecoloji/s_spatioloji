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

    Scalable to 10M+ cells — operates on the PCA embedding (typically
    50 components), not the full expression matrix.

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
