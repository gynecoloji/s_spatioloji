"""Compute layer: single-cell analysis operations on s_spatioloji.

All compute functions accept an ``s_spatioloji`` object, write results to the
``maps/`` directory under the dataset root, and return the output key string.
Two result formats are used:

- **Parquet** for embeddings, labels, and HVG-subset matrices
- **Zarr** (via :class:`ExpressionStore`) for full expression matrices

All writes use a temp-then-rename pattern for atomicity.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from s_spatioloji.data.config import ChunkConfig
    from s_spatioloji.data.core import s_spatioloji


def _load_dense(sj: s_spatioloji, key: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Load expression data as a dense numpy array.

    Args:
        sj: s_spatioloji instance.
        key: ``"expression"`` for the raw Zarr store, or a ``maps/`` key
            for a Parquet result.

    Returns:
        Tuple of ``(matrix, cell_ids, gene_names)`` where *matrix* has
        shape ``(n_cells, n_features)``.
    """
    if key == "expression":
        matrix = sj.expression.to_dask().compute()
        cell_ids = list(sj.cells.df.compute()["cell_id"])
        names = sj.expression.gene_names
        gene_names = list(names) if names is not None else [f"gene_{i}" for i in range(sj.n_genes)]
        return matrix, cell_ids, gene_names

    df = sj.maps[key].compute()
    cell_ids = list(df["cell_id"])
    gene_names = [c for c in df.columns if c != "cell_id"]
    matrix = df[gene_names].values
    return matrix, cell_ids, gene_names


def _atomic_write_parquet(df: pd.DataFrame, maps_dir: Path, key: str) -> None:
    """Write *df* to ``maps/<key>.parquet`` atomically (temp-then-rename).

    Args:
        df: DataFrame to write.  Must include ``cell_id`` as first column.
        maps_dir: Path to the ``maps/`` directory.
        key: Bare key name (no extension).
    """
    tmp_path = maps_dir / f".{key}.tmp.parquet"
    final_path = maps_dir / f"{key}.parquet"
    df.to_parquet(str(tmp_path), engine="pyarrow", index=False)
    tmp_path.replace(final_path)


def _load_hvg_genes(sj: s_spatioloji, hvg_key: str) -> list[str]:
    """Load the list of highly-variable gene names from a maps/ key.

    Args:
        sj: s_spatioloji instance.
        hvg_key: Key pointing to the HVG table (e.g. ``"hvg"``).

    Returns:
        List of gene name strings marked as highly variable.
    """
    df = sj.maps[hvg_key].compute()
    return df.query("highly_variable")["gene"].tolist()


def _atomic_write_zarr(
    matrix: np.ndarray,
    cell_ids: list[str],
    gene_names: list[str],
    zarr_path: Path,
    chunk_config: ChunkConfig,
) -> None:
    """Write a full expression matrix to Zarr atomically (temp-then-rename).

    Args:
        matrix: Dense array of shape ``(n_cells, n_genes)``.
        cell_ids: Cell ID strings aligned to rows.
        gene_names: Gene name strings aligned to columns.
        zarr_path: Final destination path (e.g. ``<root>/expression_scvi.zarr``).
        chunk_config: Chunk shape settings for the Zarr store.
    """
    import shutil

    from s_spatioloji.data.expression import ExpressionStore

    tmp_path = zarr_path.parent / f".{zarr_path.stem}.tmp.zarr"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)

    store = ExpressionStore.create(
        tmp_path,
        n_cells=len(cell_ids),
        n_genes=len(gene_names),
        chunk_config=chunk_config,
        dtype="float32",
    )
    store.write_chunk(0, matrix.astype(np.float32))
    store.gene_names = gene_names
    store.cell_ids = cell_ids

    if zarr_path.exists():
        shutil.rmtree(zarr_path)
    tmp_path.rename(zarr_path)  # safe after rmtree
