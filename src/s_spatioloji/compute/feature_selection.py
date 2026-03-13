"""Feature selection functions for identifying highly variable genes.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from s_spatioloji.compute import _atomic_write_parquet, _load_dense

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def highly_variable_genes(
    sj: s_spatioloji,
    n_top: int = 2000,
    input_key: str = "X_log1p",
    output_key: str = "hvg",
    force: bool = True,
) -> str:
    """Identify highly variable genes by dispersion ranking.

    Computes mean, variance, and dispersion (variance / mean) per gene,
    then selects the top *n_top* genes by dispersion.

    Args:
        sj: Dataset instance.
        n_top: Number of top genes to mark as highly variable.
        input_key: Key to read log-normalized expression from.
        output_key: Key to write the HVG table under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> highly_variable_genes(sj, n_top=2000)
        'hvg'
    """
    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, _, gene_names = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float32)

    means = matrix.mean(axis=0)
    variances = matrix.var(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        dispersions = np.where(means > 0, variances / means, 0.0)

    n_select = min(n_top, len(gene_names))
    top_idx = np.argsort(dispersions)[::-1][:n_select]
    is_hvg = np.zeros(len(gene_names), dtype=bool)
    is_hvg[top_idx] = True

    df = pd.DataFrame(
        {
            "gene": gene_names,
            "highly_variable": is_hvg,
            "mean": means.astype(np.float32),
            "variance": variances.astype(np.float32),
            "dispersion": dispersions.astype(np.float32),
        }
    )
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
