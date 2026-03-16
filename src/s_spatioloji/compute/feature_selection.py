"""Feature selection functions for identifying highly variable genes.

All functions follow the compute layer contract: accept an ``s_spatioloji``
object, write results to ``maps/``, and return the output key string.

Uses chunked streaming to compute per-gene statistics without loading the
full expression matrix into memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from s_spatioloji.compute import _atomic_write_parquet
from s_spatioloji.compute.normalize import (
    _get_gene_names,
    _get_n_genes,
    _iter_expression_chunks,
)

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

    Computes mean, variance, and dispersion (variance / mean) per gene
    using a single streaming pass over the data (Welford's algorithm),
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
    gene_names = _get_gene_names(sj, input_key)
    n_genes = _get_n_genes(sj, input_key)

    # Streaming mean and variance (Welford's online algorithm)
    mean = np.zeros(n_genes, dtype=np.float64)
    m2 = np.zeros(n_genes, dtype=np.float64)
    count = 0

    for start, end, chunk in _iter_expression_chunks(sj, input_key):
        chunk = chunk.astype(np.float64)
        for i in range(chunk.shape[0]):
            count += 1
            delta = chunk[i] - mean
            mean += delta / count
            delta2 = chunk[i] - mean
            m2 += delta * delta2

    means = mean.astype(np.float32)
    variances = (m2 / max(count - 1, 1)).astype(np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        dispersions = np.where(means > 0, variances / means, 0.0).astype(np.float32)

    n_select = min(n_top, len(gene_names))
    top_idx = np.argsort(dispersions)[::-1][:n_select]
    is_hvg = np.zeros(len(gene_names), dtype=bool)
    is_hvg[top_idx] = True

    df = pd.DataFrame(
        {
            "gene": gene_names,
            "highly_variable": is_hvg,
            "mean": means,
            "variance": variances,
            "dispersion": dispersions,
        }
    )
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
