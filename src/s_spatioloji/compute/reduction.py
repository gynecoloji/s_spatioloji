"""Dimensionality reduction functions.

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


def pca(
    sj: s_spatioloji,
    n_components: int = 50,
    n_subsample: int = 100_000,
    hvg: bool = True,
    hvg_key: str = "hvg",
    input_key: str = "X_scaled",
    output_key: str = "X_pca",
    output_loadings_key: str = "X_pca_loadings",
    force: bool = True,
    chunk_size: int = 50_000,
) -> str:
    """PCA via subsample-fit then chunked project-all.

    Fits PCA on a subsample of cells (if dataset is large) then projects
    all cells into the learned space in chunks.  ``n_components`` is
    automatically clamped to ``min(n_components, n_cells - 1, n_features)``.

    Args:
        sj: Dataset instance.
        n_components: Target number of principal components.
        n_subsample: Maximum cells to use for fitting (larger datasets
            are subsampled).
        hvg: Whether to subset input to HVGs before PCA.
        hvg_key: Key for the HVG table.
        input_key: Key to read input from.
        output_key: Key for the embedding output.
        output_loadings_key: Key for the gene loadings side-output.
        force: If ``False``, skip if **both** outputs already exist.
        chunk_size: Number of cells per processing chunk. Lower = less RAM.

    Returns:
        The *output_key* string.

    Example:
        >>> pca(sj, n_components=50)
        'X_pca'
        >>> pca(sj, chunk_size=10_000)  # use less RAM
        'X_pca'
    """
    from sklearn.decomposition import PCA as _PCA

    from s_spatioloji.compute.normalize import (
        _get_cell_ids,
        _get_gene_names,
        _get_n_cells,
        _get_n_genes,
        _iter_expression_chunks,
    )

    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    loadings_path = maps_dir / f"{output_loadings_key}.parquet"
    if not force and out_path.exists() and loadings_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)

    gene_names = _get_gene_names(sj, input_key)
    cell_ids = _get_cell_ids(sj, input_key)
    n_cells = _get_n_cells(sj, input_key)

    # Determine HVG subset indices
    gene_idx = None
    if hvg:
        hvg_genes = _load_hvg_genes(sj, hvg_key)
        gene_idx = [gene_names.index(g) for g in hvg_genes if g in gene_names]
        gene_names = [gene_names[i] for i in gene_idx]

    n_features = len(gene_names)
    n_comp = min(n_components, n_cells - 1, n_features)

    # Pass 1: collect subsample for fitting
    fit_rows = []
    n_collected = 0
    rng = np.random.default_rng(42)

    for start, end, chunk in _iter_expression_chunks(sj, input_key, chunk_size=chunk_size):
        chunk = chunk.astype(np.float32)
        if gene_idx is not None:
            chunk = chunk[:, gene_idx]

        if n_cells <= n_subsample:
            fit_rows.append(chunk)
        else:
            # Reservoir-style: keep proportional sample from each chunk
            n_chunk = chunk.shape[0]
            n_take = max(1, int(n_subsample * n_chunk / n_cells))
            if n_take >= n_chunk:
                fit_rows.append(chunk)
            else:
                idx = rng.choice(n_chunk, n_take, replace=False)
                fit_rows.append(chunk[idx])
            n_collected += n_chunk

    fit_data = np.concatenate(fit_rows, axis=0)
    if fit_data.shape[0] > n_subsample:
        idx = rng.choice(fit_data.shape[0], n_subsample, replace=False)
        fit_data = fit_data[idx]

    model = _PCA(n_components=n_comp)
    model.fit(fit_data)
    del fit_data, fit_rows  # free memory

    # Pass 2: project all cells in chunks
    comp_cols = [f"PC_{i + 1}" for i in range(n_comp)]
    embedding_chunks = []
    cell_id_offset = 0

    for start, end, chunk in _iter_expression_chunks(sj, input_key, chunk_size=chunk_size):
        chunk = chunk.astype(np.float32)
        if gene_idx is not None:
            chunk = chunk[:, gene_idx]
        projected = model.transform(chunk)
        n = chunk.shape[0]
        chunk_ids = cell_ids[cell_id_offset:cell_id_offset + n]
        cell_id_offset += n
        chunk_df = pd.DataFrame(projected, columns=comp_cols)
        chunk_df.insert(0, "cell_id", chunk_ids)
        embedding_chunks.append(chunk_df)

    df = pd.concat(embedding_chunks, ignore_index=True)
    _atomic_write_parquet(df, maps_dir, output_key)

    loadings = model.components_.T  # (n_features, n_comp)
    ldf = pd.DataFrame(loadings, columns=comp_cols)
    ldf.insert(0, "gene", gene_names)
    _atomic_write_parquet(ldf, maps_dir, output_loadings_key)

    return output_key


def umap(
    sj: s_spatioloji,
    n_neighbors: int = 15,
    min_dist: float = 0.5,
    n_epochs: int | None = None,
    low_memory: bool = True,
    n_jobs: int = -1,
    input_key: str = "X_pca",
    output_key: str = "X_umap",
    force: bool = True,
) -> str:
    """UMAP embedding of the dataset.

    Optimized for large datasets with ``low_memory=True`` (reduces peak
    RAM) and ``n_jobs=-1`` (parallel KNN graph building).

    Args:
        sj: Dataset instance.
        n_neighbors: Number of neighbours for the UMAP graph.
        min_dist: Minimum distance between points in the embedding.
        n_epochs: Number of optimization epochs.  ``None`` uses the
            umap-learn default (200 for large datasets, 500 for small).
            Set to 200 or lower for faster results on 10M+ cells.
        low_memory: Use low-memory mode for KNN graph construction.
            Recommended for datasets over 1M cells.
        n_jobs: Number of parallel workers for KNN.  ``-1`` = all cores.
        input_key: Key to read input from (typically PCA embedding).
        output_key: Key to write the 2-D embedding under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Raises:
        ImportError: If ``umap-learn`` is not installed.

    Example:
        >>> umap(sj)
        'X_umap'
        >>> umap(sj, n_epochs=200, n_jobs=8)  # faster on large datasets
        'X_umap'
    """
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("Install with: pip install umap-learn") from None

    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, _ = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float32)

    kwargs: dict = {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "low_memory": low_memory,
        "n_jobs": n_jobs,
        "random_state": 42,
    }
    if n_epochs is not None:
        kwargs["n_epochs"] = n_epochs

    model = UMAP(**kwargs)
    embedding = model.fit_transform(matrix)

    df = pd.DataFrame(embedding, columns=["UMAP_1", "UMAP_2"])
    df.insert(0, "cell_id", cell_ids)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def tsne(
    sj: s_spatioloji,
    perplexity: float = 30,
    input_key: str = "X_pca",
    output_key: str = "X_tsne",
    force: bool = True,
) -> str:
    """t-SNE embedding of the dataset.

    Args:
        sj: Dataset instance.
        perplexity: Perplexity parameter for t-SNE.
        input_key: Key to read input from (typically PCA embedding).
        output_key: Key to write the 2-D embedding under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Raises:
        ImportError: If ``openTSNE`` is not installed.

    Example:
        >>> tsne(sj)
        'X_tsne'
    """
    try:
        from openTSNE import TSNE
    except ImportError:
        raise ImportError("Install with: pip install s_spatioloji[reduction]") from None

    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, _ = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float32)

    model = TSNE(perplexity=perplexity, random_state=42)
    embedding = model.fit(matrix)

    df = pd.DataFrame(np.array(embedding), columns=["tSNE_1", "tSNE_2"])
    df.insert(0, "cell_id", cell_ids)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key


def diffmap(
    sj: s_spatioloji,
    n_components: int = 15,
    input_key: str = "X_pca",
    output_key: str = "X_diffmap",
    force: bool = True,
) -> str:
    """Diffusion map embedding.

    Builds a KNN graph, constructs a diffusion operator, and extracts
    the top eigenvectors (skipping the trivial first).  Uses only
    ``scipy`` (core dependency).

    Args:
        sj: Dataset instance.
        n_components: Number of diffusion components to retain.
        input_key: Key to read input from (typically PCA embedding).
        output_key: Key to write the embedding under.
        force: If ``False``, skip if output already exists.

    Returns:
        The *output_key* string.

    Example:
        >>> diffmap(sj, n_components=15)
        'X_diffmap'
    """
    from scipy.sparse import diags, lil_matrix
    from scipy.sparse.linalg import eigs
    from sklearn.neighbors import NearestNeighbors

    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, _ = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float64)
    n_cells = matrix.shape[0]
    n_comp = min(n_components, n_cells - 2)

    k = min(30, n_cells - 1)
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(matrix)
    distances, indices = nn.kneighbors()

    sigma = distances[:, -1].copy()
    sigma[sigma == 0] = 1e-10

    W = lil_matrix((n_cells, n_cells))
    for i in range(n_cells):
        for j_idx, j in enumerate(indices[i]):
            d = distances[i, j_idx]
            w = np.exp(-(d**2) / (sigma[i] * sigma[j]))
            W[i, j] = w
            W[j, i] = w

    W = W.tocsr()
    row_sums = np.array(W.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    D_inv = diags(1.0 / row_sums)
    T = D_inv @ W

    eigenvalues, eigenvectors = eigs(T, k=n_comp + 1, which="LR")
    order = np.argsort(-eigenvalues.real)
    eigenvectors = eigenvectors[:, order].real
    embedding = eigenvectors[:, 1 : n_comp + 1]

    comp_cols = [f"DC_{i + 1}" for i in range(embedding.shape[1])]
    df = pd.DataFrame(embedding.astype(np.float32), columns=comp_cols)
    df.insert(0, "cell_id", cell_ids)
    _atomic_write_parquet(df, maps_dir, output_key)
    return output_key
