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
) -> str:
    """PCA via subsample-fit then project-all.

    Fits PCA on a subsample of cells (if dataset is large) then projects
    all cells into the learned space.  ``n_components`` is automatically
    clamped to ``min(n_components, n_cells - 1, n_features)``.

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

    Returns:
        The *output_key* string.

    Example:
        >>> pca(sj, n_components=50)
        'X_pca'
    """
    from sklearn.decomposition import PCA as _PCA

    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    loadings_path = maps_dir / f"{output_loadings_key}.parquet"
    if not force and out_path.exists() and loadings_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, gene_names = _load_dense(sj, input_key)

    if hvg:
        hvg_genes = _load_hvg_genes(sj, hvg_key)
        gene_idx = [gene_names.index(g) for g in hvg_genes if g in gene_names]
        matrix = matrix[:, gene_idx]
        gene_names = [gene_names[i] for i in gene_idx]

    matrix = matrix.astype(np.float32)
    n_cells, n_features = matrix.shape
    n_comp = min(n_components, n_cells - 1, n_features)

    if n_cells > n_subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_cells, n_subsample, replace=False)
        fit_data = matrix[idx]
    else:
        fit_data = matrix

    model = _PCA(n_components=n_comp)
    model.fit(fit_data)
    embedding = model.transform(matrix)

    comp_cols = [f"PC_{i + 1}" for i in range(n_comp)]
    df = pd.DataFrame(embedding, columns=comp_cols)
    df.insert(0, "cell_id", cell_ids)
    _atomic_write_parquet(df, maps_dir, output_key)

    loadings = model.components_.T  # (n_features, n_comp)
    ldf = pd.DataFrame(loadings, columns=comp_cols)
    ldf.insert(0, "gene", gene_names)
    _atomic_write_parquet(ldf, maps_dir, output_loadings_key)

    return output_key


def umap(
    sj: s_spatioloji,
    n_neighbors: int = 15,
    input_key: str = "X_pca",
    output_key: str = "X_umap",
    force: bool = True,
) -> str:
    """UMAP embedding of the dataset.

    Args:
        sj: Dataset instance.
        n_neighbors: Number of neighbours for the UMAP graph.
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
    """
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("Install with: pip install s_spatioloji[reduction]") from None

    maps_dir = sj.config.root / "maps"
    out_path = maps_dir / f"{output_key}.parquet"
    if not force and out_path.exists():
        return output_key

    maps_dir.mkdir(exist_ok=True)
    matrix, cell_ids, _ = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float32)

    model = UMAP(n_neighbors=n_neighbors, random_state=42)
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
