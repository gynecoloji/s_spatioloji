"""Subprocess entrypoint for conda bridge functions.

This module is invoked by the conda bridge as::

    python -m s_spatioloji.compute._runner \\
        --fn <function_name> \\
        --input <input_path> \\
        --output <output_path> \\
        --kwargs-file <kwargs_json_path>

It has **no dependency on the main s_spatioloji stack** — only ``numpy``,
``pandas``, and the target tool are imported inside each function handler.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _handle_magic(input_path: Path, output_path: Path, kwargs: dict) -> None:
    """Run MAGIC imputation."""
    import magic
    import pandas as pd

    data = np.load(input_path)
    X = data["X"]
    gene_names = kwargs.get("gene_names", [f"gene_{i}" for i in range(X.shape[1])])
    knn = kwargs.get("knn", 5)

    operator = magic.MAGIC(knn=knn)
    imputed = operator.fit_transform(pd.DataFrame(X, columns=gene_names))
    np.savez(output_path, X=imputed.values.astype(np.float32))


def _handle_alra(input_path: Path, output_path: Path, kwargs: dict) -> None:
    """Run ALRA imputation via rpy2."""
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    data = np.load(input_path)
    X = data["X"]

    numpy2ri.activate()
    ro.r("library(ALRA)")
    ro.globalenv["expr_matrix"] = X
    result = ro.r("alra(expr_matrix)[[1]]")
    numpy2ri.deactivate()

    np.savez(output_path, X=np.array(result, dtype=np.float32))


def _handle_scvi_train(input_path: Path, output_path: Path, kwargs: dict) -> None:
    """Train an scVI model and save to output_path."""
    import anndata
    import pandas as pd
    import scvi

    data = np.load(input_path)
    X = data["X"]
    gene_names = kwargs["gene_names"]
    cell_ids = kwargs["cell_ids"]
    n_latent = kwargs["n_latent"]
    n_epochs = kwargs["n_epochs"]
    batch_key = kwargs["batch_key"]
    batch_values = kwargs.get("batch_values")

    obs = pd.DataFrame({"cell_id": cell_ids})
    if batch_key is not None and batch_values is not None:
        obs[batch_key] = batch_values

    adata = anndata.AnnData(
        X=X.astype(np.float32),
        obs=obs,
        var=pd.DataFrame(index=gene_names),
    )

    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
    model = scvi.model.SCVI(adata, n_latent=n_latent)
    model.train(max_epochs=n_epochs)

    model_dir = Path(output_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir), overwrite=True)


def _handle_gpu_umap(input_path: Path, output_path: Path, kwargs: dict) -> None:
    """Run GPU UMAP via cuml."""
    from cuml.manifold import UMAP

    data = np.load(input_path)
    X = data["X"].astype(np.float32)

    umap_kwargs = {
        "n_neighbors": kwargs.get("n_neighbors", 15),
        "min_dist": kwargs.get("min_dist", 0.5),
        "random_state": 42,
    }
    if "n_epochs" in kwargs and kwargs["n_epochs"] is not None:
        umap_kwargs["n_epochs"] = kwargs["n_epochs"]

    model = UMAP(**umap_kwargs)
    embedding = model.fit_transform(X)
    if hasattr(embedding, "get"):
        embedding = embedding.get()
    np.savez(output_path, X=np.asarray(embedding, dtype=np.float32))


def _handle_gpu_leiden(input_path: Path, output_path: Path, kwargs: dict) -> None:
    """Run GPU Leiden via cuml (KNN) + cugraph (Leiden)."""
    import cudf
    import cugraph
    from cuml.neighbors import NearestNeighbors

    data = np.load(input_path)
    X = data["X"].astype(np.float32)
    n_cells = X.shape[0]
    n_neighbors = kwargs.get("n_neighbors", 15)
    resolution = kwargs.get("resolution", 1.0)
    k = min(n_neighbors, n_cells - 1)

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X)
    _, indices = nn.kneighbors(X)
    if hasattr(indices, "get"):
        indices = indices.get()
    indices = np.asarray(indices)[:, 1:]

    sources = np.repeat(np.arange(n_cells), k)
    targets = indices.ravel()
    lo = np.minimum(sources, targets)
    hi = np.maximum(sources, targets)
    edge_pairs = np.unique(np.column_stack([lo, hi]), axis=0)
    mask = edge_pairs[:, 0] != edge_pairs[:, 1]
    edge_pairs = edge_pairs[mask]

    edge_df = cudf.DataFrame({"src": edge_pairs[:, 0], "dst": edge_pairs[:, 1]})
    G = cugraph.Graph()
    G.from_cudf_edgelist(edge_df, source="src", destination="dst")
    parts, _ = cugraph.leiden(G, resolution=resolution)
    parts = parts.sort_values("vertex").reset_index(drop=True)
    labels = parts["partition"].values
    if hasattr(labels, "get"):
        labels = labels.get()
    np.savez(output_path, labels=np.asarray(labels, dtype=np.int32))


_HANDLERS = {
    "magic": _handle_magic,
    "alra": _handle_alra,
    "scvi_train": _handle_scvi_train,
    "gpu_umap": _handle_gpu_umap,
    "gpu_leiden": _handle_gpu_leiden,
}


def main() -> None:
    """Parse arguments and dispatch to the appropriate handler."""
    parser = argparse.ArgumentParser(description="s_spatioloji conda bridge runner")
    parser.add_argument("--fn", required=True, help="Function name to run")
    parser.add_argument("--input", required=True, help="Path to input file")
    parser.add_argument("--output", required=True, help="Path to output file")
    parser.add_argument("--kwargs-file", required=True, help="Path to kwargs JSON file")
    args = parser.parse_args()

    kwargs = json.loads(Path(args.kwargs_file).read_text())

    handler = _HANDLERS.get(args.fn)
    if handler is None:
        print(f"Unknown function: {args.fn}", file=sys.stderr)
        sys.exit(1)

    handler(Path(args.input), Path(args.output), kwargs)


if __name__ == "__main__":
    main()
