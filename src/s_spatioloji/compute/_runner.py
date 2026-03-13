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


_HANDLERS = {
    "magic": _handle_magic,
    "alra": _handle_alra,
    "scvi_train": _handle_scvi_train,
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
