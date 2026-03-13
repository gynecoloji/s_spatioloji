"""Shared scVI model training helper.

This is a private module — not part of the public API.  Both ``scvi_batch``
and ``scvi_impute`` delegate model training here.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from s_spatioloji.data.core import s_spatioloji


def _scvi_train(
    sj: s_spatioloji,
    input_key: str,
    batch_key: str | None,
    n_latent: int,
    n_epochs: int,
    conda_env: str | None,
    timeout: int,
    force: bool,
) -> Path:
    """Train or reuse a cached scVI model.

    The model is cached at ``maps/_scvi_model/``.  A fingerprint file
    (``params.json``) stores ``input_key``, ``n_latent``, ``batch_key``,
    and ``n_epochs``.  The cache is reused if the fingerprint matches;
    otherwise the old model is deleted and a new one is trained.

    Args:
        sj: Dataset instance.
        input_key: Expression input key (must be raw counts).
        batch_key: Batch column name, or ``None`` for unsupervised.
        n_latent: Latent dimension size.
        n_epochs: Number of training epochs.
        conda_env: Conda environment name, or ``None`` for in-process.
        timeout: Subprocess timeout in seconds.
        force: If ``True``, always retrain (deletes existing cache).

    Returns:
        Path to the ``maps/_scvi_model/`` directory.
    """
    maps_dir = sj.config.root / "maps"
    maps_dir.mkdir(exist_ok=True)
    model_dir = maps_dir / "_scvi_model"
    params_path = model_dir / "params.json"

    fingerprint = {
        "input_key": input_key,
        "n_latent": n_latent,
        "batch_key": "null" if batch_key is None else batch_key,
        "n_epochs": n_epochs,
    }

    if force and model_dir.exists():
        shutil.rmtree(model_dir)

    if not force and params_path.exists():
        existing = json.loads(params_path.read_text())
        if existing == fingerprint:
            return model_dir
        # Fingerprint mismatch — retrain
        shutil.rmtree(model_dir)

    # Prepare input data
    from s_spatioloji.compute import _load_dense

    matrix, cell_ids, gene_names = _load_dense(sj, input_key)
    matrix = matrix.astype(np.float32)

    if conda_env is not None:
        _train_via_conda(
            matrix, cell_ids, gene_names, batch_key, sj, n_latent, n_epochs,
            conda_env, timeout, model_dir, fingerprint,
        )
    else:
        _train_in_process(
            matrix, cell_ids, gene_names, batch_key, sj, n_latent, n_epochs,
            model_dir, fingerprint,
        )

    return model_dir


def _train_in_process(
    matrix, cell_ids, gene_names, batch_key, sj, n_latent, n_epochs,
    model_dir, fingerprint,
):
    """Train scVI model in the current Python process."""
    try:
        import anndata
        import scvi
    except ImportError:
        raise ImportError("Install with: pip install s_spatioloji[imputation]") from None

    import pandas as pd

    obs = pd.DataFrame({"cell_id": cell_ids})
    if batch_key is not None:
        cells_df = sj.cells.df.compute()
        obs[batch_key] = cells_df[batch_key].values

    adata = anndata.AnnData(
        X=matrix,
        obs=obs,
        var=pd.DataFrame(index=gene_names),
    )

    scvi.model.SCVI.setup_anndata(
        adata,
        batch_key=batch_key,
    )
    model = scvi.model.SCVI(adata, n_latent=n_latent)
    model.train(max_epochs=n_epochs)

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir), overwrite=True)
    (model_dir / "params.json").write_text(json.dumps(fingerprint))


def _train_via_conda(
    matrix, cell_ids, gene_names, batch_key, sj, n_latent, n_epochs,
    conda_env, timeout, model_dir, fingerprint,
):
    """Train scVI model via conda bridge subprocess."""
    _validate_conda_env(conda_env)


    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Serialise expression
        np.savez(tmpdir / "expression.npz", X=matrix)

        # Serialise kwargs
        batch_values = None
        if batch_key is not None:
            cells_df = sj.cells.df.compute()
            batch_values = list(cells_df[batch_key].astype(str))

        kwargs = {
            "fn": "scvi_train",
            "n_latent": n_latent,
            "n_epochs": n_epochs,
            "batch_key": batch_key,
            "batch_values": batch_values,
            "gene_names": gene_names,
            "cell_ids": cell_ids,
            "model_dir": str(model_dir),
        }
        (tmpdir / "kwargs.json").write_text(json.dumps(kwargs))

        cmd = [
            "conda", "run", "-n", conda_env,
            "python", "-m", "s_spatioloji.compute._runner",
            "--fn", "scvi_train",
            "--input", str(tmpdir / "expression.npz"),
            "--output", str(model_dir),
            "--kwargs-file", str(tmpdir / "kwargs.json"),
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"scVI training failed in conda env '{conda_env}':\n{result.stderr}"
            )

    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "params.json").write_text(json.dumps(fingerprint))


def _validate_conda_env(conda_env: str) -> None:
    """Raise ValueError if the conda environment does not exist."""
    result = subprocess.run(
        ["conda", "env", "list"], capture_output=True, text=True,
    )
    envs = result.stdout
    if conda_env not in envs:
        raise ValueError(
            f"Conda environment '{conda_env}' not found. "
            f"Available envs:\n{envs}"
        )
