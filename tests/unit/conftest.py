"""Shared fixtures for compute layer unit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from s_spatioloji.data.cells import CellStore
from s_spatioloji.data.config import SpatiolojiConfig
from s_spatioloji.data.core import s_spatioloji
from s_spatioloji.data.expression import ExpressionStore

N_CELLS = 200
N_GENES = 30


@pytest.fixture()
def sj(tmp_path: Path) -> s_spatioloji:
    """Minimal dataset: 200 cells x 30 genes, uint16 raw counts.

    Provides:
        - expression.zarr with random integer counts [0, 100)
        - cells.parquet with cell_id, x, y, fov_id (3 FOVs)
        - gene names: gene_0 .. gene_29
    """
    root = tmp_path / "dataset"
    root.mkdir()
    cfg = SpatiolojiConfig(root=root)

    # Expression store with raw counts
    rng = np.random.default_rng(42)
    data = rng.integers(0, 100, (N_CELLS, N_GENES), dtype=np.uint16)
    store = ExpressionStore.create(
        cfg.paths.expression, N_CELLS, N_GENES, cfg.chunks, dtype="uint16"
    )
    store.write_chunk(0, data)
    store.gene_names = [f"gene_{i}" for i in range(N_GENES)]

    # Cell metadata
    records = [
        {
            "cell_id": f"cell_{i}",
            "x": float(i % 20) * 50.0,
            "y": float(i // 20) * 50.0,
            "fov_id": i % 3,
        }
        for i in range(N_CELLS)
    ]
    CellStore.create(cfg.paths.cells, pd.DataFrame(records))

    return s_spatioloji.open(root)
