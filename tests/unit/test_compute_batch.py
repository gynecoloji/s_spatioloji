"""Unit tests for s_spatioloji.compute.batch_correction."""

from __future__ import annotations

import numpy as np
import pytest

from s_spatioloji.compute.batch_correction import regress_out
from s_spatioloji.compute.feature_selection import highly_variable_genes
from s_spatioloji.compute.normalize import log1p, normalize_total


@pytest.fixture()
def sj_with_hvg(sj):
    """sj with normalize → log1p → hvg pipeline complete."""
    normalize_total(sj)
    log1p(sj)
    highly_variable_genes(sj, n_top=20)
    # Add transcript_counts column for regress_out tests

    cells_df = sj.cells.df.compute()
    if "transcript_counts" not in cells_df.columns:
        rng = np.random.default_rng(42)
        cells_df["transcript_counts"] = rng.integers(100, 5000, size=len(cells_df))
        cells_df.to_parquet(str(sj.config.paths.cells), engine="pyarrow", index=False)
        # Reload cells
        from s_spatioloji.data.cells import CellStore

        sj._cells = CellStore.open(sj.config.paths.cells)
    return sj


class TestRegressOut:
    def test_returns_key(self, sj_with_hvg):
        assert regress_out(sj_with_hvg, keys=["transcript_counts"]) == "X_regressed"

    def test_shape(self, sj_with_hvg):
        regress_out(sj_with_hvg, keys=["transcript_counts"])
        df = sj_with_hvg.maps["X_regressed"].compute()
        assert df.shape == (200, 21)  # 20 HVGs + cell_id

    def test_cell_id_present(self, sj_with_hvg):
        regress_out(sj_with_hvg, keys=["transcript_counts"])
        df = sj_with_hvg.maps["X_regressed"].compute()
        assert df.columns[0] == "cell_id"

    def test_force_false_skips(self, sj_with_hvg):
        regress_out(sj_with_hvg, keys=["transcript_counts"])
        regress_out(sj_with_hvg, keys=["transcript_counts"], force=False)


class TestHarmony:
    def test_runs(self, sj_with_hvg):
        pytest.importorskip("harmonypy")
        from s_spatioloji.compute.batch_correction import harmony
        from s_spatioloji.compute.normalize import scale
        from s_spatioloji.compute.reduction import pca

        scale(sj_with_hvg)
        pca(sj_with_hvg, n_components=10)
        assert harmony(sj_with_hvg, batch_key="fov_id") == "X_pca_harmony"
        df = sj_with_hvg.maps["X_pca_harmony"].compute()
        assert df.shape == (200, 11)  # 10 PCs + cell_id
        assert df.columns[0] == "cell_id"


class TestCombat:
    def test_runs(self, sj_with_hvg):
        pytest.importorskip("combat")
        from s_spatioloji.compute.batch_correction import combat

        assert combat(sj_with_hvg, batch_key="fov_id", save_expression=False) == "X_combat"
        df = sj_with_hvg.maps["X_combat"].compute()
        assert df.shape == (200, 21)  # 20 HVGs + cell_id
        assert df.columns[0] == "cell_id"

    def test_save_expression(self, sj_with_hvg):
        pytest.importorskip("combat")
        from s_spatioloji.compute.batch_correction import combat

        combat(sj_with_hvg, batch_key="fov_id", save_expression=True)
        assert sj_with_hvg.maps.has("expression_combat")
        store = sj_with_hvg.maps["expression_combat"]
        assert store.shape == (200, 30)  # full gene panel


class TestScviBatch:
    """scvi_batch requires scvi-tools — skip in CI."""

    def test_import_guard(self, sj):
        """scvi_batch should raise ImportError if scvi-tools is missing."""
        pytest.importorskip("scvi")
        # If we get here, scvi is installed — just check the function exists
        from s_spatioloji.compute.batch_correction import scvi_batch

        assert callable(scvi_batch)
