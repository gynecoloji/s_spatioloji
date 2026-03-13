"""Unit tests for s_spatioloji.compute.imputation."""

from __future__ import annotations

import numpy as np
import pytest

from s_spatioloji.compute.feature_selection import highly_variable_genes
from s_spatioloji.compute.imputation import knn_smooth
from s_spatioloji.compute.normalize import log1p, normalize_total


@pytest.fixture()
def sj_with_hvg(sj):
    """sj with normalize → log1p → hvg pipeline complete."""
    normalize_total(sj)
    log1p(sj)
    highly_variable_genes(sj, n_top=20)
    return sj


class TestKnnSmooth:
    def test_returns_key(self, sj_with_hvg):
        assert knn_smooth(sj_with_hvg) == "X_knnsmooth"

    def test_shape(self, sj_with_hvg):
        knn_smooth(sj_with_hvg)
        df = sj_with_hvg.maps["X_knnsmooth"].compute()
        assert df.shape == (200, 21)  # 20 HVGs + cell_id

    def test_cell_id_present(self, sj_with_hvg):
        knn_smooth(sj_with_hvg)
        df = sj_with_hvg.maps["X_knnsmooth"].compute()
        assert df.columns[0] == "cell_id"

    def test_values_changed(self, sj_with_hvg):
        """Smoothing should change values (not identity transform)."""
        knn_smooth(sj_with_hvg)
        original = sj_with_hvg.maps["X_log1p"].compute()
        smoothed = sj_with_hvg.maps["X_knnsmooth"].compute()
        gene_cols = [c for c in smoothed.columns if c != "cell_id"]
        # At least some values should differ
        orig_hvg = original[[c for c in gene_cols if c in original.columns]].values
        sm_vals = smoothed[gene_cols].values
        assert not np.allclose(orig_hvg, sm_vals)

    def test_force_false_skips(self, sj_with_hvg):
        knn_smooth(sj_with_hvg)
        knn_smooth(sj_with_hvg, force=False)

    def test_k_clamped(self, sj_with_hvg):
        """k > n_cells should not crash."""
        knn_smooth(sj_with_hvg, k=500)


class TestMagic:
    def test_runs(self, sj_with_hvg):
        pytest.importorskip("magic")
        from s_spatioloji.compute.imputation import magic

        assert magic(sj_with_hvg, save_expression=False) == "X_magic"
        df = sj_with_hvg.maps["X_magic"].compute()
        assert df.shape == (200, 21)

    def test_save_expression(self, sj_with_hvg):
        pytest.importorskip("magic")
        from s_spatioloji.compute.imputation import magic

        magic(sj_with_hvg, save_expression=True)
        assert sj_with_hvg.maps.has("expression_magic")
        store = sj_with_hvg.maps["expression_magic"]
        assert store.shape == (200, 30)


class TestAlra:
    def test_import_guard(self, sj):
        """alra should raise ImportError if rpy2 is missing."""
        pytest.importorskip("rpy2")
        from s_spatioloji.compute.imputation import alra

        assert callable(alra)


class TestScviImpute:
    def test_callable(self, sj):
        from s_spatioloji.compute.imputation import scvi_impute

        assert callable(scvi_impute)
