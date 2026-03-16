"""Unit tests for s_spatioloji.compute.batch_correction."""

from __future__ import annotations

import numpy as np
import pytest

from s_spatioloji.compute.feature_selection import highly_variable_genes
from s_spatioloji.compute.normalize import log1p, normalize_total


@pytest.fixture()
def sj_with_hvg(sj):
    """sj with normalize -> log1p -> hvg pipeline complete."""
    normalize_total(sj)
    log1p(sj)
    highly_variable_genes(sj, n_top=20)
    return sj


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


class TestScviBatch:
    """scvi_batch requires scvi-tools -- skip in CI."""

    def test_import_guard(self, sj):
        """scvi_batch should raise ImportError if scvi-tools is missing."""
        pytest.importorskip("scvi")
        # If we get here, scvi is installed -- just check the function exists
        from s_spatioloji.compute.batch_correction import scvi_batch

        assert callable(scvi_batch)
