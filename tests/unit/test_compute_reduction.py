"""Unit tests for s_spatioloji.compute.reduction."""

from __future__ import annotations

import numpy as np
import pytest

from s_spatioloji.compute.feature_selection import highly_variable_genes
from s_spatioloji.compute.normalize import log1p, normalize_total, scale
from s_spatioloji.compute.reduction import diffmap, pca


@pytest.fixture()
def sj_scaled(sj):
    """sj with normalize → log1p → hvg → scale pipeline complete."""
    normalize_total(sj)
    log1p(sj)
    highly_variable_genes(sj, n_top=20)
    scale(sj)
    return sj


class TestPCA:
    def test_returns_key(self, sj_scaled):
        assert pca(sj_scaled, n_components=10) == "X_pca"

    def test_embedding_shape(self, sj_scaled):
        pca(sj_scaled, n_components=10)
        df = sj_scaled.maps["X_pca"].compute()
        assert df.shape == (200, 11)  # 10 PCs + cell_id
        assert df.columns[0] == "cell_id"

    def test_loadings_written(self, sj_scaled):
        pca(sj_scaled, n_components=10)
        assert sj_scaled.maps.has("X_pca_loadings")
        ldf = sj_scaled.maps["X_pca_loadings"].compute()
        assert ldf.columns[0] == "gene"
        assert ldf.shape[1] == 11  # 10 PCs + gene

    def test_n_components_clamped(self, sj_scaled):
        """n_components > n_features should be silently clamped."""
        pca(sj_scaled, n_components=500)  # fixture has 20 HVGs
        df = sj_scaled.maps["X_pca"].compute()
        n_pcs = df.shape[1] - 1  # minus cell_id
        assert n_pcs <= 20  # clamped to n_features

    def test_force_false_both_must_exist(self, sj_scaled):
        pca(sj_scaled, n_components=5)
        # Delete loadings — force=False should NOT skip
        loadings_path = sj_scaled.config.root / "maps" / "X_pca_loadings.parquet"
        loadings_path.unlink()
        pca(sj_scaled, n_components=5, force=False)
        assert loadings_path.exists()  # recomputed because loadings were missing

    def test_custom_output_key(self, sj_scaled):
        pca(sj_scaled, n_components=5, output_key="X_pca_v2", output_loadings_key="X_pca_v2_loadings")
        assert sj_scaled.maps.has("X_pca_v2")
        assert sj_scaled.maps.has("X_pca_v2_loadings")


class TestDiffmap:
    def test_returns_key(self, sj_scaled):
        pca(sj_scaled, n_components=10)
        assert diffmap(sj_scaled, n_components=5) == "X_diffmap"

    def test_shape(self, sj_scaled):
        pca(sj_scaled, n_components=10)
        diffmap(sj_scaled, n_components=5)
        df = sj_scaled.maps["X_diffmap"].compute()
        assert df.shape == (200, 6)  # 5 DCs + cell_id

    def test_cell_id_present(self, sj_scaled):
        pca(sj_scaled, n_components=10)
        diffmap(sj_scaled, n_components=5)
        df = sj_scaled.maps["X_diffmap"].compute()
        assert df.columns[0] == "cell_id"

    def test_values_finite(self, sj_scaled):
        pca(sj_scaled, n_components=10)
        diffmap(sj_scaled, n_components=5)
        df = sj_scaled.maps["X_diffmap"].compute()
        gene_cols = [c for c in df.columns if c != "cell_id"]
        assert np.all(np.isfinite(df[gene_cols].values))


class TestUmapTsne:
    """UMAP and tSNE require optional deps — skip if not installed."""

    def test_umap_runs(self, sj_scaled):
        pytest.importorskip("umap")
        from s_spatioloji.compute.reduction import umap

        pca(sj_scaled, n_components=10)
        assert umap(sj_scaled) == "X_umap"
        df = sj_scaled.maps["X_umap"].compute()
        assert df.shape == (200, 3)  # UMAP_1, UMAP_2 + cell_id

    def test_tsne_runs(self, sj_scaled):
        pytest.importorskip("openTSNE")
        from s_spatioloji.compute.reduction import tsne

        pca(sj_scaled, n_components=10)
        assert tsne(sj_scaled) == "X_tsne"
        df = sj_scaled.maps["X_tsne"].compute()
        assert df.shape == (200, 3)  # tSNE_1, tSNE_2 + cell_id
