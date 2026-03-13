"""Unit tests for s_spatioloji.compute.clustering."""

from __future__ import annotations

import pytest

from s_spatioloji.compute.clustering import hierarchical, kmeans
from s_spatioloji.compute.feature_selection import highly_variable_genes
from s_spatioloji.compute.normalize import log1p, normalize_total, scale
from s_spatioloji.compute.reduction import pca


@pytest.fixture()
def sj_with_pca(sj):
    """sj with full pipeline through PCA."""
    normalize_total(sj)
    log1p(sj)
    highly_variable_genes(sj, n_top=20)
    scale(sj)
    pca(sj, n_components=10)
    return sj


class TestKmeans:
    def test_returns_key(self, sj_with_pca):
        assert kmeans(sj_with_pca, n_clusters=5) == "kmeans"

    def test_shape(self, sj_with_pca):
        kmeans(sj_with_pca, n_clusters=5)
        df = sj_with_pca.maps["kmeans"].compute()
        assert df.shape == (200, 2)  # cell_id + kmeans
        assert list(df.columns) == ["cell_id", "kmeans"]

    def test_n_clusters(self, sj_with_pca):
        kmeans(sj_with_pca, n_clusters=5)
        df = sj_with_pca.maps["kmeans"].compute()
        assert df["kmeans"].nunique() == 5

    def test_force_false_skips(self, sj_with_pca):
        kmeans(sj_with_pca, n_clusters=5)
        kmeans(sj_with_pca, n_clusters=3, force=False)
        df = sj_with_pca.maps["kmeans"].compute()
        assert df["kmeans"].nunique() == 5  # not overwritten


class TestHierarchical:
    def test_returns_key(self, sj_with_pca):
        assert hierarchical(sj_with_pca, n_clusters=5) == "hierarchical"

    def test_shape(self, sj_with_pca):
        hierarchical(sj_with_pca, n_clusters=5)
        df = sj_with_pca.maps["hierarchical"].compute()
        assert df.shape == (200, 2)
        assert list(df.columns) == ["cell_id", "hierarchical"]

    def test_n_clusters(self, sj_with_pca):
        hierarchical(sj_with_pca, n_clusters=5)
        df = sj_with_pca.maps["hierarchical"].compute()
        assert df["hierarchical"].nunique() == 5


class TestLeiden:
    def test_runs(self, sj_with_pca):
        pytest.importorskip("leidenalg")
        pytest.importorskip("igraph")
        from s_spatioloji.compute.clustering import leiden

        assert leiden(sj_with_pca, resolution=1.0) == "leiden"
        df = sj_with_pca.maps["leiden"].compute()
        assert df.shape[0] == 200
        assert list(df.columns) == ["cell_id", "leiden"]
        assert df["leiden"].nunique() >= 1

    def test_n_neighbors_clamped(self, sj_with_pca):
        """n_neighbors > n_cells should not crash."""
        pytest.importorskip("leidenalg")
        pytest.importorskip("igraph")
        from s_spatioloji.compute.clustering import leiden

        leiden(sj_with_pca, n_neighbors=500)  # 200 cells → clamped to 199


class TestLouvain:
    def test_runs(self, sj_with_pca):
        pytest.importorskip("community")
        from s_spatioloji.compute.clustering import louvain

        assert louvain(sj_with_pca, resolution=1.0) == "louvain"
        df = sj_with_pca.maps["louvain"].compute()
        assert df.shape[0] == 200
        assert list(df.columns) == ["cell_id", "louvain"]
