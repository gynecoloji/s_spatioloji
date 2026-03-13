"""Unit tests for s_spatioloji.spatial.polygon.patterns."""

from __future__ import annotations

from s_spatioloji.spatial.polygon.patterns import (
    border_enrichment,
    clustering_coefficient,
    colocalization,
    gearys_c,
    morans_i,
)


class TestColocalization:
    def test_returns_key(self, sj_with_clusters):
        assert colocalization(sj_with_clusters) == "colocalization"

    def test_output_written(self, sj_with_clusters):
        colocalization(sj_with_clusters)
        assert sj_with_clusters.maps.has("colocalization")

    def test_columns(self, sj_with_clusters):
        colocalization(sj_with_clusters)
        df = sj_with_clusters.maps["colocalization"].compute()
        assert list(df.columns) == [
            "cluster_a",
            "cluster_b",
            "observed",
            "expected",
            "ratio",
            "log2_ratio",
        ]

    def test_has_rows(self, sj_with_clusters):
        colocalization(sj_with_clusters)
        df = sj_with_clusters.maps["colocalization"].compute()
        assert len(df) > 0

    def test_observed_nonneg(self, sj_with_clusters):
        colocalization(sj_with_clusters)
        df = sj_with_clusters.maps["colocalization"].compute()
        assert (df["observed"] >= 0).all()

    def test_force_false_skips(self, sj_with_clusters):
        colocalization(sj_with_clusters)
        colocalization(sj_with_clusters, force=False)


class TestMoransI:
    def test_returns_key(self, sj_with_clusters):
        assert morans_i(sj_with_clusters) == "morans_i"

    def test_columns(self, sj_with_clusters):
        morans_i(sj_with_clusters)
        df = sj_with_clusters.maps["morans_i"].compute()
        assert list(df.columns) == ["feature", "I", "expected_I", "z_score", "p_value"]

    def test_numeric_produces_one_row(self, sj_with_clusters):
        """Integer leiden labels are treated as numeric, producing one row."""
        morans_i(sj_with_clusters)
        df = sj_with_clusters.maps["morans_i"].compute()
        assert len(df) == 1

    def test_spatially_clustered_high_I(self, sj_with_clusters):
        """Spatially coherent clusters should have positive Moran's I."""
        morans_i(sj_with_clusters)
        df = sj_with_clusters.maps["morans_i"].compute()
        assert (df["I"] > 0).any()

    def test_force_false_skips(self, sj_with_clusters):
        morans_i(sj_with_clusters)
        morans_i(sj_with_clusters, force=False)


class TestGearysC:
    def test_returns_key(self, sj_with_clusters):
        assert gearys_c(sj_with_clusters) == "gearys_c"

    def test_columns(self, sj_with_clusters):
        gearys_c(sj_with_clusters)
        df = sj_with_clusters.maps["gearys_c"].compute()
        assert list(df.columns) == ["feature", "C", "expected_C", "z_score", "p_value"]

    def test_spatially_clustered_low_C(self, sj_with_clusters):
        """Spatially coherent clusters should have C < 1."""
        gearys_c(sj_with_clusters)
        df = sj_with_clusters.maps["gearys_c"].compute()
        assert (df["C"] < 1.0).any()

    def test_force_false_skips(self, sj_with_clusters):
        gearys_c(sj_with_clusters)
        gearys_c(sj_with_clusters, force=False)


class TestClusteringCoefficient:
    def test_returns_key(self, sj_with_graph):
        assert clustering_coefficient(sj_with_graph) == "clustering_coeff"

    def test_columns(self, sj_with_graph):
        clustering_coefficient(sj_with_graph)
        df = sj_with_graph.maps["clustering_coeff"].compute()
        assert list(df.columns) == ["cell_id", "clustering_coeff"]

    def test_shape(self, sj_with_graph):
        clustering_coefficient(sj_with_graph)
        df = sj_with_graph.maps["clustering_coeff"].compute()
        assert df.shape[0] == 200

    def test_range(self, sj_with_graph):
        clustering_coefficient(sj_with_graph)
        df = sj_with_graph.maps["clustering_coeff"].compute()
        assert (df["clustering_coeff"] >= 0).all()
        assert (df["clustering_coeff"] <= 1.0).all()

    def test_force_false_skips(self, sj_with_graph):
        clustering_coefficient(sj_with_graph)
        clustering_coefficient(sj_with_graph, force=False)


class TestBorderEnrichment:
    def test_returns_key(self, sj_with_clusters):
        assert border_enrichment(sj_with_clusters) == "border_enrichment"

    def test_columns(self, sj_with_clusters):
        border_enrichment(sj_with_clusters)
        df = sj_with_clusters.maps["border_enrichment"].compute()
        assert list(df.columns) == [
            "cluster",
            "n_cells",
            "n_border",
            "border_fraction",
            "enrichment",
        ]

    def test_has_rows(self, sj_with_clusters):
        border_enrichment(sj_with_clusters)
        df = sj_with_clusters.maps["border_enrichment"].compute()
        assert len(df) == 4  # 4 clusters

    def test_n_cells_positive(self, sj_with_clusters):
        border_enrichment(sj_with_clusters)
        df = sj_with_clusters.maps["border_enrichment"].compute()
        assert (df["n_cells"] > 0).all()

    def test_border_fraction_range(self, sj_with_clusters):
        border_enrichment(sj_with_clusters)
        df = sj_with_clusters.maps["border_enrichment"].compute()
        assert (df["border_fraction"] >= 0).all()
        assert (df["border_fraction"] <= 1.0).all()

    def test_force_false_skips(self, sj_with_clusters):
        border_enrichment(sj_with_clusters)
        border_enrichment(sj_with_clusters, force=False)
