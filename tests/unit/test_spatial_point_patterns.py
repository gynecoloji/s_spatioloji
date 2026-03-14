"""Unit tests for s_spatioloji.spatial.point.patterns."""

from __future__ import annotations

import pandas as pd
import pytest

from s_spatioloji.spatial.point.patterns import (
    clustering_coefficient,
    colocalization,
    gearys_c,
    getis_ord_gi,
    morans_i,
)


class TestColocalization:
    def test_returns_key(self, sj_with_pt_clusters):
        assert colocalization(sj_with_pt_clusters) == "pt_colocalization"

    def test_output_written(self, sj_with_pt_clusters):
        colocalization(sj_with_pt_clusters)
        assert sj_with_pt_clusters.maps.has("pt_colocalization")

    def test_columns(self, sj_with_pt_clusters):
        colocalization(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_colocalization"].compute()
        assert list(df.columns) == [
            "cluster_a",
            "cluster_b",
            "observed",
            "expected",
            "ratio",
            "log2_ratio",
        ]

    def test_has_rows(self, sj_with_pt_clusters):
        colocalization(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_colocalization"].compute()
        assert len(df) > 0

    def test_observed_nonneg(self, sj_with_pt_clusters):
        colocalization(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_colocalization"].compute()
        assert (df["observed"] >= 0).all()

    def test_force_false_skips(self, sj_with_pt_clusters):
        colocalization(sj_with_pt_clusters)
        colocalization(sj_with_pt_clusters, force=False)


class TestMoransI:
    def test_returns_key(self, sj_with_pt_clusters):
        assert morans_i(sj_with_pt_clusters) == "pt_morans_i"

    def test_columns(self, sj_with_pt_clusters):
        morans_i(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_morans_i"].compute()
        assert list(df.columns) == ["feature", "I", "expected_I", "z_score", "p_value"]

    def test_numeric_produces_one_row(self, sj_with_pt_clusters):
        morans_i(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_morans_i"].compute()
        assert len(df) == 1

    def test_spatially_clustered_high_I(self, sj_with_pt_clusters):
        morans_i(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_morans_i"].compute()
        assert (df["I"] > 0).any()

    def test_force_false_skips(self, sj_with_pt_clusters):
        morans_i(sj_with_pt_clusters)
        morans_i(sj_with_pt_clusters, force=False)


class TestGearysC:
    def test_returns_key(self, sj_with_pt_clusters):
        assert gearys_c(sj_with_pt_clusters) == "pt_gearys_c"

    def test_columns(self, sj_with_pt_clusters):
        gearys_c(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_gearys_c"].compute()
        assert list(df.columns) == ["feature", "C", "expected_C", "z_score", "p_value"]

    def test_spatially_clustered_low_C(self, sj_with_pt_clusters):
        gearys_c(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_gearys_c"].compute()
        assert (df["C"] < 1.0).any()

    def test_force_false_skips(self, sj_with_pt_clusters):
        gearys_c(sj_with_pt_clusters)
        gearys_c(sj_with_pt_clusters, force=False)


class TestClusteringCoefficient:
    def test_returns_key(self, sj_with_knn_graph):
        assert clustering_coefficient(sj_with_knn_graph) == "pt_clustering_coeff"

    def test_columns(self, sj_with_knn_graph):
        clustering_coefficient(sj_with_knn_graph)
        df = sj_with_knn_graph.maps["pt_clustering_coeff"].compute()
        assert list(df.columns) == ["cell_id", "clustering_coeff"]

    def test_shape(self, sj_with_knn_graph):
        clustering_coefficient(sj_with_knn_graph)
        df = sj_with_knn_graph.maps["pt_clustering_coeff"].compute()
        assert df.shape[0] == 200

    def test_range(self, sj_with_knn_graph):
        clustering_coefficient(sj_with_knn_graph)
        df = sj_with_knn_graph.maps["pt_clustering_coeff"].compute()
        assert (df["clustering_coeff"] >= 0).all()
        assert (df["clustering_coeff"] <= 1.0).all()

    def test_force_false_skips(self, sj_with_knn_graph):
        clustering_coefficient(sj_with_knn_graph)
        clustering_coefficient(sj_with_knn_graph, force=False)


class TestGetisOrdGi:
    def test_returns_key(self, sj_with_pt_clusters):
        assert getis_ord_gi(sj_with_pt_clusters, feature_key="leiden") == "pt_getis_ord"

    def test_columns(self, sj_with_pt_clusters):
        getis_ord_gi(sj_with_pt_clusters, feature_key="leiden")
        df = sj_with_pt_clusters.maps["pt_getis_ord"].compute()
        assert list(df.columns) == ["cell_id", "gi_stat", "p_value"]

    def test_shape(self, sj_with_pt_clusters):
        getis_ord_gi(sj_with_pt_clusters, feature_key="leiden")
        df = sj_with_pt_clusters.maps["pt_getis_ord"].compute()
        assert df.shape[0] == 200

    def test_p_values_valid(self, sj_with_pt_clusters):
        getis_ord_gi(sj_with_pt_clusters, feature_key="leiden")
        df = sj_with_pt_clusters.maps["pt_getis_ord"].compute()
        assert (df["p_value"] >= 0).all()
        assert (df["p_value"] <= 1.0).all()

    def test_categorical_raises(self, sj_with_pt_clusters):
        """Categorical feature should raise ValueError."""
        cells_df = sj_with_pt_clusters.cells.df.compute()
        cat_df = pd.DataFrame(
            {
                "cell_id": cells_df["cell_id"].values,
                "leiden_cat": pd.Categorical(["A", "B"] * 100),
            }
        )
        from s_spatioloji.compute import _atomic_write_parquet

        maps_dir = sj_with_pt_clusters.config.root / "maps"
        _atomic_write_parquet(cat_df, maps_dir, "leiden_cat")
        with pytest.raises(ValueError, match="numeric feature"):
            getis_ord_gi(sj_with_pt_clusters, feature_key="leiden_cat")

    def test_force_false_skips(self, sj_with_pt_clusters):
        getis_ord_gi(sj_with_pt_clusters, feature_key="leiden")
        getis_ord_gi(sj_with_pt_clusters, feature_key="leiden", force=False)
