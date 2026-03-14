"""Unit tests for s_spatioloji.spatial.point.neighborhoods."""

from __future__ import annotations

from s_spatioloji.spatial.point.neighborhoods import (
    neighborhood_composition,
    neighborhood_diversity,
    nth_order_neighbors,
)


class TestNeighborhoodComposition:
    def test_returns_key(self, sj_with_pt_clusters):
        assert neighborhood_composition(sj_with_pt_clusters) == "pt_nhood_composition"

    def test_output_written(self, sj_with_pt_clusters):
        neighborhood_composition(sj_with_pt_clusters)
        assert sj_with_pt_clusters.maps.has("pt_nhood_composition")

    def test_columns_start_with_cell_id(self, sj_with_pt_clusters):
        neighborhood_composition(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_nhood_composition"].compute()
        assert df.columns[0] == "cell_id"
        assert df.shape[0] == 200

    def test_weighted_differs(self, sj_with_pt_clusters):
        neighborhood_composition(sj_with_pt_clusters, weighted=False, output_key="pt_nc_uw")
        neighborhood_composition(sj_with_pt_clusters, weighted=True, output_key="pt_nc_w")
        df_uw = sj_with_pt_clusters.maps["pt_nc_uw"].compute()
        df_w = sj_with_pt_clusters.maps["pt_nc_w"].compute()
        cols = [c for c in df_uw.columns if c != "cell_id"]
        assert not df_uw[cols].equals(df_w[cols])

    def test_force_false_skips(self, sj_with_pt_clusters):
        neighborhood_composition(sj_with_pt_clusters)
        neighborhood_composition(sj_with_pt_clusters, force=False)


class TestNthOrderNeighbors:
    def test_returns_key(self, sj_with_knn_graph):
        assert nth_order_neighbors(sj_with_knn_graph) == "pt_nhood_nth_order"

    def test_columns(self, sj_with_knn_graph):
        nth_order_neighbors(sj_with_knn_graph, order=2)
        df = sj_with_knn_graph.maps["pt_nhood_nth_order"].compute()
        assert list(df.columns) == ["cell_id", "n_order_1", "n_order_2"]

    def test_shape(self, sj_with_knn_graph):
        nth_order_neighbors(sj_with_knn_graph, order=2)
        df = sj_with_knn_graph.maps["pt_nhood_nth_order"].compute()
        assert df.shape[0] == 200

    def test_order_1_positive(self, sj_with_knn_graph):
        nth_order_neighbors(sj_with_knn_graph, order=1)
        df = sj_with_knn_graph.maps["pt_nhood_nth_order"].compute()
        assert (df["n_order_1"] > 0).all()

    def test_force_false_skips(self, sj_with_knn_graph):
        nth_order_neighbors(sj_with_knn_graph)
        nth_order_neighbors(sj_with_knn_graph, force=False)


class TestNeighborhoodDiversity:
    def test_returns_key(self, sj_with_pt_clusters):
        assert neighborhood_diversity(sj_with_pt_clusters) == "pt_nhood_diversity"

    def test_columns(self, sj_with_pt_clusters):
        neighborhood_diversity(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_nhood_diversity"].compute()
        assert list(df.columns) == ["cell_id", "shannon", "simpson"]

    def test_shape(self, sj_with_pt_clusters):
        neighborhood_diversity(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_nhood_diversity"].compute()
        assert df.shape[0] == 200

    def test_shannon_nonneg(self, sj_with_pt_clusters):
        neighborhood_diversity(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_nhood_diversity"].compute()
        assert (df["shannon"] >= 0).all()

    def test_simpson_range(self, sj_with_pt_clusters):
        neighborhood_diversity(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_nhood_diversity"].compute()
        assert (df["simpson"] >= 0).all()
        assert (df["simpson"] <= 1.0).all()

    def test_force_false_skips(self, sj_with_pt_clusters):
        neighborhood_diversity(sj_with_pt_clusters)
        neighborhood_diversity(sj_with_pt_clusters, force=False)
