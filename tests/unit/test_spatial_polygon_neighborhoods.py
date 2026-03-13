"""Unit tests for s_spatioloji.spatial.polygon.neighborhoods."""

from __future__ import annotations

import pytest

from s_spatioloji.spatial.polygon.neighborhoods import (
    neighborhood_composition,
    neighborhood_diversity,
    nth_order_neighbors,
)


class TestNeighborhoodComposition:
    def test_returns_key(self, sj_with_clusters):
        assert neighborhood_composition(sj_with_clusters) == "nhood_composition"

    def test_output_written(self, sj_with_clusters):
        neighborhood_composition(sj_with_clusters)
        assert sj_with_clusters.maps.has("nhood_composition")

    def test_cell_id_column(self, sj_with_clusters):
        neighborhood_composition(sj_with_clusters)
        df = sj_with_clusters.maps["nhood_composition"].compute()
        assert df.columns[0] == "cell_id"

    def test_shape(self, sj_with_clusters):
        neighborhood_composition(sj_with_clusters)
        df = sj_with_clusters.maps["nhood_composition"].compute()
        assert df.shape[0] == 200
        assert df.shape[1] >= 2

    def test_counts_nonneg(self, sj_with_clusters):
        neighborhood_composition(sj_with_clusters)
        df = sj_with_clusters.maps["nhood_composition"].compute()
        num_cols = [c for c in df.columns if c != "cell_id"]
        assert (df[num_cols].values >= 0).all()

    def test_force_false_skips(self, sj_with_clusters):
        neighborhood_composition(sj_with_clusters)
        neighborhood_composition(sj_with_clusters, force=False)

    def test_missing_graph_raises(self, sj_with_boundaries):
        with pytest.raises(FileNotFoundError):
            neighborhood_composition(sj_with_boundaries)


class TestNthOrderNeighbors:
    def test_returns_key(self, sj_with_graph):
        assert nth_order_neighbors(sj_with_graph) == "nhood_nth_order"

    def test_columns_order2(self, sj_with_graph):
        nth_order_neighbors(sj_with_graph, order=2)
        df = sj_with_graph.maps["nhood_nth_order"].compute()
        assert list(df.columns) == ["cell_id", "n_order_1", "n_order_2"]

    def test_columns_order3(self, sj_with_graph):
        nth_order_neighbors(sj_with_graph, order=3, output_key="nth3")
        df = sj_with_graph.maps["nth3"].compute()
        assert list(df.columns) == ["cell_id", "n_order_1", "n_order_2", "n_order_3"]

    def test_shape(self, sj_with_graph):
        nth_order_neighbors(sj_with_graph)
        df = sj_with_graph.maps["nhood_nth_order"].compute()
        assert df.shape[0] == 200

    def test_counts_nonneg(self, sj_with_graph):
        nth_order_neighbors(sj_with_graph)
        df = sj_with_graph.maps["nhood_nth_order"].compute()
        assert (df["n_order_1"] >= 0).all()
        assert (df["n_order_2"] >= 0).all()

    def test_force_false_skips(self, sj_with_graph):
        nth_order_neighbors(sj_with_graph)
        nth_order_neighbors(sj_with_graph, force=False)


class TestNeighborhoodDiversity:
    def test_returns_key(self, sj_with_clusters):
        assert neighborhood_diversity(sj_with_clusters) == "nhood_diversity"

    def test_columns(self, sj_with_clusters):
        neighborhood_diversity(sj_with_clusters)
        df = sj_with_clusters.maps["nhood_diversity"].compute()
        assert list(df.columns) == ["cell_id", "shannon", "simpson"]

    def test_shape(self, sj_with_clusters):
        neighborhood_diversity(sj_with_clusters)
        df = sj_with_clusters.maps["nhood_diversity"].compute()
        assert df.shape == (200, 3)

    def test_shannon_nonneg(self, sj_with_clusters):
        neighborhood_diversity(sj_with_clusters)
        df = sj_with_clusters.maps["nhood_diversity"].compute()
        assert (df["shannon"] >= 0).all()

    def test_simpson_range(self, sj_with_clusters):
        neighborhood_diversity(sj_with_clusters)
        df = sj_with_clusters.maps["nhood_diversity"].compute()
        assert (df["simpson"] >= 0).all()
        assert (df["simpson"] <= 1.0).all()

    def test_force_false_skips(self, sj_with_clusters):
        neighborhood_diversity(sj_with_clusters)
        neighborhood_diversity(sj_with_clusters, force=False)
