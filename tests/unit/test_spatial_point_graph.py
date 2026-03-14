"""Unit tests for s_spatioloji.spatial.point.graph."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import scipy.sparse

from s_spatioloji.spatial.point.graph import (
    _load_point_graph,
    _load_point_graph_sparse,
    build_knn_graph,
    build_radius_graph,
)


class TestBuildKnnGraph:
    def test_returns_key(self, sj):
        assert build_knn_graph(sj, k=6) == "knn_graph"

    def test_output_written(self, sj):
        build_knn_graph(sj, k=6)
        assert sj.maps.has("knn_graph")

    def test_columns(self, sj):
        build_knn_graph(sj, k=6)
        df = sj.maps["knn_graph"].compute()
        assert list(df.columns) == ["cell_id_a", "cell_id_b", "distance"]

    def test_edges_exist(self, sj):
        build_knn_graph(sj, k=6)
        df = sj.maps["knn_graph"].compute()
        assert len(df) > 100

    def test_edge_ordering(self, sj):
        """cell_id_a < cell_id_b lexicographically."""
        build_knn_graph(sj, k=6)
        df = sj.maps["knn_graph"].compute()
        assert (df["cell_id_a"] < df["cell_id_b"]).all()

    def test_distance_positive(self, sj):
        build_knn_graph(sj, k=6)
        df = sj.maps["knn_graph"].compute()
        assert (df["distance"] > 0).all()

    def test_force_false_skips(self, sj):
        build_knn_graph(sj, k=6)
        build_knn_graph(sj, k=6, force=False)

    def test_custom_output_key(self, sj):
        assert build_knn_graph(sj, k=6, output_key="my_knn") == "my_knn"
        assert sj.maps.has("my_knn")


class TestBuildRadiusGraph:
    def test_returns_key(self, sj):
        assert build_radius_graph(sj, radius=60.0) == "radius_graph"

    def test_output_written(self, sj):
        build_radius_graph(sj, radius=60.0)
        assert sj.maps.has("radius_graph")

    def test_columns(self, sj):
        build_radius_graph(sj, radius=60.0)
        df = sj.maps["radius_graph"].compute()
        assert list(df.columns) == ["cell_id_a", "cell_id_b", "distance"]

    def test_edges_within_radius(self, sj):
        build_radius_graph(sj, radius=60.0)
        df = sj.maps["radius_graph"].compute()
        assert (df["distance"] <= 60.0).all()
        assert (df["distance"] > 0).all()

    def test_force_false_skips(self, sj):
        build_radius_graph(sj, radius=60.0)
        build_radius_graph(sj, radius=60.0, force=False)


class TestLoadPointGraph:
    def test_returns_networkx_graph(self, sj_with_knn_graph):
        G = _load_point_graph(sj_with_knn_graph)
        assert isinstance(G, nx.Graph)

    def test_node_count(self, sj_with_knn_graph):
        G = _load_point_graph(sj_with_knn_graph)
        assert len(G.nodes) > 100

    def test_edge_weights(self, sj_with_knn_graph):
        G = _load_point_graph(sj_with_knn_graph)
        for _, _, data in G.edges(data=True):
            assert "distance" in data
            assert data["distance"] > 0

    def test_missing_graph_raises(self, sj):
        with pytest.raises(FileNotFoundError, match="Point graph not found"):
            _load_point_graph(sj)


class TestLoadPointGraphSparse:
    def test_returns_tuple(self, sj_with_knn_graph):
        adj, cell_ids = _load_point_graph_sparse(sj_with_knn_graph)
        assert isinstance(adj, scipy.sparse.csr_matrix)
        assert isinstance(cell_ids, np.ndarray)

    def test_symmetric(self, sj_with_knn_graph):
        adj, _ = _load_point_graph_sparse(sj_with_knn_graph)
        diff = adj - adj.T
        assert diff.nnz == 0

    def test_binary_unweighted(self, sj_with_knn_graph):
        adj, _ = _load_point_graph_sparse(sj_with_knn_graph)
        assert set(adj.data).issubset({1.0})

    def test_weighted_inverse_distance(self, sj_with_knn_graph):
        adj, _ = _load_point_graph_sparse(sj_with_knn_graph, weighted=True)
        assert (adj.data > 0).all()

    def test_missing_graph_raises(self, sj):
        with pytest.raises(FileNotFoundError):
            _load_point_graph_sparse(sj)
