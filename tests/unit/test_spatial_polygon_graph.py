"""Unit tests for s_spatioloji.spatial.polygon.graph."""

from __future__ import annotations

import networkx as nx
import pytest

from s_spatioloji.spatial.polygon.graph import _load_contact_graph, build_contact_graph


class TestBuildContactGraph:
    def test_returns_key(self, sj_with_boundaries):
        assert build_contact_graph(sj_with_boundaries) == "contact_graph"

    def test_output_written(self, sj_with_boundaries):
        build_contact_graph(sj_with_boundaries)
        assert sj_with_boundaries.maps.has("contact_graph")

    def test_columns(self, sj_with_boundaries):
        build_contact_graph(sj_with_boundaries)
        df = sj_with_boundaries.maps["contact_graph"].compute()
        assert list(df.columns) == ["cell_id_a", "cell_id_b", "shared_length"]

    def test_edges_exist(self, sj_with_boundaries):
        """Voronoi tessellation should produce many touching pairs."""
        build_contact_graph(sj_with_boundaries)
        df = sj_with_boundaries.maps["contact_graph"].compute()
        assert len(df) > 50

    def test_edge_ordering(self, sj_with_boundaries):
        """cell_id_a < cell_id_b lexicographically."""
        build_contact_graph(sj_with_boundaries)
        df = sj_with_boundaries.maps["contact_graph"].compute()
        assert (df["cell_id_a"] < df["cell_id_b"]).all()

    def test_shared_length_nonneg(self, sj_with_boundaries):
        build_contact_graph(sj_with_boundaries)
        df = sj_with_boundaries.maps["contact_graph"].compute()
        assert (df["shared_length"] >= 0).all()

    def test_buffer_distance(self, sj_with_boundaries):
        """Larger buffer -> more edges."""
        build_contact_graph(sj_with_boundaries, buffer_distance=0.0, output_key="g0")
        build_contact_graph(sj_with_boundaries, buffer_distance=10.0, output_key="g10")
        df0 = sj_with_boundaries.maps["g0"].compute()
        df10 = sj_with_boundaries.maps["g10"].compute()
        assert len(df10) >= len(df0)

    def test_force_false_skips(self, sj_with_boundaries):
        build_contact_graph(sj_with_boundaries)
        build_contact_graph(sj_with_boundaries, force=False)

    def test_custom_output_key(self, sj_with_boundaries):
        assert build_contact_graph(sj_with_boundaries, output_key="my_graph") == "my_graph"
        assert sj_with_boundaries.maps.has("my_graph")


class TestLoadContactGraph:
    def test_returns_networkx_graph(self, sj_with_graph):
        G = _load_contact_graph(sj_with_graph)
        assert isinstance(G, nx.Graph)

    def test_node_count(self, sj_with_graph):
        G = _load_contact_graph(sj_with_graph)
        assert len(G.nodes) > 100

    def test_edge_weights(self, sj_with_graph):
        G = _load_contact_graph(sj_with_graph)
        for _, _, data in G.edges(data=True):
            assert "shared_length" in data
            assert data["shared_length"] >= 0

    def test_missing_graph_raises(self, sj_with_boundaries):
        with pytest.raises(FileNotFoundError, match="Contact graph not found"):
            _load_contact_graph(sj_with_boundaries)
