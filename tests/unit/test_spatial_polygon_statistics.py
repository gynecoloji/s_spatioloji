"""Unit tests for s_spatioloji.spatial.polygon.statistics."""

from __future__ import annotations

import numpy as np

from s_spatioloji.spatial.polygon.statistics import permutation_test, quadrat_density


class TestPermutationTest:
    def test_returns_key(self, sj_with_clusters):
        assert permutation_test(sj_with_clusters, n_permutations=50) == "permutation_test"

    def test_output_written(self, sj_with_clusters):
        permutation_test(sj_with_clusters, n_permutations=50)
        assert sj_with_clusters.maps.has("permutation_test")

    def test_columns(self, sj_with_clusters):
        permutation_test(sj_with_clusters, n_permutations=50)
        df = sj_with_clusters.maps["permutation_test"].compute()
        assert list(df.columns) == [
            "cluster_a",
            "cluster_b",
            "observed_ratio",
            "p_value",
            "z_score",
        ]

    def test_has_rows(self, sj_with_clusters):
        permutation_test(sj_with_clusters, n_permutations=50)
        df = sj_with_clusters.maps["permutation_test"].compute()
        assert len(df) > 0

    def test_p_values_range(self, sj_with_clusters):
        permutation_test(sj_with_clusters, n_permutations=50)
        df = sj_with_clusters.maps["permutation_test"].compute()
        assert (df["p_value"] >= 0).all()
        assert (df["p_value"] <= 1.0).all()

    def test_reproducible(self, sj_with_clusters):
        permutation_test(sj_with_clusters, n_permutations=50, output_key="pt1")
        permutation_test(sj_with_clusters, n_permutations=50, output_key="pt2")
        df1 = sj_with_clusters.maps["pt1"].compute()
        df2 = sj_with_clusters.maps["pt2"].compute()
        np.testing.assert_array_equal(df1["p_value"].values, df2["p_value"].values)

    def test_force_false_skips(self, sj_with_clusters):
        permutation_test(sj_with_clusters, n_permutations=50)
        permutation_test(sj_with_clusters, n_permutations=50, force=False)


class TestQuadratDensity:
    def test_returns_key(self, sj_with_clusters):
        assert quadrat_density(sj_with_clusters) == "quadrat_density"

    def test_output_written(self, sj_with_clusters):
        quadrat_density(sj_with_clusters)
        assert sj_with_clusters.maps.has("quadrat_density")

    def test_columns(self, sj_with_clusters):
        quadrat_density(sj_with_clusters)
        df = sj_with_clusters.maps["quadrat_density"].compute()
        assert list(df.columns) == ["cluster", "chi2", "p_value", "density_mean", "density_std"]

    def test_has_rows(self, sj_with_clusters):
        quadrat_density(sj_with_clusters)
        df = sj_with_clusters.maps["quadrat_density"].compute()
        assert len(df) == 4

    def test_chi2_nonneg(self, sj_with_clusters):
        quadrat_density(sj_with_clusters)
        df = sj_with_clusters.maps["quadrat_density"].compute()
        assert (df["chi2"] >= 0).all()

    def test_p_values_range(self, sj_with_clusters):
        quadrat_density(sj_with_clusters)
        df = sj_with_clusters.maps["quadrat_density"].compute()
        assert (df["p_value"] >= 0).all()
        assert (df["p_value"] <= 1.0).all()

    def test_no_graph_needed(self, sj_with_clusters):
        """quadrat_density does NOT depend on contact graph."""
        quadrat_density(sj_with_clusters)

    def test_force_false_skips(self, sj_with_clusters):
        quadrat_density(sj_with_clusters)
        quadrat_density(sj_with_clusters, force=False)
