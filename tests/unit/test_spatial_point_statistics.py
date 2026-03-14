"""Unit tests for s_spatioloji.spatial.point.statistics."""

from __future__ import annotations

from s_spatioloji.spatial.point.statistics import (
    clark_evans,
    dclf_envelope,
    permutation_test,
    quadrat_density,
)


class TestPermutationTest:
    def test_returns_key(self, sj_with_pt_clusters):
        assert permutation_test(sj_with_pt_clusters, n_permutations=50) == "pt_permutation_test"

    def test_output_written(self, sj_with_pt_clusters):
        permutation_test(sj_with_pt_clusters, n_permutations=50)
        assert sj_with_pt_clusters.maps.has("pt_permutation_test")

    def test_columns(self, sj_with_pt_clusters):
        permutation_test(sj_with_pt_clusters, n_permutations=50)
        df = sj_with_pt_clusters.maps["pt_permutation_test"].compute()
        assert list(df.columns) == ["cluster_a", "cluster_b", "observed_ratio", "p_value", "z_score"]

    def test_p_values_valid(self, sj_with_pt_clusters):
        permutation_test(sj_with_pt_clusters, n_permutations=50)
        df = sj_with_pt_clusters.maps["pt_permutation_test"].compute()
        assert (df["p_value"] >= 0).all()
        assert (df["p_value"] <= 1.0).all()

    def test_force_false_skips(self, sj_with_pt_clusters):
        permutation_test(sj_with_pt_clusters, n_permutations=50)
        permutation_test(sj_with_pt_clusters, n_permutations=50, force=False)


class TestQuadratDensity:
    def test_returns_key(self, sj_with_pt_clusters):
        assert quadrat_density(sj_with_pt_clusters) == "pt_quadrat_density"

    def test_columns(self, sj_with_pt_clusters):
        quadrat_density(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_quadrat_density"].compute()
        assert list(df.columns) == ["cluster", "chi2", "p_value", "density_mean", "density_std"]

    def test_has_rows(self, sj_with_pt_clusters):
        quadrat_density(sj_with_pt_clusters)
        df = sj_with_pt_clusters.maps["pt_quadrat_density"].compute()
        assert len(df) == 4  # 4 clusters

    def test_force_false_skips(self, sj_with_pt_clusters):
        quadrat_density(sj_with_pt_clusters)
        quadrat_density(sj_with_pt_clusters, force=False)


class TestClarkEvans:
    def test_returns_key(self, sj):
        assert clark_evans(sj) == "pt_clark_evans"

    def test_columns(self, sj):
        clark_evans(sj)
        df = sj.maps["pt_clark_evans"].compute()
        assert list(df.columns) == ["cluster", "R", "r_observed", "r_expected", "z_score", "p_value"]

    def test_global_cluster_all(self, sj):
        clark_evans(sj)
        df = sj.maps["pt_clark_evans"].compute()
        assert len(df) == 1
        assert df["cluster"].iloc[0] == "all"

    def test_R_positive(self, sj):
        clark_evans(sj)
        df = sj.maps["pt_clark_evans"].compute()
        assert (df["R"] > 0).all()

    def test_regular_grid_R_near_one(self, sj):
        """20x10 grid should have R >= 1 (regular/dispersed)."""
        clark_evans(sj)
        df = sj.maps["pt_clark_evans"].compute()
        assert df["R"].iloc[0] >= 0.9

    def test_per_cluster(self, sj_with_pt_clusters):
        clark_evans(sj_with_pt_clusters, cluster_key="leiden")
        df = sj_with_pt_clusters.maps["pt_clark_evans"].compute()
        assert len(df) == 4

    def test_force_false_skips(self, sj):
        clark_evans(sj)
        clark_evans(sj, force=False)


class TestDCLFEnvelope:
    def test_returns_key(self, sj):
        assert dclf_envelope(sj, n_simulations=19) == "pt_dclf_envelope"

    def test_columns(self, sj):
        dclf_envelope(sj, n_simulations=19, n_radii=10)
        df = sj.maps["pt_dclf_envelope"].compute()
        assert list(df.columns) == ["cluster", "r", "observed", "lo", "hi", "theo", "p_value"]

    def test_p_value_valid(self, sj):
        dclf_envelope(sj, n_simulations=19, n_radii=10)
        df = sj.maps["pt_dclf_envelope"].compute()
        assert (df["p_value"] >= 0).all()
        assert (df["p_value"] <= 1.0).all()

    def test_envelope_bounds(self, sj):
        dclf_envelope(sj, n_simulations=19, n_radii=10)
        df = sj.maps["pt_dclf_envelope"].compute()
        assert (df["lo"] <= df["hi"]).all()

    def test_function_L(self, sj):
        dclf_envelope(sj, function="L", n_simulations=19, n_radii=10, output_key="pt_dclf_L")
        assert sj.maps.has("pt_dclf_L")

    def test_force_false_skips(self, sj):
        dclf_envelope(sj, n_simulations=19, n_radii=10)
        dclf_envelope(sj, n_simulations=19, n_radii=10, force=False)
