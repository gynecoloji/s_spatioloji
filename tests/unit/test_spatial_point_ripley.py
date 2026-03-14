"""Unit tests for s_spatioloji.spatial.point.ripley."""

from __future__ import annotations

import numpy as np

from s_spatioloji.spatial.point.ripley import ripley_f, ripley_g, ripley_k, ripley_l


class TestRipleyK:
    def test_returns_key(self, sj):
        assert ripley_k(sj) == "pt_ripley_k"

    def test_output_written(self, sj):
        ripley_k(sj)
        assert sj.maps.has("pt_ripley_k")

    def test_columns(self, sj):
        ripley_k(sj)
        df = sj.maps["pt_ripley_k"].compute()
        assert list(df.columns) == ["cluster", "r", "K", "K_theo"]

    def test_cluster_all_when_no_key(self, sj):
        ripley_k(sj)
        df = sj.maps["pt_ripley_k"].compute()
        assert (df["cluster"] == "all").all()

    def test_n_radii(self, sj):
        ripley_k(sj, n_radii=20)
        df = sj.maps["pt_ripley_k"].compute()
        assert len(df) == 20

    def test_K_theo_is_pi_r_sq(self, sj):
        ripley_k(sj, n_radii=10)
        df = sj.maps["pt_ripley_k"].compute()
        np.testing.assert_allclose(df["K_theo"].values, np.pi * df["r"].values ** 2)

    def test_force_false_skips(self, sj):
        ripley_k(sj)
        ripley_k(sj, force=False)

    def test_per_cluster(self, sj_with_pt_clusters):
        ripley_k(sj_with_pt_clusters, cluster_key="leiden", n_radii=10)
        df = sj_with_pt_clusters.maps["pt_ripley_k"].compute()
        assert len(df["cluster"].unique()) == 4


class TestRipleyL:
    def test_returns_key(self, sj):
        assert ripley_l(sj) == "pt_ripley_l"

    def test_columns(self, sj):
        ripley_l(sj)
        df = sj.maps["pt_ripley_l"].compute()
        assert list(df.columns) == ["cluster", "r", "L"]

    def test_force_false_skips(self, sj):
        ripley_l(sj)
        ripley_l(sj, force=False)


class TestRipleyG:
    def test_returns_key(self, sj):
        assert ripley_g(sj) == "pt_ripley_g"

    def test_columns(self, sj):
        ripley_g(sj)
        df = sj.maps["pt_ripley_g"].compute()
        assert list(df.columns) == ["cluster", "r", "G", "G_theo"]

    def test_G_range(self, sj):
        ripley_g(sj)
        df = sj.maps["pt_ripley_g"].compute()
        assert (df["G"] >= 0).all()
        assert (df["G"] <= 1.0).all()

    def test_force_false_skips(self, sj):
        ripley_g(sj)
        ripley_g(sj, force=False)


class TestRipleyF:
    def test_returns_key(self, sj):
        assert ripley_f(sj) == "pt_ripley_f"

    def test_columns(self, sj):
        ripley_f(sj)
        df = sj.maps["pt_ripley_f"].compute()
        assert list(df.columns) == ["cluster", "r", "F", "F_theo"]

    def test_F_range(self, sj):
        ripley_f(sj)
        df = sj.maps["pt_ripley_f"].compute()
        assert (df["F"] >= 0).all()
        assert (df["F"] <= 1.0).all()

    def test_force_false_skips(self, sj):
        ripley_f(sj)
        ripley_f(sj, force=False)
