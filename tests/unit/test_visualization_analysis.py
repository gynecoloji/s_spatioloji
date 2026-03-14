"""Unit tests for s_spatioloji.visualization.analysis."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from s_spatioloji.spatial.point.patterns import colocalization
from s_spatioloji.spatial.point.neighborhoods import neighborhood_composition
from s_spatioloji.spatial.point.ripley import ripley_k
from s_spatioloji.spatial.point.statistics import dclf_envelope
from s_spatioloji.visualization.analysis import (
    colocalization_heatmap,
    envelope_plot,
    neighborhood_bar,
    ripley_plot,
)


class TestRipleyPlot:
    def test_returns_axes(self, sj_with_pt_clusters):
        ripley_k(sj_with_pt_clusters, n_radii=10)
        ax = ripley_plot(sj_with_pt_clusters, function="K", save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj_with_pt_clusters):
        ripley_k(sj_with_pt_clusters, n_radii=10)
        ripley_plot(sj_with_pt_clusters, function="K")
        figures_dir = sj_with_pt_clusters.config.root / "figures"
        assert (figures_dir / "ripley_k.png").exists()
        plt.close("all")

    def test_has_lines(self, sj_with_pt_clusters):
        ripley_k(sj_with_pt_clusters, n_radii=10)
        ax = ripley_plot(sj_with_pt_clusters, function="K", save=False)
        assert len(ax.get_lines()) >= 2
        plt.close("all")


class TestColocalizationHeatmap:
    def test_returns_axes(self, sj_with_pt_clusters):
        colocalization(sj_with_pt_clusters)
        ax = colocalization_heatmap(sj_with_pt_clusters, save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj_with_pt_clusters):
        colocalization(sj_with_pt_clusters)
        colocalization_heatmap(sj_with_pt_clusters)
        figures_dir = sj_with_pt_clusters.config.root / "figures"
        assert (figures_dir / "colocalization_log2_ratio.png").exists()
        plt.close("all")


class TestNeighborhoodBar:
    def test_returns_axes(self, sj_with_pt_clusters):
        neighborhood_composition(sj_with_pt_clusters)
        ax = neighborhood_bar(sj_with_pt_clusters, save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj_with_pt_clusters):
        neighborhood_composition(sj_with_pt_clusters)
        neighborhood_bar(sj_with_pt_clusters)
        figures_dir = sj_with_pt_clusters.config.root / "figures"
        assert (figures_dir / "neighborhood_bar.png").exists()
        plt.close("all")


class TestEnvelopePlot:
    def test_returns_axes(self, sj):
        dclf_envelope(sj, n_simulations=9, n_radii=5)
        ax = envelope_plot(sj, save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj):
        dclf_envelope(sj, n_simulations=9, n_radii=5)
        envelope_plot(sj)
        figures_dir = sj.config.root / "figures"
        assert (figures_dir / "envelope_pt_dclf_envelope.png").exists()
        plt.close("all")

    def test_has_fill_between(self, sj):
        dclf_envelope(sj, n_simulations=9, n_radii=5)
        ax = envelope_plot(sj, save=False)
        assert len(ax.collections) >= 1
        plt.close("all")
