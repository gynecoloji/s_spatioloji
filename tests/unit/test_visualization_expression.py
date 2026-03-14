"""Unit tests for s_spatioloji.visualization.expression."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from s_spatioloji.visualization.expression import dotplot, heatmap, violin


GENES = ["gene_0", "gene_1", "gene_2"]


class TestHeatmap:
    def test_returns_axes(self, sj_with_embeddings):
        ax = heatmap(sj_with_embeddings, GENES, save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj_with_embeddings):
        heatmap(sj_with_embeddings, GENES)
        figures_dir = sj_with_embeddings.config.root / "figures"
        assert (figures_dir / "heatmap_leiden.png").exists()
        plt.close("all")

    def test_standardize_switches_cmap(self, sj_with_embeddings):
        ax = heatmap(sj_with_embeddings, GENES, standardize=True, save=False)
        # When standardize=True and cmap defaults to "magma", it should switch to RdBu_r
        images = ax.get_images()
        assert len(images) > 0
        assert images[0].cmap.name == "RdBu_r"
        plt.close("all")

    def test_no_standardize_keeps_cmap(self, sj_with_embeddings):
        ax = heatmap(sj_with_embeddings, GENES, standardize=False, save=False)
        images = ax.get_images()
        assert images[0].cmap.name == "magma"
        plt.close("all")

    def test_existing_ax(self, sj_with_embeddings):
        _, existing_ax = plt.subplots()
        ax = heatmap(sj_with_embeddings, GENES, ax=existing_ax, save=False)
        assert ax is existing_ax
        plt.close("all")


class TestViolin:
    def test_returns_figure(self, sj_with_embeddings):
        fig = violin(sj_with_embeddings, GENES, save=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close("all")

    def test_saves_figure(self, sj_with_embeddings):
        violin(sj_with_embeddings, GENES)
        figures_dir = sj_with_embeddings.config.root / "figures"
        assert (figures_dir / "violin_leiden.png").exists()
        plt.close("all")

    def test_multi_gene_grid(self, sj_with_embeddings):
        many_genes = [f"gene_{i}" for i in range(6)]
        fig = violin(sj_with_embeddings, many_genes, ncols=3, save=False)
        axes = [a for a in fig.axes if a.get_visible()]
        assert len(axes) == 6
        plt.close("all")


class TestDotplot:
    def test_returns_axes(self, sj_with_embeddings):
        ax = dotplot(sj_with_embeddings, GENES, save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj_with_embeddings):
        dotplot(sj_with_embeddings, GENES)
        figures_dir = sj_with_embeddings.config.root / "figures"
        assert (figures_dir / "dotplot_leiden.png").exists()
        plt.close("all")

    def test_existing_ax(self, sj_with_embeddings):
        _, existing_ax = plt.subplots()
        ax = dotplot(sj_with_embeddings, GENES, ax=existing_ax, save=False)
        assert ax is existing_ax
        plt.close("all")
