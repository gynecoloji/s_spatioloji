"""Unit tests for s_spatioloji.visualization.embedding."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from s_spatioloji.visualization.embedding import scatter


class TestScatter:
    def test_returns_axes(self, sj_with_embeddings):
        ax = scatter(sj_with_embeddings, basis="X_umap", color_by="leiden", save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj_with_embeddings):
        scatter(sj_with_embeddings, basis="X_umap", color_by="leiden")
        figures_dir = sj_with_embeddings.config.root / "figures"
        assert (figures_dir / "scatter_X_umap_leiden.png").exists()
        plt.close("all")

    def test_save_false_skips(self, sj_with_embeddings):
        scatter(sj_with_embeddings, basis="X_umap", color_by="leiden", save=False)
        figures_dir = sj_with_embeddings.config.root / "figures"
        assert not figures_dir.exists() or not (figures_dir / "scatter_X_umap_leiden.png").exists()
        plt.close("all")

    def test_existing_ax(self, sj_with_embeddings):
        _, existing_ax = plt.subplots()
        ax = scatter(sj_with_embeddings, basis="X_umap", color_by="leiden", ax=existing_ax, save=False)
        assert ax is existing_ax
        plt.close("all")

    def test_continuous_color(self, sj_with_embeddings):
        ax = scatter(sj_with_embeddings, basis="X_umap", color_by="gene_0", save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_custom_figsize(self, sj_with_embeddings):
        ax = scatter(sj_with_embeddings, basis="X_umap", color_by="leiden", figsize=(12, 4), save=False)
        fig = ax.get_figure()
        w, h = fig.get_size_inches()
        assert abs(w - 12) < 0.1
        plt.close("all")
