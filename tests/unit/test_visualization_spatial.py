"""Unit tests for s_spatioloji.visualization.spatial."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from s_spatioloji.visualization.spatial import (
    spatial_expression,
    spatial_polygons,
    spatial_scatter,
)


class TestSpatialScatter:
    def test_returns_axes(self, sj_with_embeddings):
        ax = spatial_scatter(sj_with_embeddings, color_by="leiden", save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj_with_embeddings):
        spatial_scatter(sj_with_embeddings, color_by="leiden")
        figures_dir = sj_with_embeddings.config.root / "figures"
        assert (figures_dir / "spatial_scatter_leiden.png").exists()
        plt.close("all")

    def test_save_false_skips(self, sj_with_embeddings):
        spatial_scatter(sj_with_embeddings, color_by="leiden", save=False)
        figures_dir = sj_with_embeddings.config.root / "figures"
        assert not figures_dir.exists() or not (figures_dir / "spatial_scatter_leiden.png").exists()
        plt.close("all")

    def test_existing_ax(self, sj_with_embeddings):
        _, existing_ax = plt.subplots()
        ax = spatial_scatter(sj_with_embeddings, color_by="leiden", ax=existing_ax, save=False)
        assert ax is existing_ax
        plt.close("all")

    def test_bbox_filter(self, sj_with_embeddings):
        ax = spatial_scatter(
            sj_with_embeddings, color_by="leiden", xlim=(0.0, 200.0), ylim=(0.0, 200.0), save=False
        )
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_continuous_color(self, sj_with_embeddings):
        ax = spatial_scatter(sj_with_embeddings, color_by="gene_0", save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_image_overlay(self, sj_with_embeddings, tmp_path):
        try:
            import tifffile
        except ImportError:
            pytest.skip("tifffile not installed")
        # Create a small synthetic TIF image
        img = np.random.default_rng(0).integers(0, 255, (100, 100), dtype=np.uint8)
        tif_path = tmp_path / "test_image.tif"
        tifffile.imwrite(str(tif_path), img)

        ax = spatial_scatter(
            sj_with_embeddings, color_by="leiden", image_path=tif_path, save=False
        )
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")


class TestSpatialPolygons:
    def test_returns_axes(self, sj_with_clusters):
        ax = spatial_polygons(sj_with_clusters, color_by="leiden", save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj_with_clusters):
        spatial_polygons(sj_with_clusters, color_by="leiden")
        figures_dir = sj_with_clusters.config.root / "figures"
        assert (figures_dir / "spatial_polygons_leiden.png").exists()
        plt.close("all")

    def test_existing_ax(self, sj_with_clusters):
        _, existing_ax = plt.subplots()
        ax = spatial_polygons(sj_with_clusters, color_by="leiden", ax=existing_ax, save=False)
        assert ax is existing_ax
        plt.close("all")


class TestSpatialExpression:
    def test_returns_axes_scatter(self, sj_with_embeddings):
        ax = spatial_expression(sj_with_embeddings, gene="gene_0", mode="scatter", save=False)
        assert isinstance(ax, matplotlib.axes.Axes)
        plt.close("all")

    def test_saves_figure(self, sj_with_embeddings):
        spatial_expression(sj_with_embeddings, gene="gene_0", mode="scatter")
        figures_dir = sj_with_embeddings.config.root / "figures"
        assert (figures_dir / "spatial_scatter_gene_0.png").exists()
        plt.close("all")

    def test_invalid_gene_raises(self, sj_with_embeddings):
        with pytest.raises(ValueError, match="not found in expression matrix"):
            spatial_expression(sj_with_embeddings, gene="NONEXISTENT_GENE", save=False)
        plt.close("all")
