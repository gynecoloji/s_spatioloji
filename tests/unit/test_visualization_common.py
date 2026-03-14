"""Unit tests for s_spatioloji.visualization._common."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from s_spatioloji.visualization._common import (
    _categorical_palette,
    _continuous_cmap,
    _filter_bbox,
    _get_feature_values,
    _save_figure,
    _setup_ax,
    _subsample,
)


class TestSetupAx:
    def test_creates_figure_when_none(self):
        fig, ax = _setup_ax(None)
        assert fig is not None
        assert ax is not None
        plt.close("all")

    def test_uses_existing_ax(self):
        _, existing_ax = plt.subplots()
        fig, ax = _setup_ax(existing_ax)
        assert ax is existing_ax
        plt.close("all")

    def test_custom_figsize(self):
        fig, ax = _setup_ax(None, figsize=(12, 4))
        w, h = fig.get_size_inches()
        assert abs(w - 12) < 0.1
        assert abs(h - 4) < 0.1
        plt.close("all")


class TestSaveFigure:
    def test_saves_to_figures_dir(self, sj):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        _save_figure(fig, sj, "test_plot")
        figures_dir = sj.config.root / "figures"
        assert (figures_dir / "test_plot.png").exists()
        plt.close("all")

    def test_save_false_skips(self, sj):
        fig, ax = plt.subplots()
        _save_figure(fig, sj, "test_plot", save=False)
        figures_dir = sj.config.root / "figures"
        assert not (figures_dir / "test_plot.png").exists()
        plt.close("all")

    def test_custom_save_path(self, sj, tmp_path):
        fig, ax = plt.subplots()
        custom = tmp_path / "custom.png"
        _save_figure(fig, sj, "test_plot", save_path=custom)
        assert custom.exists()
        plt.close("all")

    def test_custom_fmt(self, sj):
        fig, ax = plt.subplots()
        _save_figure(fig, sj, "test_plot", fmt="pdf")
        figures_dir = sj.config.root / "figures"
        assert (figures_dir / "test_plot.pdf").exists()
        plt.close("all")


class TestSubsample:
    def test_no_change_when_small(self):
        df = pd.DataFrame({"x": range(10)})
        result = _subsample(df, max_cells=100)
        assert len(result) == 10

    def test_subsamples_when_large(self):
        df = pd.DataFrame({"x": range(1000)})
        result = _subsample(df, max_cells=100)
        assert len(result) == 100

    def test_reproducible(self):
        df = pd.DataFrame({"x": range(1000)})
        r1 = _subsample(df, max_cells=100)
        r2 = _subsample(df, max_cells=100)
        assert r1["x"].tolist() == r2["x"].tolist()


class TestFilterBbox:
    def test_no_filter_when_none(self):
        df = pd.DataFrame({"x": [0, 1, 2], "y": [0, 1, 2]})
        result = _filter_bbox(df)
        assert len(result) == 3

    def test_filters_x(self):
        df = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 0, 0, 0]})
        result = _filter_bbox(df, xlim=(0.5, 2.5))
        assert len(result) == 2

    def test_filters_both(self):
        df = pd.DataFrame({"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]})
        result = _filter_bbox(df, xlim=(0.5, 2.5), ylim=(0.5, 2.5))
        assert len(result) == 2


class TestGetFeatureValues:
    def test_resolves_maps_key(self, sj_with_embeddings):
        cells_df = sj_with_embeddings.cells.df.compute()
        cell_ids = cells_df["cell_id"].values
        values, is_cat = _get_feature_values(sj_with_embeddings, "leiden", cell_ids)
        assert len(values) == len(cell_ids)

    def test_resolves_gene_name(self, sj_with_embeddings):
        cells_df = sj_with_embeddings.cells.df.compute()
        cell_ids = cells_df["cell_id"].values
        values, is_cat = _get_feature_values(sj_with_embeddings, "gene_0", cell_ids)
        assert len(values) == len(cell_ids)
        assert not is_cat

    def test_unknown_raises(self, sj_with_embeddings):
        cells_df = sj_with_embeddings.cells.df.compute()
        cell_ids = cells_df["cell_id"].values
        with pytest.raises(ValueError, match="Cannot resolve"):
            _get_feature_values(sj_with_embeddings, "nonexistent_thing_xyz", cell_ids)


class TestCategoricalPalette:
    def test_returns_n_colors(self):
        colors = _categorical_palette(5)
        assert len(colors) == 5

    def test_cycles_when_large(self):
        colors = _categorical_palette(30)
        assert len(colors) == 30

    def test_returns_hex_strings(self):
        colors = _categorical_palette(3)
        for c in colors:
            assert c.startswith("#")


class TestContinuousCmap:
    def test_returns_colormap(self):
        cmap = _continuous_cmap("magma")
        assert cmap is not None

    def test_custom_name(self):
        cmap = _continuous_cmap("viridis")
        assert cmap.name == "viridis"
