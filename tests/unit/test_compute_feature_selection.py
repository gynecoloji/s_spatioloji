"""Unit tests for s_spatioloji.compute.feature_selection."""

from __future__ import annotations

from s_spatioloji.compute.feature_selection import highly_variable_genes
from s_spatioloji.compute.normalize import log1p, normalize_total


class TestHighlyVariableGenes:
    def test_returns_key(self, sj):
        normalize_total(sj)
        log1p(sj)
        assert highly_variable_genes(sj) == "hvg"

    def test_output_written(self, sj):
        normalize_total(sj)
        log1p(sj)
        highly_variable_genes(sj)
        assert sj.maps.has("hvg")

    def test_hvg_table_columns(self, sj):
        normalize_total(sj)
        log1p(sj)
        highly_variable_genes(sj)
        df = sj.maps["hvg"].compute()
        assert list(df.columns) == ["gene", "highly_variable", "mean", "variance", "dispersion"]

    def test_n_top_respected(self, sj):
        normalize_total(sj)
        log1p(sj)
        highly_variable_genes(sj, n_top=10)
        df = sj.maps["hvg"].compute()
        assert df["highly_variable"].sum() == 10

    def test_n_top_clamped_to_n_genes(self, sj):
        normalize_total(sj)
        log1p(sj)
        highly_variable_genes(sj, n_top=5000)  # more than 30 genes
        df = sj.maps["hvg"].compute()
        assert df["highly_variable"].sum() == 30  # clamped to n_genes

    def test_all_genes_present(self, sj):
        normalize_total(sj)
        log1p(sj)
        highly_variable_genes(sj)
        df = sj.maps["hvg"].compute()
        assert len(df) == 30
        assert df["gene"].iloc[0] == "gene_0"

    def test_force_false_skips(self, sj):
        normalize_total(sj)
        log1p(sj)
        highly_variable_genes(sj, n_top=10)
        highly_variable_genes(sj, n_top=5, force=False)
        # Should still have 10 HVGs (not overwritten)
        df = sj.maps["hvg"].compute()
        assert df["highly_variable"].sum() == 10
