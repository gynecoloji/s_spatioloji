"""Unit tests for s_spatioloji.compute.normalize."""

from __future__ import annotations

import numpy as np

from s_spatioloji.compute.normalize import log1p, normalize_total, pearson_residuals, scale


class TestNormalizeTotal:
    def test_returns_key(self, sj):
        assert normalize_total(sj) == "X_norm"

    def test_output_written(self, sj):
        normalize_total(sj)
        assert sj.maps.has("X_norm")

    def test_shape(self, sj):
        normalize_total(sj)
        df = sj.maps["X_norm"].compute()
        assert df.shape == (200, 31)  # 30 genes + cell_id

    def test_cell_id_column(self, sj):
        normalize_total(sj)
        df = sj.maps["X_norm"].compute()
        assert df.columns[0] == "cell_id"
        assert df["cell_id"].iloc[0] == "cell_0"

    def test_row_sums(self, sj):
        normalize_total(sj, target_sum=1e4)
        df = sj.maps["X_norm"].compute()
        gene_cols = [c for c in df.columns if c != "cell_id"]
        row_sums = df[gene_cols].values.sum(axis=1)
        # All non-zero rows should sum to target_sum
        nonzero = row_sums[row_sums > 0]
        np.testing.assert_allclose(nonzero, 1e4, rtol=1e-5)

    def test_force_false_skips(self, sj):
        normalize_total(sj)
        # Modify the output to detect if it gets overwritten
        df = sj.maps["X_norm"].compute()
        original_val = df.iloc[0, 1]
        normalize_total(sj, target_sum=999, force=False)
        df2 = sj.maps["X_norm"].compute()
        assert df2.iloc[0, 1] == original_val  # not overwritten


class TestLog1p:
    def test_returns_key(self, sj):
        normalize_total(sj)
        assert log1p(sj) == "X_log1p"

    def test_output_written(self, sj):
        normalize_total(sj)
        log1p(sj)
        assert sj.maps.has("X_log1p")

    def test_values_are_log_transformed(self, sj):
        normalize_total(sj)
        norm_df = sj.maps["X_norm"].compute()
        log1p(sj)
        log_df = sj.maps["X_log1p"].compute()
        gene_cols = [c for c in norm_df.columns if c != "cell_id"]
        expected = np.log1p(norm_df[gene_cols].values.astype(np.float32))
        np.testing.assert_allclose(log_df[gene_cols].values, expected, rtol=1e-5)

    def test_force_false_skips(self, sj):
        normalize_total(sj)
        log1p(sj)
        log1p(sj, force=False)  # should not raise


class TestScale:
    def test_returns_key(self, sj):
        normalize_total(sj)
        log1p(sj)
        from s_spatioloji.compute.feature_selection import highly_variable_genes

        highly_variable_genes(sj, n_top=20)
        assert scale(sj) == "X_scaled"

    def test_hvg_subset(self, sj):
        normalize_total(sj)
        log1p(sj)
        from s_spatioloji.compute.feature_selection import highly_variable_genes

        highly_variable_genes(sj, n_top=20)
        scale(sj, hvg=True)
        df = sj.maps["X_scaled"].compute()
        # 20 HVGs + cell_id = 21 columns
        assert df.shape[1] == 21
        assert df.shape[0] == 200

    def test_zero_centred(self, sj):
        normalize_total(sj)
        log1p(sj)
        from s_spatioloji.compute.feature_selection import highly_variable_genes

        highly_variable_genes(sj, n_top=20)
        scale(sj)
        df = sj.maps["X_scaled"].compute()
        gene_cols = [c for c in df.columns if c != "cell_id"]
        means = df[gene_cols].values.mean(axis=0)
        np.testing.assert_allclose(means, 0, atol=1e-5)

    def test_clipped(self, sj):
        normalize_total(sj)
        log1p(sj)
        from s_spatioloji.compute.feature_selection import highly_variable_genes

        highly_variable_genes(sj, n_top=20)
        scale(sj, max_value=5.0)
        df = sj.maps["X_scaled"].compute()
        gene_cols = [c for c in df.columns if c != "cell_id"]
        assert df[gene_cols].values.max() <= 5.0
        assert df[gene_cols].values.min() >= -5.0


class TestPearsonResiduals:
    def test_returns_key(self, sj):
        assert pearson_residuals(sj) == "X_residuals"

    def test_shape(self, sj):
        pearson_residuals(sj)
        df = sj.maps["X_residuals"].compute()
        assert df.shape == (200, 31)  # 30 genes + cell_id

    def test_values_finite(self, sj):
        pearson_residuals(sj)
        df = sj.maps["X_residuals"].compute()
        gene_cols = [c for c in df.columns if c != "cell_id"]
        assert np.all(np.isfinite(df[gene_cols].values))

    def test_clipped(self, sj):
        pearson_residuals(sj)
        df = sj.maps["X_residuals"].compute()
        gene_cols = [c for c in df.columns if c != "cell_id"]
        clip_val = np.sqrt(200)
        assert df[gene_cols].values.max() <= clip_val + 1e-5
        assert df[gene_cols].values.min() >= -clip_val - 1e-5
