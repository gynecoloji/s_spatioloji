"""Unit tests for s_spatioloji.compute.normalize."""

from __future__ import annotations

import numpy as np

from s_spatioloji.compute.normalize import log1p, normalize_total, pearson_residuals, scale
from s_spatioloji.data.expression import ExpressionStore


def _load_result(sj, key):
    """Load a compute result, handling both Parquet and Zarr outputs."""
    result = sj.maps[key]
    if isinstance(result, ExpressionStore):
        return result.to_dask().compute(), result.gene_names
    # dask DataFrame (Parquet)
    df = result.compute()
    gene_cols = [c for c in df.columns if c != "cell_id"]
    return df[gene_cols].values, gene_cols


class TestNormalizeTotal:
    def test_returns_key(self, sj):
        assert normalize_total(sj) == "X_norm"

    def test_output_written(self, sj):
        normalize_total(sj)
        assert sj.maps.has("X_norm")

    def test_shape(self, sj):
        normalize_total(sj)
        matrix, genes = _load_result(sj, "X_norm")
        assert matrix.shape == (200, 30)

    def test_row_sums(self, sj):
        normalize_total(sj, target_sum=1e4)
        matrix, _ = _load_result(sj, "X_norm")
        row_sums = matrix.sum(axis=1)
        nonzero = row_sums[row_sums > 0]
        np.testing.assert_allclose(nonzero, 1e4, rtol=1e-5)

    def test_force_false_skips(self, sj):
        normalize_total(sj)
        matrix1, _ = _load_result(sj, "X_norm")
        original_val = matrix1[0, 0]
        normalize_total(sj, target_sum=999, force=False)
        matrix2, _ = _load_result(sj, "X_norm")
        assert matrix2[0, 0] == original_val


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
        norm_matrix, _ = _load_result(sj, "X_norm")
        log1p(sj)
        log_matrix, _ = _load_result(sj, "X_log1p")
        expected = np.log1p(norm_matrix.astype(np.float32))
        np.testing.assert_allclose(log_matrix, expected, rtol=1e-5)

    def test_force_false_skips(self, sj):
        normalize_total(sj)
        log1p(sj)
        log1p(sj, force=False)


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
        # 20 HVGs + cell_id = 21 columns (Parquet output for HVG subset)
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
        matrix, _ = _load_result(sj, "X_residuals")
        assert matrix.shape == (200, 30)

    def test_values_finite(self, sj):
        pearson_residuals(sj)
        matrix, _ = _load_result(sj, "X_residuals")
        assert np.all(np.isfinite(matrix))

    def test_clipped(self, sj):
        pearson_residuals(sj)
        matrix, _ = _load_result(sj, "X_residuals")
        clip_val = np.sqrt(200)
        assert matrix.max() <= clip_val + 1e-5
        assert matrix.min() >= -clip_val - 1e-5
