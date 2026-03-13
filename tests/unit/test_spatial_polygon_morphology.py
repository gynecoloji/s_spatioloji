"""Unit tests for s_spatioloji.spatial.polygon.morphology."""

from __future__ import annotations

from s_spatioloji.spatial.polygon.morphology import cell_morphology

EXPECTED_COLUMNS = [
    "cell_id",
    "area",
    "perimeter",
    "centroid_x",
    "centroid_y",
    "circularity",
    "elongation",
    "solidity",
    "eccentricity",
    "aspect_ratio",
    "fractal_dimension",
    "vertex_count",
    "convexity_defects",
    "rectangularity",
]


class TestCellMorphology:
    def test_returns_key(self, sj_with_boundaries):
        assert cell_morphology(sj_with_boundaries) == "morphology"

    def test_output_written(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        assert sj_with_boundaries.maps.has("morphology")

    def test_columns(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        df = sj_with_boundaries.maps["morphology"].compute()
        assert list(df.columns) == EXPECTED_COLUMNS

    def test_shape(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        df = sj_with_boundaries.maps["morphology"].compute()
        assert df.shape == (200, 14)  # 200 cells, 13 metrics + cell_id

    def test_area_positive(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        df = sj_with_boundaries.maps["morphology"].compute()
        assert (df["area"] > 0).all()

    def test_perimeter_positive(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        df = sj_with_boundaries.maps["morphology"].compute()
        assert (df["perimeter"] > 0).all()

    def test_circularity_range(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        df = sj_with_boundaries.maps["morphology"].compute()
        assert (df["circularity"] > 0).all()
        assert (df["circularity"] <= 1.0 + 1e-6).all()

    def test_solidity_range(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        df = sj_with_boundaries.maps["morphology"].compute()
        assert (df["solidity"] > 0).all()
        assert (df["solidity"] <= 1.0 + 1e-6).all()

    def test_elongation_range(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        df = sj_with_boundaries.maps["morphology"].compute()
        assert (df["elongation"] >= 0).all()
        assert (df["elongation"] < 1.0).all()

    def test_vertex_count_positive(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        df = sj_with_boundaries.maps["morphology"].compute()
        assert (df["vertex_count"] >= 3).all()

    def test_rectangularity_range(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        df = sj_with_boundaries.maps["morphology"].compute()
        assert (df["rectangularity"] > 0).all()
        assert (df["rectangularity"] <= 1.0 + 1e-6).all()

    def test_force_false_skips(self, sj_with_boundaries):
        cell_morphology(sj_with_boundaries)
        cell_morphology(sj_with_boundaries, force=False)

    def test_custom_output_key(self, sj_with_boundaries):
        assert cell_morphology(sj_with_boundaries, output_key="morph2") == "morph2"
        assert sj_with_boundaries.maps.has("morph2")
