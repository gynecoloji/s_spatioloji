"""Unit tests for s_spatioloji.compute._runner (conda bridge)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from s_spatioloji.compute.feature_selection import highly_variable_genes
from s_spatioloji.compute.normalize import log1p, normalize_total


@pytest.fixture()
def sj_with_hvg(sj):
    """sj with normalize → log1p → hvg pipeline complete."""
    normalize_total(sj)
    log1p(sj)
    highly_variable_genes(sj, n_top=20)
    return sj


class TestCondaBridgeMagic:
    """Test magic via conda bridge with mocked subprocess."""

    def test_conda_env_validated(self, sj_with_hvg):
        """Should raise ValueError for nonexistent conda env."""
        from s_spatioloji.compute.imputation import magic

        with patch("s_spatioloji.compute._scvi.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="base\nother_env\n", returncode=0)
            with pytest.raises(ValueError, match="not found"):
                magic(sj_with_hvg, conda_env="nonexistent_env")

    def test_subprocess_called_with_kwargs_file(self, sj_with_hvg):
        """Kwargs should be written to a file, not passed on CLI."""
        from s_spatioloji.compute.imputation import magic

        def mock_subprocess_run(cmd, **kwargs):
            mock = MagicMock()
            # Validate env check
            if "env" in cmd and "list" in cmd:
                mock.stdout = "base\ntest_env\n"
                mock.returncode = 0
                return mock
            # Actual magic call — verify kwargs-file is used
            assert "--kwargs-file" in cmd
            kwargs_idx = cmd.index("--kwargs-file") + 1
            kwargs_path = cmd[kwargs_idx]
            assert Path(kwargs_path).exists()
            content = json.loads(Path(kwargs_path).read_text())
            assert "fn" in content
            # Write fake output
            output_idx = cmd.index("--output") + 1
            output_path = cmd[output_idx]
            fake_result = np.random.rand(200, 30).astype(np.float32)
            np.savez(output_path, X=fake_result)
            mock.returncode = 0
            mock.stderr = ""
            return mock

        with patch("s_spatioloji.compute.imputation.subprocess.run", side_effect=mock_subprocess_run):
            with patch("s_spatioloji.compute._scvi.subprocess.run", side_effect=mock_subprocess_run):
                magic(sj_with_hvg, conda_env="test_env", save_expression=False)

        assert sj_with_hvg.maps.has("X_magic")

    def test_subprocess_failure_raises(self, sj_with_hvg):
        """Non-zero exit should raise RuntimeError with stderr."""
        from s_spatioloji.compute.imputation import magic

        def mock_subprocess_run(cmd, **kwargs):
            mock = MagicMock()
            if "env" in cmd and "list" in cmd:
                mock.stdout = "base\ntest_env\n"
                mock.returncode = 0
                return mock
            mock.returncode = 1
            mock.stderr = "R not found"
            return mock

        with patch("s_spatioloji.compute.imputation.subprocess.run", side_effect=mock_subprocess_run):
            with patch("s_spatioloji.compute._scvi.subprocess.run", side_effect=mock_subprocess_run):
                with pytest.raises(RuntimeError, match="R not found"):
                    magic(sj_with_hvg, conda_env="test_env")


class TestRunnerModule:
    """Test the _runner module's argument parsing and dispatch."""

    def test_handler_dispatch(self):
        from s_spatioloji.compute._runner import _HANDLERS

        assert "magic" in _HANDLERS
        assert "alra" in _HANDLERS
        assert "scvi_train" in _HANDLERS

    def test_unknown_function_exits(self, tmp_path):
        """Unknown function name should exit with error."""
        from s_spatioloji.compute._runner import _HANDLERS

        assert "nonexistent" not in _HANDLERS
