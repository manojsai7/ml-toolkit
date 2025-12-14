"""
Tests for logging utilities.
"""

import pytest
import tempfile
from pathlib import Path

from ml_toolkit.logging import ExperimentLogger


def test_experiment_logger_initialization():
    """Test ExperimentLogger initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(
            experiment_name="test_exp",
            log_dir=tmpdir,
            backend="local",
        )

        assert logger.experiment_name == "test_exp"
        assert logger.backend == "local"
        assert Path(tmpdir).exists()


def test_experiment_logger_log_params():
    """Test logging parameters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(
            experiment_name="test_exp",
            log_dir=tmpdir,
            backend="local",
        )

        params = {"learning_rate": 0.001, "batch_size": 32}
        logger.log_params(params)

        assert logger.params == params


def test_experiment_logger_log_metric():
    """Test logging metric."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(
            experiment_name="test_exp",
            log_dir=tmpdir,
            backend="local",
        )

        logger.log_metric("loss", 0.5, step=1)
        logger.log_metric("loss", 0.3, step=2)

        assert "loss" in logger.metrics
        assert len(logger.metrics["loss"]) == 2
        assert logger.metrics["loss"][0]["value"] == 0.5
        assert logger.metrics["loss"][1]["value"] == 0.3


def test_experiment_logger_log_metrics():
    """Test logging multiple metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(
            experiment_name="test_exp",
            log_dir=tmpdir,
            backend="local",
        )

        metrics = {"loss": 0.5, "accuracy": 0.9}
        logger.log_metrics(metrics, step=1)

        assert "loss" in logger.metrics
        assert "accuracy" in logger.metrics


def test_experiment_logger_save_summary():
    """Test saving experiment summary."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(
            experiment_name="test_exp",
            log_dir=tmpdir,
            backend="local",
        )

        logger.log_params({"lr": 0.001})
        logger.log_metric("loss", 0.5)
        logger.save_summary()

        summary_file = Path(tmpdir) / "test_exp_summary.json"
        assert summary_file.exists()


def test_experiment_logger_finish():
    """Test finishing experiment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ExperimentLogger(
            experiment_name="test_exp",
            log_dir=tmpdir,
            backend="local",
        )

        logger.log_params({"lr": 0.001})
        logger.log_metric("loss", 0.5)
        logger.finish()

        summary_file = Path(tmpdir) / "test_exp_summary.json"
        assert summary_file.exists()
