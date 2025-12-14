"""
Tests for configuration utilities.
"""

import pytest
import tempfile
from pathlib import Path

from ml_toolkit.config import (
    BaseConfig,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    ConfigManager,
)


def test_base_config():
    """Test BaseConfig."""
    config = BaseConfig(seed=123, device="cuda")
    assert config.seed == 123
    assert config.device == "cuda"


def test_data_config():
    """Test DataConfig."""
    config = DataConfig(batch_size=64, num_workers=8)
    assert config.batch_size == 64
    assert config.num_workers == 8
    assert config.seed == 42  # Default value


def test_model_config():
    """Test ModelConfig."""
    config = ModelConfig(model_name="resnet50", num_classes=100)
    assert config.model_name == "resnet50"
    assert config.num_classes == 100


def test_training_config():
    """Test TrainingConfig."""
    config = TrainingConfig(epochs=50, learning_rate=0.001)
    assert config.epochs == 50
    assert config.learning_rate == 0.001


def test_config_manager_from_dict():
    """Test creating config from dict."""
    config_dict = {"seed": 42, "device": "cpu"}
    config = ConfigManager.from_dict(config_dict)

    assert config.seed == 42
    assert config.device == "cpu"


def test_config_manager_to_dict():
    """Test converting config to dict."""
    config_dict = {"seed": 42, "device": "cpu"}
    config = ConfigManager.from_dict(config_dict)
    result = ConfigManager.to_dict(config)

    assert result == config_dict


def test_config_manager_merge():
    """Test merging configs."""
    config1 = ConfigManager.from_dict({"seed": 42, "device": "cpu"})
    config2 = ConfigManager.from_dict({"device": "cuda", "epochs": 10})

    merged = ConfigManager.merge_configs(config1, config2)
    result = ConfigManager.to_dict(merged)

    assert result["seed"] == 42
    assert result["device"] == "cuda"  # config2 overrides
    assert result["epochs"] == 10


def test_config_manager_save_load():
    """Test saving and loading config."""
    config_dict = {"seed": 42, "device": "cpu", "epochs": 10}
    config = ConfigManager.from_dict(config_dict)

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "config.yaml"

        manager = ConfigManager()
        manager.save(config, str(filepath))

        loaded_config = manager.load_yaml(str(filepath))
        result = ConfigManager.to_dict(loaded_config)

        assert result == config_dict
