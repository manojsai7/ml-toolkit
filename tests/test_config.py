"""
Tests for configuration utilities
"""

import pytest
import sys
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_toolkit.config import Config, create_default_config


def test_config_get_set():
    """Test getting and setting config values."""
    config = Config()
    
    config.set("model.learning_rate", 0.001)
    config.set("data.batch_size", 32)
    
    assert config.get("model.learning_rate") == 0.001
    assert config.get("data.batch_size") == 32
    assert config.get("nonexistent", "default") == "default"


def test_config_nested_keys():
    """Test nested key access."""
    config = Config({
        "model": {
            "params": {
                "lr": 0.001
            }
        }
    })
    
    assert config.get("model.params.lr") == 0.001


def test_config_update():
    """Test updating configuration."""
    config = Config({"a": 1, "b": {"c": 2}})
    config.update({"b": {"d": 3}, "e": 4})
    
    assert config.get("a") == 1
    assert config.get("b.c") == 2
    assert config.get("b.d") == 3
    assert config.get("e") == 4


def test_config_save_load_yaml():
    """Test saving and loading YAML config."""
    config = Config({
        "model": {"type": "classifier"},
        "data": {"path": "data.csv"}
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "config.yaml")
        config.save_yaml(filepath)
        
        loaded_config = Config.from_yaml(filepath)
        
        assert loaded_config.get("model.type") == "classifier"
        assert loaded_config.get("data.path") == "data.csv"


def test_config_save_load_json():
    """Test saving and loading JSON config."""
    config = Config({
        "model": {"type": "regressor"},
        "data": {"path": "data.csv"}
    })
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "config.json")
        config.save_json(filepath)
        
        loaded_config = Config.from_json(filepath)
        
        assert loaded_config.get("model.type") == "regressor"
        assert loaded_config.get("data.path") == "data.csv"


def test_create_default_config():
    """Test creating default configuration."""
    config = create_default_config()
    
    assert config.get("project.name") is not None
    assert config.get("data.test_size") is not None
    assert config.get("model.random_state") is not None
