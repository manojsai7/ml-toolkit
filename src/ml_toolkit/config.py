"""
Configuration management utilities for ML projects
"""

import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """
    Configuration management class for ML projects.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Initial configuration dictionary
        """
        self._config = config_dict or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key (supports dot notation, e.g., "model.learning_rate")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set a configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, config_dict: Dict[str, Any]):
        """
        Update configuration with a dictionary.
        
        Args:
            config_dict: Dictionary to merge into configuration
        """
        self._deep_update(self._config, config_dict)
    
    def _deep_update(self, base: Dict, update: Dict):
        """Recursively update nested dictionaries."""
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self._config.copy()
    
    def save_yaml(self, filepath: str):
        """
        Save configuration to YAML file.
        
        Args:
            filepath: Path to save the configuration
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    def save_json(self, filepath: str):
        """
        Save configuration to JSON file.
        
        Args:
            filepath: Path to save the configuration
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self._config, f, indent=2)
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            filepath: Path to the YAML file
            
        Returns:
            Config object
        """
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'Config':
        """
        Load configuration from JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Config object
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)


def create_default_config() -> Config:
    """
    Create a default ML project configuration.
    
    Returns:
        Config object with default settings
    """
    default_config = {
        "project": {
            "name": "ml_project",
            "version": "0.1.0",
            "description": "Machine Learning Project"
        },
        "data": {
            "train_path": "data/train.csv",
            "test_path": "data/test.csv",
            "target_column": "target",
            "test_size": 0.2,
            "val_size": 0.1,
            "random_state": 42
        },
        "preprocessing": {
            "scaling_method": "standard",
            "handle_missing": "mean"
        },
        "model": {
            "type": "classifier",
            "random_state": 42
        },
        "training": {
            "cv_folds": 5,
            "metrics": ["accuracy", "precision", "recall", "f1_score"]
        },
        "output": {
            "model_path": "models/model.pkl",
            "results_path": "results/",
            "plots_path": "plots/"
        }
    }
    
    return Config(default_config)
