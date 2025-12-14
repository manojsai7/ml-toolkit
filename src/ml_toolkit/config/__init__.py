"""
Configuration management utilities using Hydra and Pydantic.
"""

from typing import Any, Dict, Optional
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    """Base configuration model using Pydantic."""

    seed: int = Field(default=42, description="Random seed for reproducibility")
    device: str = Field(default="cpu", description="Device to use (cpu/cuda)")
    log_dir: str = Field(default="logs", description="Directory for logs")


class DataConfig(BaseConfig):
    """Data configuration."""

    data_dir: str = Field(default="data", description="Data directory")
    batch_size: int = Field(default=32, description="Batch size")
    num_workers: int = Field(default=4, description="Number of data loader workers")
    train_split: float = Field(default=0.8, description="Train split ratio")


class ModelConfig(BaseConfig):
    """Model configuration."""

    model_name: str = Field(default="resnet50", description="Model architecture")
    num_classes: int = Field(default=10, description="Number of output classes")
    pretrained: bool = Field(default=True, description="Use pretrained weights")


class TrainingConfig(BaseConfig):
    """Training configuration."""

    epochs: int = Field(default=10, description="Number of training epochs")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    optimizer: str = Field(default="adam", description="Optimizer type")
    scheduler: Optional[str] = Field(default=None, description="LR scheduler")


class ConfigManager:
    """
    Configuration manager supporting both Hydra and Pydantic configs.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config: Optional[DictConfig] = None

    @staticmethod
    def load_yaml(filepath: str) -> DictConfig:
        """
        Load configuration from YAML file.

        Args:
            filepath: Path to YAML file

        Returns:
            Configuration object
        """
        return OmegaConf.load(filepath)

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> DictConfig:
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Configuration object
        """
        return OmegaConf.create(config_dict)

    @staticmethod
    def to_dict(config: DictConfig) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Args:
            config: Configuration object

        Returns:
            Dictionary representation
        """
        return OmegaConf.to_container(config, resolve=True)

    @staticmethod
    def merge_configs(*configs: DictConfig) -> DictConfig:
        """
        Merge multiple configurations.

        Args:
            configs: Configuration objects to merge

        Returns:
            Merged configuration
        """
        return OmegaConf.merge(*configs)

    def save(self, config: DictConfig, filepath: str) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration to save
            filepath: Output file path
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config, filepath)


__all__ = [
    "BaseConfig",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "ConfigManager",
]
