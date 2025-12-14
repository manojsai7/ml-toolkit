"""
Example demonstrating configuration management with Hydra.
"""

from ml_toolkit.config import ConfigManager, DataConfig, ModelConfig, TrainingConfig


def main():
    print("Configuration Management Example\n")

    # Create configs using Pydantic models
    data_config = DataConfig(
        data_dir="data/images",
        batch_size=64,
        num_workers=8,
    )

    model_config = ModelConfig(
        model_name="resnet50",
        num_classes=100,
        pretrained=True,
    )

    training_config = TrainingConfig(
        epochs=50,
        learning_rate=0.001,
        optimizer="adam",
    )

    print("Data Config:")
    print(data_config.model_dump())

    print("\nModel Config:")
    print(model_config.model_dump())

    print("\nTraining Config:")
    print(training_config.model_dump())

    # Load config from YAML
    print("\n" + "=" * 50)
    print("Loading config from YAML file")

    config_manager = ConfigManager()

    try:
        config = config_manager.load_yaml("configs/base_config.yaml")
        print("\nLoaded config:")
        print(config_manager.to_dict(config))
    except FileNotFoundError:
        print("\nConfig file not found. Run from repository root.")

    # Merge configurations
    print("\n" + "=" * 50)
    print("Merging configurations")

    config_dict = {
        "model": {"model_name": "vgg16"},
        "training": {"epochs": 100},
    }

    base_config = config_manager.from_dict(config_dict)
    override_config = config_manager.from_dict({"training": {"epochs": 200}})

    merged = config_manager.merge_configs(base_config, override_config)
    print("\nMerged config:")
    print(config_manager.to_dict(merged))


if __name__ == "__main__":
    main()
