"""
Basic training example using ML Toolkit.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml_toolkit.data import CustomDataset
from ml_toolkit.training import Trainer
from ml_toolkit.config import ConfigManager, TrainingConfig
from ml_toolkit.logging import ExperimentLogger


def create_simple_model(input_size: int, output_size: int) -> nn.Module:
    """Create a simple neural network model."""
    return nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, output_size),
    )


def main():
    # Configuration
    config = TrainingConfig(
        epochs=10,
        learning_rate=0.001,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"Training configuration: {config}")

    # Generate dummy data
    n_samples = 1000
    input_size = 20
    output_size = 3

    X_train = torch.randn(n_samples, input_size)
    y_train = torch.randint(0, output_size, (n_samples,))

    # Create dataset and dataloader
    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Create model
    model = create_simple_model(input_size, output_size)

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Initialize logger
    logger = ExperimentLogger("basic_training_example", backend="local")
    logger.log_params(config.dict())

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=config.device,
    )

    # Train model
    print("\nStarting training...")
    history = trainer.train(train_loader, epochs=config.epochs)

    # Log final metrics
    logger.log_metrics(
        {
            "final_train_loss": history["train_loss"][-1],
        }
    )
    logger.finish()

    print("\nTraining completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
