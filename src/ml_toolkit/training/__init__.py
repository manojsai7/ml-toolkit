"""
Training and evaluation utilities.
"""

from typing import Any, Callable, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    Generic training loop for PyTorch models.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cpu",
        callbacks: Optional[list] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: PyTorch model to train
            optimizer: Optimizer instance
            criterion: Loss function
            device: Device to use ('cpu' or 'cuda')
            callbacks: List of callback functions
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.callbacks = callbacks or []
        self.history = {"train_loss": [], "val_loss": []}

    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc="Training"):
            # Move batch to device
            inputs = batch["data"].to(self.device)
            targets = batch["label"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
    ) -> Dict[str, list]:
        """
        Train model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train

        Returns:
            Training history dictionary
        """
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")

            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history["val_loss"].append(val_loss)
                print(f"Val Loss: {val_loss:.4f}")

            # Run callbacks
            for callback in self.callbacks:
                callback(epoch, self.history)

        return self.history

    def validate(self, dataloader: DataLoader) -> float:
        """
        Validate model.

        Args:
            dataloader: Validation data loader

        Returns:
            Average validation loss
        """
        evaluator = Evaluator(self.model, self.criterion, self.device)
        return evaluator.evaluate(dataloader)


class Evaluator:
    """
    Model evaluation utilities.
    """

    def __init__(
        self, model: nn.Module, criterion: nn.Module, device: str = "cpu"
    ):
        """
        Initialize evaluator.

        Args:
            model: PyTorch model to evaluate
            criterion: Loss function
            device: Device to use
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device

    def evaluate(self, dataloader: DataLoader) -> float:
        """
        Evaluate model on data.

        Args:
            dataloader: Data loader

        Returns:
            Average loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                inputs = batch["data"].to(self.device)
                targets = batch["label"].to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def predict(self, dataloader: DataLoader) -> list:
        """
        Generate predictions.

        Args:
            dataloader: Data loader

        Returns:
            List of predictions
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                inputs = batch["data"].to(self.device)
                outputs = self.model(inputs)
                predictions.extend(outputs.cpu().numpy())

        return predictions


__all__ = ["Trainer", "Evaluator"]
