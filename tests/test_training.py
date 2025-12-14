"""
Tests for training utilities.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml_toolkit.training import Trainer, Evaluator
from ml_toolkit.data import CustomDataset


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    )


@pytest.fixture
def dummy_data():
    """Create dummy data for testing."""
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = CustomDataset(X, y)
    loader = DataLoader(dataset, batch_size=10)
    return loader


def test_trainer_initialization(simple_model):
    """Test Trainer initialization."""
    optimizer = torch.optim.Adam(simple_model.parameters())
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=simple_model,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
    )

    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.criterion is not None
    assert trainer.device == "cpu"


def test_trainer_train_epoch(simple_model, dummy_data):
    """Test training for one epoch."""
    optimizer = torch.optim.Adam(simple_model.parameters())
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=simple_model,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
    )

    loss = trainer.train_epoch(dummy_data)
    assert isinstance(loss, float)
    assert loss > 0


def test_trainer_train(simple_model, dummy_data):
    """Test full training loop."""
    optimizer = torch.optim.Adam(simple_model.parameters())
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=simple_model,
        optimizer=optimizer,
        criterion=criterion,
        device="cpu",
    )

    history = trainer.train(dummy_data, epochs=2)
    assert "train_loss" in history
    assert len(history["train_loss"]) == 2


def test_evaluator(simple_model, dummy_data):
    """Test Evaluator."""
    criterion = nn.CrossEntropyLoss()

    evaluator = Evaluator(
        model=simple_model,
        criterion=criterion,
        device="cpu",
    )

    loss = evaluator.evaluate(dummy_data)
    assert isinstance(loss, float)
    assert loss > 0


def test_evaluator_predict(simple_model, dummy_data):
    """Test prediction."""
    criterion = nn.CrossEntropyLoss()

    evaluator = Evaluator(
        model=simple_model,
        criterion=criterion,
        device="cpu",
    )

    predictions = evaluator.predict(dummy_data)
    assert len(predictions) == 100
    assert len(predictions[0]) == 2
