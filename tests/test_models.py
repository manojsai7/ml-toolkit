"""
Tests for model utilities
"""

import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_toolkit.models import (
    ModelTracker, save_model, load_model,
    evaluate_classification, evaluate_regression
)
from sklearn.ensemble import RandomForestClassifier


def test_model_tracker():
    """Test ModelTracker functionality."""
    tracker = ModelTracker()
    
    # Log some epochs
    tracker.log_epoch(1, {"loss": 0.5, "accuracy": 0.8})
    tracker.log_epoch(2, {"loss": 0.3, "accuracy": 0.85})
    
    history = tracker.get_history()
    
    assert len(history["epochs"]) == 2
    assert "loss" in history["metrics"]
    assert "accuracy" in history["metrics"]
    assert history["metrics"]["loss"] == [0.5, 0.3]


def test_save_and_load_model():
    """Test saving and loading models."""
    # Create a simple model
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_model.pkl")
        metadata = {"test": "value"}
        save_model(model, filepath, metadata)
        
        # Load model
        loaded_model, loaded_metadata = load_model(filepath)
        
        # Test that loaded model works
        predictions = loaded_model.predict(X[:5])
        assert len(predictions) == 5
        assert loaded_metadata["test"] == "value"


def test_evaluate_classification():
    """Test classification evaluation."""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    
    metrics = evaluate_classification(y_true, y_pred)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_evaluate_regression():
    """Test regression evaluation."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.1])
    
    metrics = evaluate_regression(y_true, y_pred)
    
    assert "mse" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2_score" in metrics
    assert metrics["mse"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
