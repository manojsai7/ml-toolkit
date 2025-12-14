"""
Model utilities for training, evaluation, and model management
"""

import pickle
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)


class ModelTracker:
    """
    Track model training history and metrics.
    """
    
    def __init__(self):
        self.history = {
            "epochs": [],
            "metrics": {}
        }
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for an epoch."""
        self.history["epochs"].append(epoch)
        for metric_name, metric_value in metrics.items():
            if metric_name not in self.history["metrics"]:
                self.history["metrics"][metric_name] = []
            self.history["metrics"][metric_name].append(metric_value)
    
    def get_history(self) -> Dict[str, Any]:
        """Get the complete training history."""
        return self.history
    
    def save_history(self, filepath: str):
        """Save training history to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)


def save_model(model: Any, filepath: str, metadata: Optional[Dict] = None):
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        filepath: Path where to save the model
        metadata: Optional metadata to save with the model
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        "model": model,
        "metadata": metadata or {}
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_dict, f)


def load_model(filepath: str) -> Union[Any, tuple]:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model object or tuple of (model, metadata)
    """
    with open(filepath, 'rb') as f:
        save_dict = pickle.load(f)
    
    if isinstance(save_dict, dict) and "model" in save_dict:
        return save_dict["model"], save_dict.get("metadata", {})
    else:
        # Backward compatibility for models saved without metadata
        return save_dict, {}


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted"
) -> Dict[str, float]:
    """
    Evaluate classification model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging strategy for multi-class metrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    return metrics


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate regression model performance.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2_score": r2_score(y_true, y_pred)
    }
    return metrics


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None
) -> str:
    """
    Generate a detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional list of target class names
        
    Returns:
        Classification report as string
    """
    return classification_report(y_true, y_pred, target_names=target_names)


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Generate confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(y_true, y_pred)


class CrossValidator:
    """
    Utility class for cross-validation.
    """
    
    def __init__(self, model, cv: int = 5):
        """
        Initialize cross-validator.
        
        Args:
            model: Model to cross-validate
            cv: Number of cross-validation folds
        """
        self.model = model
        self.cv = cv
        self.results = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, metric_func):
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
            metric_func: Function to compute metrics
        """
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone model for each fold
            from sklearn.base import clone
            model_clone = clone(self.model)
            
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_val)
            
            metrics = metric_func(y_val, y_pred)
            self.results.append({
                "fold": fold + 1,
                "metrics": metrics
            })
        
        return self.results
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics across all folds."""
        if not self.results:
            return {}
        
        all_metrics = {}
        for result in self.results:
            for metric_name, metric_value in result["metrics"].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_value)
        
        return {
            metric_name: np.mean(values)
            for metric_name, values in all_metrics.items()
        }
