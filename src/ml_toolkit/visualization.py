"""
Visualization utilities for plotting and metrics visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, List, Dict, Any
from pathlib import Path


def set_style(style: str = "whitegrid"):
    """
    Set the plotting style.
    
    Args:
        style: Seaborn style name (default: "whitegrid")
    """
    sns.set_style(style)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None
):
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_history(
    history: Dict[str, List],
    metrics: Optional[List[str]] = None,
    title: str = "Training History",
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None
):
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary with training history
        metrics: List of metrics to plot (plots all if None)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot (optional)
    """
    if metrics is None:
        metrics = list(history.get("metrics", {}).keys())
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    
    epochs = history.get("epochs", [])
    
    for idx, metric in enumerate(metrics):
        values = history.get("metrics", {}).get(metric, [])
        axes[idx].plot(epochs, values, marker='o')
        axes[idx].set_title(f'{metric.capitalize()}')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(metric.capitalize())
        axes[idx].grid(True)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: Optional[int] = None,
    title: str = "Feature Importance",
    figsize: tuple = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot feature importances.
    
    Args:
        feature_names: List of feature names
        importances: Array of feature importance values
        top_n: Number of top features to show (shows all if None)
        title: Plot title
        figsize: Figure size
        save_path: Path to save the plot (optional)
    """
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    if top_n is not None:
        indices = indices[:top_n]
    
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    plt.figure(figsize=figsize)
    plt.barh(range(len(sorted_features)), sorted_importances)
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Importance')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_correlation_matrix(
    data,
    figsize: tuple = (10, 8),
    title: str = "Correlation Matrix",
    save_path: Optional[str] = None
):
    """
    Plot correlation matrix heatmap.
    
    Args:
        data: DataFrame or correlation matrix
        figsize: Figure size
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    import pandas as pd
    
    if isinstance(data, pd.DataFrame):
        corr = data.corr()
    else:
        corr = data
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_distributions(
    data,
    columns: Optional[List[str]] = None,
    figsize: tuple = (15, 10),
    bins: int = 30,
    save_path: Optional[str] = None
):
    """
    Plot distributions of features.
    
    Args:
        data: DataFrame
        columns: List of columns to plot (plots all numeric if None)
        figsize: Figure size
        bins: Number of bins for histograms
        save_path: Path to save the plot (optional)
    """
    import pandas as pd
    
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx, col in enumerate(columns):
        axes[idx].hist(data[col].dropna(), bins=bins, edgecolor='black')
        axes[idx].set_title(col)
        axes[idx].set_xlabel('Value')
        axes[idx].set_ylabel('Frequency')
    
    # Hide unused subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_scatter_matrix(
    data,
    columns: Optional[List[str]] = None,
    figsize: tuple = (12, 12),
    save_path: Optional[str] = None
):
    """
    Plot scatter matrix for feature pairs.
    
    Args:
        data: DataFrame
        columns: List of columns to include (uses all numeric if None)
        figsize: Figure size
        save_path: Path to save the plot (optional)
    """
    import pandas as pd
    
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    pd.plotting.scatter_matrix(data[columns], figsize=figsize, diagonal='hist')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
