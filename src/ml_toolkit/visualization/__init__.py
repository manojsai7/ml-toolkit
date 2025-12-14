"""
Visualization utilities and helpers.
"""

from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training history.

    Args:
        history: Dictionary with 'train_loss' and optionally 'val_loss'
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], "b-", label="Train Loss")

    if "val_loss" in history and history["val_loss"]:
        ax.plot(epochs, history["val_loss"], "r-", label="Val Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot confusion matrix.

    Args:
        cm: Confusion matrix
        class_names: Optional class names
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_metrics_comparison(
    metrics: Dict[str, List[float]],
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot comparison of multiple metrics.

    Args:
        metrics: Dictionary of metric names to values
        labels: Optional labels for x-axis
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    for metric_name, values in metrics.items():
        x = labels if labels else range(len(values))
        ax.plot(x, values, marker="o", label=metric_name)

    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_title("Metrics Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def setup_plotting_style(style: str = "seaborn-v0_8") -> None:
    """
    Setup matplotlib plotting style.

    Args:
        style: Style name
    """
    try:
        plt.style.use(style)
    except (OSError, KeyError):
        # Fallback to seaborn default if style not available
        try:
            sns.set_theme()
        except Exception:
            # Use matplotlib defaults as last resort
            pass

    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["font.size"] = 10


__all__ = [
    "plot_training_history",
    "plot_confusion_matrix",
    "plot_metrics_comparison",
    "setup_plotting_style",
]
