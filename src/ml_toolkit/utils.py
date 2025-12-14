"""
General utility functions for ML projects
"""

import time
import logging
from functools import wraps
from typing import Callable, Any
from pathlib import Path


def setup_logging(
    log_file: str = None,
    level: int = logging.INFO,
    format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
):
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
        format: Log message format
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format,
        handlers=handlers
    )


def timer(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        logging.info(f"{func.__name__} executed in {execution_time:.4f} seconds")
        return result
    
    return wrapper


def create_directory_structure(base_path: str):
    """
    Create standard ML project directory structure.
    
    Args:
        base_path: Base path for the project
    """
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "notebooks",
        "src",
        "results",
        "plots",
        "logs",
        "configs"
    ]
    
    base = Path(base_path)
    for directory in directories:
        (base / directory).mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Created directory structure at {base_path}")


def log_metrics(metrics: dict, prefix: str = ""):
    """
    Log metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for metric names
    """
    logging.info("=" * 50)
    if prefix:
        logging.info(f"{prefix} Metrics:")
    else:
        logging.info("Metrics:")
    logging.info("-" * 50)
    
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            logging.info(f"{metric_name}: {metric_value:.4f}")
        else:
            logging.info(f"{metric_name}: {metric_value}")
    
    logging.info("=" * 50)


class ProgressTracker:
    """
    Simple progress tracker for iterative processes.
    """
    
    def __init__(self, total: int, description: str = "Progress"):
        """
        Initialize progress tracker.
        
        Args:
            total: Total number of items
            description: Description of the process
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """
        Update progress.
        
        Args:
            n: Number of items completed
        """
        self.current += n
        percentage = (self.current / self.total) * 100
        elapsed_time = time.time() - self.start_time
        
        if self.current > 0:
            estimated_total = (elapsed_time / self.current) * self.total
            remaining = estimated_total - elapsed_time
            logging.info(
                f"{self.description}: {self.current}/{self.total} "
                f"({percentage:.1f}%) - "
                f"Elapsed: {elapsed_time:.1f}s, "
                f"Remaining: {remaining:.1f}s"
            )
    
    def close(self):
        """Complete the progress tracking."""
        total_time = time.time() - self.start_time
        logging.info(f"{self.description} completed in {total_time:.2f}s")


def get_random_state(seed: int = None) -> int:
    """
    Get random state for reproducibility.
    
    Args:
        seed: Random seed (uses 42 if None)
        
    Returns:
        Random state value
    """
    return seed if seed is not None else 42
