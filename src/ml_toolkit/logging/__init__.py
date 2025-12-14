"""
Logging and experiment tracking utilities.
"""

from typing import Any, Dict, Optional
import json
from pathlib import Path
from datetime import datetime


class ExperimentLogger:
    """
    Base experiment logger with support for MLflow and W&B.
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = "logs",
        backend: str = "local",
    ):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for logs
            backend: Logging backend ('local', 'mlflow', 'wandb')
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.backend = backend
        self.metrics = {}
        self.params = {}

        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize backend
        self._init_backend()

    def _init_backend(self) -> None:
        """Initialize logging backend."""
        if self.backend == "mlflow":
            try:
                import mlflow

                mlflow.set_experiment(self.experiment_name)
                mlflow.start_run()
            except ImportError:
                print("MLflow not installed, falling back to local logging")
                self.backend = "local"

        elif self.backend == "wandb":
            try:
                import wandb

                wandb.init(project=self.experiment_name)
            except ImportError:
                print("W&B not installed, falling back to local logging")
                self.backend = "local"

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters.

        Args:
            params: Dictionary of parameters
        """
        self.params.update(params)

        if self.backend == "mlflow":
            import mlflow

            mlflow.log_params(params)

        elif self.backend == "wandb":
            import wandb

            wandb.config.update(params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric value.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step/iteration number
        """
        if key not in self.metrics:
            self.metrics[key] = []

        self.metrics[key].append({"value": value, "step": step})

        if self.backend == "mlflow":
            import mlflow

            mlflow.log_metric(key, value, step=step)

        elif self.backend == "wandb":
            import wandb

            wandb.log({key: value}, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log multiple metrics.

        Args:
            metrics: Dictionary of metrics
            step: Optional step/iteration number
        """
        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def log_artifact(self, filepath: str) -> None:
        """
        Log an artifact file.

        Args:
            filepath: Path to artifact file
        """
        if self.backend == "mlflow":
            import mlflow

            mlflow.log_artifact(filepath)

        elif self.backend == "wandb":
            import wandb

            wandb.save(filepath)

    def save_summary(self) -> None:
        """Save experiment summary to file."""
        summary = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "params": self.params,
            "metrics": self.metrics,
        }

        summary_path = self.log_dir / f"{self.experiment_name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

    def finish(self) -> None:
        """Finish logging and cleanup."""
        self.save_summary()

        if self.backend == "mlflow":
            import mlflow

            mlflow.end_run()

        elif self.backend == "wandb":
            import wandb

            wandb.finish()


__all__ = ["ExperimentLogger"]
