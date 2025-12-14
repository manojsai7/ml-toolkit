# ML Toolkit

Curated utilities, templates, and best practices for machine learning projects.

[![CI](https://github.com/manojsai7/ml-toolkit/workflows/Lint%20and%20Test/badge.svg)](https://github.com/manojsai7/ml-toolkit/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- 📦 **Data loaders and preprocessing utilities** - Unified interfaces for loading and preprocessing data from multiple formats
- 🚀 **Training/evaluation loops and experiment tracking** - Ready-to-use training loops with built-in logging
- ⚙️ **Reusable configs (Hydra/pydantic)** - Type-safe configuration management with Hydra and Pydantic
- 📊 **Logging/monitoring hooks** - Integration with MLflow and Weights & Biases
- 📓 **Notebook starters and visualization helpers** - Jupyter notebooks and visualization utilities to get started quickly
- 🔧 **CI templates for lint/test/build** - Pre-configured GitHub Actions workflows

## Tech Stack

- **ML Frameworks:** PyTorch, TensorFlow, scikit-learn
- **Config Management:** Hydra, Pydantic, OmegaConf
- **Experiment Tracking:** MLflow, Weights & Biases
- **Development Tools:** Pre-commit hooks, Black, Flake8, MyPy, Pytest

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/manojsai7/ml-toolkit
cd ml-toolkit
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

### Quick Start

#### Basic Training Example

```python
import torch
import torch.nn as nn
from ml_toolkit.training import Trainer
from ml_toolkit.data import CustomDataset

# Create your model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)

# Setup training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

trainer = Trainer(model, optimizer, criterion)

# Train
history = trainer.train(train_loader, val_loader, epochs=10)
```

#### Configuration Management

```python
from ml_toolkit.config import ConfigManager, TrainingConfig

# Use Pydantic models
config = TrainingConfig(
    epochs=50,
    learning_rate=0.001,
    optimizer="adam"
)

# Or load from YAML
config_manager = ConfigManager()
config = config_manager.load_yaml("configs/base_config.yaml")
```

#### Experiment Logging

```python
from ml_toolkit.logging import ExperimentLogger

# Initialize logger (supports MLflow, W&B, or local)
logger = ExperimentLogger("my_experiment", backend="mlflow")

# Log parameters and metrics
logger.log_params({"lr": 0.001, "batch_size": 32})
logger.log_metric("loss", 0.5, step=1)
logger.finish()
```

### Examples

Explore the `examples/` directory for more detailed examples:

- `examples/basic_training.py` - Complete training pipeline
- `examples/data_preprocessing.py` - Data loading and preprocessing
- `examples/config_management.py` - Configuration management examples
- `examples/getting_started.ipynb` - Interactive Jupyter notebook

### Configuration Templates

Pre-configured YAML templates are available in `configs/`:

- `base_config.yaml` - Base configuration template
- `image_classification.yaml` - Image classification settings
- `text_classification.yaml` - Text classification settings

## Project Structure

```
ml-toolkit/
├── src/ml_toolkit/          # Main package
│   ├── data/                # Data loading and preprocessing
│   ├── training/            # Training and evaluation utilities
│   ├── config/              # Configuration management
│   ├── logging/             # Experiment tracking
│   └── visualization/       # Plotting and visualization
├── examples/                # Example scripts and notebooks
├── configs/                 # Configuration templates
├── tests/                   # Unit tests
├── .github/workflows/       # CI/CD pipelines
├── requirements.txt         # Package dependencies
└── pyproject.toml          # Package configuration
```

## Development

### Setup Development Environment

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/ml_toolkit --cov-report=html
```

### Code Quality

```bash
# Format code
black src/

# Lint
flake8 src/

# Type checking
mypy src/
```

## Roadmap

- [ ] Add JAX/Flax examples
- [ ] Add Lightning/FastAI wrappers
- [ ] Template for deployment inference service
- [ ] Enhanced typing + mypy presets
- [ ] Docker containers for reproducible environments
- [ ] Model registry integration
- [ ] Hyperparameter optimization utilities
- [ ] Distributed training support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by best practices from the ML community
- Built with popular open-source ML frameworks
- Designed for ease of use and extensibility
