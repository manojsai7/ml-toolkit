# ML Project Best Practices

This document outlines best practices for machine learning projects using ml-toolkit.

## 1. Project Organization

### Directory Structure

Always use a consistent directory structure:

```
my_project/
├── data/
│   ├── raw/              # Original, immutable data
│   └── processed/        # Cleaned and transformed data
├── models/               # Trained models with metadata
├── notebooks/            # Exploratory Jupyter notebooks
├── src/                  # Source code
│   ├── __init__.py
│   ├── data.py          # Data processing scripts
│   ├── models.py        # Model definitions
│   └── train.py         # Training scripts
├── configs/              # Configuration files
│   └── config.yaml
├── results/              # Model outputs and metrics
├── plots/                # Visualizations
├── logs/                 # Log files
├── tests/                # Unit tests
├── requirements.txt      # Dependencies
└── README.md            # Project documentation
```

## 2. Reproducibility

### Set Random Seeds

Always set random seeds for reproducibility:

```python
import numpy as np
import random
from ml_toolkit.utils import get_random_state

RANDOM_STATE = get_random_state(42)

# Set seeds
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
```

### Version Everything

- **Code**: Use Git for version control
- **Data**: Track data versions or use DVC
- **Models**: Save models with metadata including training date, hyperparameters, and metrics
- **Environment**: Pin package versions in requirements.txt

## 3. Configuration Management

### Use Configuration Files

Store hyperparameters and settings in config files:

```yaml
# config.yaml
model:
  type: "random_forest"
  n_estimators: 100
  max_depth: 10
  random_state: 42

data:
  train_path: "data/train.csv"
  test_size: 0.2
  val_size: 0.1
```

Load in code:

```python
from ml_toolkit.config import Config

config = Config.from_yaml("configs/config.yaml")
n_estimators = config.get("model.n_estimators")
```

## 4. Data Management

### Keep Raw Data Immutable

- Never modify raw data files
- Store transformations as code
- Save processed data separately

### Use ML Toolkit Data Utilities

```python
from ml_toolkit.data import load_csv, split_data, scale_features

# Load and split
data = load_csv("data/train.csv")
X_train, X_test, y_train, y_test = split_data(X, y)

# Scale features
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
```

## 5. Model Development

### Start Simple

1. Begin with a simple baseline model
2. Establish performance metrics
3. Iteratively improve

### Track Experiments

Use ml-toolkit's ModelTracker:

```python
from ml_toolkit.models import ModelTracker

tracker = ModelTracker()
tracker.log_epoch(1, {"loss": 0.5, "accuracy": 0.8})
tracker.save_history("results/training_history.json")
```

### Validate Properly

- Use cross-validation
- Hold out a test set for final evaluation
- Watch for data leakage

```python
from ml_toolkit.models import CrossValidator
from ml_toolkit.models import evaluate_classification

cv = CrossValidator(model, cv=5)
results = cv.fit(X, y, metric_func=lambda y_true, y_pred: evaluate_classification(y_true, y_pred))
avg_metrics = cv.get_average_metrics()
```

## 6. Code Quality

### Write Docstrings

```python
def train_model(X_train, y_train, config):
    """
    Train a machine learning model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        config: Configuration object
        
    Returns:
        Trained model instance
    """
    # Training code
    return model
```

### Add Tests

Write tests for critical functionality and run them:

```bash
pytest tests/
```

## 7. Logging and Monitoring

### Use Logging Instead of Print

```python
from ml_toolkit.utils import setup_logging, log_metrics

setup_logging(log_file="logs/training.log")

# Log metrics
metrics = evaluate_classification(y_test, y_pred)
log_metrics(metrics, prefix="Test")
```

### Monitor Training

Track progress:

```python
from ml_toolkit.utils import ProgressTracker

tracker = ProgressTracker(total=num_epochs, description="Training")
for epoch in range(num_epochs):
    # Training code
    tracker.update(1)
tracker.close()
```

## 8. Model Evaluation

### Use Multiple Metrics

```python
from ml_toolkit.models import evaluate_classification, evaluate_regression

# For classification
metrics = evaluate_classification(y_test, y_pred)

# For regression
metrics = evaluate_regression(y_test, y_pred)
```

### Visualize Results

```python
from ml_toolkit.visualization import plot_confusion_matrix, plot_feature_importance
from ml_toolkit.models import get_confusion_matrix

# Confusion matrix
cm = get_confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, class_names=['Class 0', 'Class 1'])

# Feature importance
plot_feature_importance(feature_names, importances, top_n=10)
```

## 9. Model Deployment

### Save Models with Metadata

```python
from ml_toolkit.models import save_model

metadata = {
    "model_type": "RandomForest",
    "train_date": "2025-01-01",
    "metrics": test_metrics,
    "features": feature_names,
    "version": "1.0.0"
}

save_model(model, "models/model_v1.pkl", metadata=metadata)
```

## 10. Documentation

### Project README

Include:
- Project description
- Setup instructions
- Usage examples
- Results summary
- Contact information

## Summary Checklist

Before finalizing a project, check:

- [ ] Code is version controlled with Git
- [ ] Random seeds are set for reproducibility
- [ ] Configuration files are used for hyperparameters
- [ ] Raw data is kept separate and immutable
- [ ] Models are saved with metadata
- [ ] Tests are written for critical functions
- [ ] Logging is used instead of print statements
- [ ] Multiple evaluation metrics are reported
- [ ] Results are visualized
- [ ] Documentation is complete
