# ML Toolkit

Reusable ML utilities, templates, and best practices to start and standardize projects faster.

## Features

- **Data Utilities**: Load, preprocess, split, and scale datasets
- **Model Management**: Save, load, and track models with metadata
- **Evaluation**: Comprehensive metrics for classification and regression
- **Visualization**: Plot confusion matrices, training history, feature importance, and more
- **Configuration**: Flexible YAML/JSON-based configuration management
- **Templates**: Ready-to-use project templates and examples
- **Best Practices**: Built-in logging, timing, and progress tracking

## Installation

### From Source

```bash
git clone https://github.com/manojsai7/ml-toolkit.git
cd ml-toolkit
pip install -r requirements.txt
pip install -e .
```

### Dependencies

- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- pyyaml >= 5.4.0

## Quick Start

### 1. Data Preparation

```python
from ml_toolkit.data import load_csv, split_data, scale_features

# Load data
data = load_csv("data.csv")

# Split into train/val/test
X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    X, y, test_size=0.2, val_size=0.1, random_state=42
)

# Scale features
X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
    X_train, X_test, X_val, method="standard"
)
```

### 2. Model Training and Evaluation

```python
from ml_toolkit.models import save_model, evaluate_classification
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
metrics = evaluate_classification(y_test, y_pred)
print(metrics)  # {'accuracy': 0.95, 'precision': 0.94, ...}

# Save with metadata
save_model(model, "models/my_model.pkl", metadata={"version": "1.0"})
```

### 3. Visualization

```python
from ml_toolkit.visualization import plot_confusion_matrix, plot_training_history
from ml_toolkit.models import get_confusion_matrix

# Plot confusion matrix
cm = get_confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, class_names=['Class 0', 'Class 1'])

# Plot training history
history = {"epochs": [1, 2, 3], "metrics": {"loss": [0.5, 0.3, 0.2]}}
plot_training_history(history, metrics=["loss"])
```

### 4. Configuration Management

```python
from ml_toolkit.config import Config, create_default_config

# Create default config
config = create_default_config()

# Load from file
config = Config.from_yaml("config.yaml")

# Access values
test_size = config.get("data.test_size")

# Save config
config.save_yaml("output_config.yaml")
```

## Project Templates

### Starting a New ML Project

Use the classification or regression templates to quickly set up a new project:

```bash
# Run classification template
python examples/classification_template.py

# Run regression template
python examples/regression_template.py
```

### Project Structure Template

```
my_ml_project/
├── data/
│   ├── raw/           # Raw, immutable data
│   └── processed/     # Cleaned, processed data
├── models/            # Trained models
├── notebooks/         # Jupyter notebooks
├── src/               # Source code
├── results/           # Model evaluation results
├── plots/             # Visualizations
├── logs/              # Log files
├── configs/           # Configuration files
│   └── config.yaml    # Main configuration
├── requirements.txt   # Dependencies
└── README.md          # Project documentation
```

## Module Documentation

### ml_toolkit.data

Data loading and preprocessing utilities:

- `load_csv()`: Load CSV files
- `split_data()`: Split data into train/val/test sets
- `scale_features()`: Scale features using StandardScaler or MinMaxScaler
- `handle_missing_values()`: Handle missing values with various strategies
- `get_feature_target_split()`: Split DataFrame into features and target

### ml_toolkit.models

Model training, evaluation, and management:

- `ModelTracker`: Track training history and metrics
- `save_model()`: Save models with metadata
- `load_model()`: Load saved models
- `evaluate_classification()`: Comprehensive classification metrics
- `evaluate_regression()`: Comprehensive regression metrics
- `CrossValidator`: Cross-validation utility

### ml_toolkit.visualization

Plotting and visualization utilities:

- `plot_confusion_matrix()`: Visualize confusion matrices
- `plot_training_history()`: Plot training metrics over epochs
- `plot_feature_importance()`: Display feature importances
- `plot_correlation_matrix()`: Correlation heatmap
- `plot_distributions()`: Feature distribution plots
- `set_style()`: Set plotting style

### ml_toolkit.config

Configuration management:

- `Config`: Configuration class with get/set/update methods
- `create_default_config()`: Generate default ML project config
- Load/save configs in YAML or JSON format

### ml_toolkit.utils

General utilities:

- `setup_logging()`: Configure logging
- `timer`: Decorator to measure execution time
- `create_directory_structure()`: Create standard project structure
- `ProgressTracker`: Track progress of long-running operations
- `log_metrics()`: Pretty-print metrics

## Examples

### Complete Classification Pipeline

```python
from ml_toolkit.data import split_data, scale_features
from ml_toolkit.models import save_model, evaluate_classification, ModelTracker
from ml_toolkit.visualization import plot_confusion_matrix
from ml_toolkit.utils import setup_logging, timer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Setup
setup_logging(log_file="logs/training.log")

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Split and scale
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# Train
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
metrics = evaluate_classification(y_test, y_pred)
print(f"Test Accuracy: {metrics['accuracy']:.4f}")

# Save
save_model(model, "models/classifier.pkl", metadata={"metrics": metrics})
```

## Testing

Run tests with pytest:

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=ml_toolkit --cov-report=html
```

## Best Practices

1. **Reproducibility**: Always set random seeds
2. **Configuration**: Use config files for hyperparameters
3. **Logging**: Use logging instead of print statements
4. **Data Management**: Keep raw and processed data separate
5. **Model Versioning**: Save models with metadata
6. **Documentation**: Document experiments and results
7. **Testing**: Write tests for critical functionality

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Author

ML Toolkit Contributors

## Changelog

### 0.1.0 (Initial Release)

- Core data utilities (loading, preprocessing, splitting)
- Model management utilities (save, load, evaluate)
- Visualization tools
- Configuration management
- Project templates
- Comprehensive test suite
