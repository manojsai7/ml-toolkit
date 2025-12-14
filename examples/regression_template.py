"""
Basic Regression Template
A template for building a regression model using ml-toolkit utilities.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_toolkit.data import split_data, scale_features
from ml_toolkit.models import save_model, evaluate_regression
from ml_toolkit.config import create_default_config
from ml_toolkit.utils import setup_logging, timer, log_metrics

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression


@timer
def prepare_data(config):
    """Prepare and split data."""
    # For demonstration, generate synthetic data
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        noise=10.0,
        random_state=config.get("data.random_state")
    )
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        test_size=config.get("data.test_size"),
        val_size=config.get("data.val_size"),
        random_state=config.get("data.random_state")
    )
    
    # Scale features
    X_train_scaled, X_test_scaled, X_val_scaled, scaler = scale_features(
        X_train, X_test, X_val,
        method=config.get("preprocessing.scaling_method")
    )
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test


@timer
def train_model(X_train, y_train, config):
    """Train the model."""
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=config.get("model.random_state")
    )
    
    model.fit(X_train, y_train)
    return model


def main():
    """Main execution function."""
    # Setup
    setup_logging(log_file="logs/regression.log")
    config = create_default_config()
    
    # Prepare data
    print("\n" + "="*50)
    print("PREPARING DATA")
    print("="*50)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(config)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    print("\n" + "="*50)
    print("TRAINING MODEL")
    print("="*50)
    model = train_model(X_train, y_train, config)
    
    # Evaluate on validation set
    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    y_val_pred = model.predict(X_val)
    val_metrics = evaluate_regression(y_val, y_val_pred)
    log_metrics(val_metrics, "Validation")
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    y_test_pred = model.predict(X_test)
    test_metrics = evaluate_regression(y_test, y_test_pred)
    log_metrics(test_metrics, "Test")
    
    # Save model
    print("\n" + "="*50)
    print("SAVING MODEL")
    print("="*50)
    metadata = {
        "model_type": "RandomForestRegressor",
        "test_metrics": test_metrics,
        "config": config.to_dict()
    }
    save_model(model, "models/regression_model.pkl", metadata)
    print("Model saved to models/regression_model.pkl")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()
