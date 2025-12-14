"""
Tests for data utilities
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ml_toolkit.data import (
    split_data, scale_features, handle_missing_values,
    get_feature_target_split
)


def test_split_data_basic():
    """Test basic train-test split."""
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_split_data_with_validation():
    """Test train-val-test split."""
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y, test_size=0.2, val_size=0.1
    )
    
    assert len(X_train) == 70
    assert len(X_val) == 10
    assert len(X_test) == 20


def test_scale_features_standard():
    """Test standard scaling."""
    X_train = np.random.rand(100, 5)
    X_test = np.random.rand(20, 5)
    
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, method="standard"
    )
    
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-7)


def test_scale_features_minmax():
    """Test minmax scaling."""
    X_train = np.random.rand(100, 5)
    X_test = np.random.rand(20, 5)
    
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, method="minmax"
    )
    
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert X_train_scaled.min() >= -1e-10  # Allow for floating point precision
    assert X_train_scaled.max() <= 1 + 1e-10  # Allow for floating point precision


def test_handle_missing_values_mean():
    """Test handling missing values with mean strategy."""
    df = pd.DataFrame({
        'a': [1, 2, np.nan, 4],
        'b': [5, np.nan, 7, 8],
        'c': ['x', 'y', 'z', 'w']
    })
    
    result = handle_missing_values(df, strategy="mean")
    
    assert not result['a'].isna().any()
    assert not result['b'].isna().any()
    assert np.isclose(result['a'].iloc[2], df['a'].mean())


def test_handle_missing_values_drop():
    """Test handling missing values with drop strategy."""
    df = pd.DataFrame({
        'a': [1, 2, np.nan, 4],
        'b': [5, 6, 7, 8]
    })
    
    result = handle_missing_values(df, strategy="drop")
    
    assert len(result) == 3
    assert not result['a'].isna().any()


def test_get_feature_target_split():
    """Test splitting DataFrame into features and target."""
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [0, 1, 0]
    })
    
    X, y = get_feature_target_split(df, 'target')
    
    assert 'target' not in X.columns
    assert 'feature1' in X.columns
    assert 'feature2' in X.columns
    assert len(y) == 3
