"""
Data utilities for loading, preprocessing, and splitting datasets
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        **kwargs: Additional arguments to pass to pd.read_csv
        
    Returns:
        DataFrame containing the loaded data
    """
    return pd.read_csv(filepath, **kwargs)


def split_data(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    random_state: int = 42,
    stratify: bool = False
) -> Union[Tuple, Tuple]:
    """
    Split data into train, validation (optional), and test sets.
    
    Args:
        X: Features
        y: Target variable
        test_size: Proportion of data for test set (default: 0.2)
        val_size: Proportion of data for validation set (default: None)
        random_state: Random seed for reproducibility
        stratify: Whether to use stratified splitting based on y
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) or
        (X_train, X_val, X_test, y_train, y_val, y_test) if val_size is specified
    """
    stratify_param = y if stratify else None
    
    if val_size is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        return X_train, X_test, y_train, y_test
    else:
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        # Second split: separate validation set from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        stratify_temp = y_temp if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=stratify_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    method: str = "standard"
) -> Tuple:
    """
    Scale features using StandardScaler or MinMaxScaler.
    
    Args:
        X_train: Training features
        X_test: Test features (optional)
        X_val: Validation features (optional)
        method: Scaling method - "standard" or "minmax" (default: "standard")
        
    Returns:
        Tuple of scaled arrays and the fitted scaler
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}. Use 'standard' or 'minmax'")
    
    X_train_scaled = scaler.fit_transform(X_train)
    
    result = [X_train_scaled]
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        result.append(X_test_scaled)
    
    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        result.append(X_val_scaled)
    
    result.append(scaler)
    
    return tuple(result)


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "mean",
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame.
    
    Args:
        df: Input DataFrame
        strategy: Strategy for handling missing values - "mean", "median", "mode", or "drop"
        columns: Specific columns to handle (default: all numeric columns)
        
    Returns:
        DataFrame with missing values handled
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    if strategy == "drop":
        df_copy = df_copy.dropna(subset=columns)
    elif strategy == "mean":
        df_copy[columns] = df_copy[columns].fillna(df_copy[columns].mean())
    elif strategy == "median":
        df_copy[columns] = df_copy[columns].fillna(df_copy[columns].median())
    elif strategy == "mode":
        df_copy[columns] = df_copy[columns].fillna(df_copy[columns].mode().iloc[0])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_copy


def get_feature_target_split(
    df: pd.DataFrame,
    target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target.
    
    Args:
        df: Input DataFrame
        target_column: Name of the target column
        
    Returns:
        Tuple of (features, target)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
